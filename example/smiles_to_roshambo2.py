"""
smiles_to_roshambo2.py

Takes query molecules (from a CSV with SDF paths and UIDs) and a dataset
(either a pre-built H5 or a CSV of SMILES), and runs roshambo2 molecular
shape overlay calculation.

Optimized for large-scale runs (e.g. 10K queries × 600 dataset molecules):
  - 10K query SDFs are loaded with a progress bar and conformers are generated
    in chunks through nvMolKit to avoid GPU OOM.
  - Queries are scored against the dataset in batches (--query_batch_size) to
    cap peak memory.  The dataset H5 is loaded once per batch; since the
    reference library is small it stays fast.
  - Uses CUDA backend by default for GPU-accelerated overlay.
  - Results are saved incrementally per batch (no full accumulation in RAM).
  - SDF blocks are optional (--save_sdf) — skipping them saves ~10x memory.
  - Conformer reduction is done by roshambo2 internally (reduce_over_conformers).
  - max_results caps per-query hits to avoid unbounded memory growth.

--dataset_h5 specifies the path to the H5 dataset file. If the file already
exists it is loaded directly. If it does not exist, one of the following
must be provided:
  - --dataset_sdf: One or more SDF files with pre-computed 3D poses.
    Molecules are loaded with their existing coordinates (no conformer
    generation). Pass multiple paths to combine several SDF files.
  - --dataset_csv: A CSV of SMILES. Conformers are generated via nvMolKit.
--dataset_sdf takes priority over --dataset_csv. The resulting H5 is saved
to the --dataset_h5 path for future reuse.

Uses nvMolKit for:
  - ETKDG conformer embedding on GPU (EmbedMolecules)
  - MMFF energy minimization on GPU (MMFFOptimizeMoleculesConfs)

Query CSV format (--query_csv) — SDF mode (default):
    uid,sdf_path
    QUERY_001,/path/to/query1.sdf
    QUERY_002,/path/to/query2.sdf

Query CSV format (--query_csv) — SMILES mode (--query_smiles_col):
    uid,smiles
    QUERY_001,CCO
    QUERY_002,c1ccccc1

Dataset CSV format (--dataset_csv):
    smiles
    CCO
    CCCO
    CCCCO
    (or with a name column: smiles,name)

Usage:
    # Use a pre-built H5 dataset
    python smiles_to_roshambo2.py --query_csv queries.csv --dataset_h5 dataset.h5

    # Use pre-computed SDF poses as reference (no conformer generation)
    python smiles_to_roshambo2.py --query_csv queries.csv \\
        --dataset_h5 /data/dataset.h5 --dataset_sdf docked_poses.sdf

    # Generate H5 from a CSV of SMILES (H5 is saved and reused on next run)
    python smiles_to_roshambo2.py --query_csv queries.csv \\
        --dataset_h5 /data/dataset.h5 --dataset_csv dataset.csv

    # Large-scale run: 10K queries, 600 ref dataset, CUDA, batched
    python smiles_to_roshambo2.py --query_csv queries.csv --dataset_h5 dataset.h5 \\
        --backend cuda --query_batch_size 200 --max_results 100
"""

import argparse
import gc
import json
import os
import sys
import time

import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier, rdmolops
from rdkit.Chem.rdDistGeom import ETKDGv3
from tqdm import tqdm


from roshambo2 import Roshambo2
from roshambo2.prepare import prepare_from_rdkitmols


# ---------------------------------------------------------------------------
# Molecule preparation helpers
# ---------------------------------------------------------------------------

def canonicalize_smiles(smiles):
    """Return the canonical SMILES for a molecule, or None if invalid.

    Round-trips through RDKit to guarantee a reproducible canonical form
    regardless of the input representation.
    """
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def prepare_mol(smiles, name=None):
    """Create an RDKit molecule (with hydrogens) from a canonical SMILES.

    Does NOT generate conformers — that is done in batch via nvMolKit.
    The input SMILES is canonicalized before use.

    Args:
        smiles (str): SMILES string.
        name (str, optional): Molecule name. If None, uses the canonical SMILES.

    Returns:
        rdkit.Chem.Mol or None: Molecule with explicit Hs (no conformers yet),
            or None if SMILES parsing failed.
    """
    smi = canonicalize_smiles(smiles.strip())
    if smi is None:
        print(f"  [WARNING] Failed to parse SMILES: {smiles.strip()}")
        return None

    if name is None:
        name = smi

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    mol.SetProp("_Name", name)
    return mol


def embed_and_optimize(mols, n_confs=5, hardware_options=None, mmff_max_iters=500,
                       chunk_size=2000):
    """Generate conformers and MMFF-optimize them, in chunks if needed.

    For large molecule lists (e.g. 10K queries) the GPU call is split into
    chunks of *chunk_size* molecules to avoid GPU OOM.  Results are
    concatenated transparently.

    Attempts to use nvMolKit for GPU-accelerated embedding and MMFF
    optimization.  If nvMolKit is not installed, falls back to RDKit
    CPU-based methods.

    Args:
        mols (list[rdkit.Chem.Mol]): Molecules (with Hs, no conformers).
        n_confs (int): Conformers to generate per molecule.
        hardware_options: nvMolKit HardwareOptions (ignored in RDKit fallback).
        mmff_max_iters (int): Max MMFF iterations.
        chunk_size (int): Max molecules per nvMolKit GPU call (default: 2000).

    Returns:
        list[rdkit.Chem.Mol]: Molecules with conformers (failed ones removed).
    """
    if not mols:
        return []

    try:
        import nvmolkit.embedMolecules as nv_embed
        from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
        from nvmolkit.types import HardwareOptions
    except ImportError:
        nv_embed = None

    if nv_embed is not None:
        # ---- GPU path (nvMolKit), chunked ----

        # Verify CUDA / GPU availability
        _gpu_ok = False
        try:
            import torch
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
                print(f"  [GPU CHECK] CUDA available — {n_gpus} GPU(s): {', '.join(gpu_names)}")
                _gpu_ok = True
            else:
                print("  [GPU CHECK] torch.cuda.is_available() = False")
        except ImportError:
            # torch not installed; try nvidia-smi as fallback
            import subprocess
            try:
                smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                )
                if smi.returncode == 0 and smi.stdout.strip():
                    print(f"  [GPU CHECK] nvidia-smi reports:\n{smi.stdout.strip()}")
                    _gpu_ok = True
                else:
                    print(f"  [GPU CHECK] nvidia-smi returned code {smi.returncode}")
            except FileNotFoundError:
                print("  [GPU CHECK] Neither torch nor nvidia-smi found — cannot verify GPU")
            except Exception as e:
                print(f"  [GPU CHECK] nvidia-smi error: {e}")

        if not _gpu_ok:
            print("  [WARNING] No GPU detected — nvMolKit may fail or run on CPU."
                  "  Consider using --backend cpp if this is intentional.")

        if hardware_options is None:
            from nvmolkit.types import HardwareOptions
            hardware_options = HardwareOptions()

        params = ETKDGv3()
        params.useRandomCoords = True  # Required for nvMolKit ETKDG

        n_total = len(mols)
        n_chunks = (n_total + chunk_size - 1) // chunk_size
        print(f"  [INFO] Using nvMolKit GPU for {n_total} molecule(s) × {n_confs} confs"
              f"  ({n_chunks} chunk(s) of ≤{chunk_size})")

        all_successful = []
        for ci in range(n_chunks):
            start = ci * chunk_size
            end = min(start + chunk_size, n_total)
            chunk = mols[start:end]

            if n_chunks > 1:
                print(f"    Chunk {ci + 1}/{n_chunks}: molecules {start}–{end - 1}")

            nv_embed.EmbedMolecules(
                chunk, params, confsPerMolecule=n_confs, hardwareOptions=hardware_options
            )
            good = _filter_embedded(chunk)
            if good:
                # Pre-filter: separate molecules MMFF can handle from those it can't
                from rdkit.Chem import rdForceFieldHelpers
                mmff_ok = []
                mmff_skip = []
                for mol in good:
                    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
                        mmff_ok.append(mol)
                    else:
                        mmff_skip.append(mol)

                if mmff_skip:
                    from rdkit.Chem import AllChem
                    n_uff_ok = 0
                    n_uff_fail = 0
                    for mol in mmff_skip:
                        label = mol.GetProp("uid") if mol.HasProp("uid") else (
                            mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown")
                        # Fallback to UFF for MMFF-incompatible molecules
                        uff_success = False
                        for cid in range(mol.GetNumConformers()):
                            try:
                                AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=mmff_max_iters)
                                uff_success = True
                            except Exception:
                                pass
                        if uff_success:
                            n_uff_ok += 1
                        else:
                            n_uff_fail += 1
                            print(f"    [WARNING] Both MMFF and UFF failed for {label} "
                                  f"— keeping un-optimized conformers")
                    print(f"  [INFO] MMFF pre-check: {len(mmff_ok)} MMFF, "
                          f"{n_uff_ok} UFF fallback, {n_uff_fail} un-optimized")

                # Optimize the ones MMFF supports
                if mmff_ok:
                    try:
                        MMFFOptimizeMoleculesConfs(
                            mmff_ok,
                            maxIters=mmff_max_iters,
                            hardwareOptions=hardware_options,
                        )
                    except RuntimeError as e:
                        print(f"  [WARNING] Batch MMFF failed unexpectedly ({e}) "
                              f"— keeping un-optimized conformers")

                # All molecules proceed (optimized or not)
                all_successful.extend(mmff_ok)
                all_successful.extend(mmff_skip)

        return all_successful
    else:
        # ---- CPU fallback (RDKit) ----
        print(f"  [INFO] nvMolKit not found — falling back to RDKit CPU "
              f"for {len(mols)} molecule(s) × {n_confs} confs")
        from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

        successful_mols = []
        for mol in tqdm(mols, desc="Generating conformers (RDKit)"):
            params = ETKDGv3()
            params.randomSeed = 42
            conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
            if len(conf_ids) == 0:
                label = mol.GetProp("uid") if mol.HasProp("uid") else (
                    mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown")
                print(f"  [WARNING] Conformer embedding failed for: {label}")
                continue

            # Try MMFF first; fall back to UFF if MMFF lacks atom type params
            use_uff = not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
            for cid in conf_ids:
                try:
                    if use_uff:
                        AllChem.UFFOptimizeMolecule(mol, confId=cid, maxIters=mmff_max_iters)
                    else:
                        AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=mmff_max_iters)
                except Exception:
                    pass
            successful_mols.append(mol)
        return successful_mols


def _filter_embedded(mols):
    """Keep only molecules that received at least one conformer."""
    good = []
    for mol in mols:
        if mol.GetNumConformers() > 0:
            good.append(mol)
        else:
            label = mol.GetProp("uid") if mol.HasProp("uid") else (
                mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown")
            print(f"  [WARNING] Conformer embedding failed for: {label}")
    return good


def get_largest_fragment(mol):
    """Return the largest fragment of a molecule by heavy atom count.

    If the molecule has only one fragment, returns it unchanged.
    Preserves 3D coordinates, properties, and the _Name field.

    Args:
        mol (rdkit.Chem.Mol): Input molecule (may contain multiple fragments).

    Returns:
        rdkit.Chem.Mol: The largest fragment with all original properties.
    """
    frags = rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return mol

    # Pick fragment with the most heavy atoms
    largest = max(frags, key=lambda f: f.GetNumHeavyAtoms())

    # GetMolFrags doesn't carry over properties — copy them from the original
    for prop_name in mol.GetPropsAsDict():
        if not largest.HasProp(prop_name):
            largest.SetProp(prop_name, mol.GetProp(prop_name))

    return largest


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def load_queries_from_csv(csv_path, uid_col="uid", sdf_col="sdf_path"):
    """Load query molecules from a CSV file with SDF paths and UIDs.

    Includes a tqdm progress bar so that loading 10K+ SDF files shows
    progress.

    Args:
        csv_path (str): Path to the CSV file.
        uid_col (str): Column name for the unique identifier.
        sdf_col (str): Column name for the SDF file path.

    Returns:
        tuple[list[rdkit.Chem.Mol], dict[str, str]]:
            - list of cleaned RDKit mol objects (each carries a ``uid`` prop)
            - dict mapping roshambo2 internal name → UID
    """
    query_df = pd.read_csv(csv_path)

    if uid_col not in query_df.columns:
        raise ValueError(f"Column '{uid_col}' not found in {csv_path}. "
                         f"Available columns: {list(query_df.columns)}")
    if sdf_col not in query_df.columns:
        raise ValueError(f"Column '{sdf_col}' not found in {csv_path}. "
                         f"Available columns: {list(query_df.columns)}")

    query_mols = []
    name_to_uid = {}
    n_skipped = 0
    n_defragged = 0

    for _, row in tqdm(query_df.iterrows(), total=len(query_df),
                       desc="  Loading query SDFs"):
        uid = str(row[uid_col])
        sdf_path = str(row[sdf_col])

        if not os.path.isfile(sdf_path):
            n_skipped += 1
            continue

        with SDMolSupplier(sdf_path, sanitize=True, removeHs=False) as supplier:
            for mol in supplier:
                if mol is None:
                    n_skipped += 1
                    continue

                # Keep largest fragment if multi-fragment
                num_frags = len(rdmolops.GetMolFrags(mol))
                if num_frags > 1:
                    mol = get_largest_fragment(mol)
                    n_defragged += 1

                mol = Chem.AddHs(mol, addCoords=True)
                mol.SetProp("uid", uid)
                mol.SetProp("_Name", uid)  # roshambo2 keys by _Name
                name_to_uid[uid] = uid
                query_mols.append(mol)

    if n_defragged > 0:
        print(f"  [INFO] Extracted largest fragment for {n_defragged} molecule(s)")
    if n_skipped > 0:
        print(f"  [INFO] Skipped {n_skipped} invalid/missing entries")

    return query_mols, name_to_uid


def load_queries_from_smiles_csv(csv_path, uid_col="uid", smiles_col="smiles"):
    """Load query molecules from a CSV file with SMILES and UIDs.

    Unlike ``load_queries_from_csv`` (which reads SDF files), this function
    parses SMILES strings.  The returned molecules have explicit Hs but
    **no conformers** — conformers must be generated afterwards via
    ``embed_and_optimize``.

    Args:
        csv_path (str): Path to the CSV file.
        uid_col (str): Column name for the unique identifier.
        smiles_col (str): Column name for the SMILES strings.

    Returns:
        tuple[list[rdkit.Chem.Mol], dict[str, str]]:
            - list of RDKit mol objects (each carries a ``uid`` prop, no conformers)
            - dict mapping roshambo2 internal name → UID
    """
    query_df = pd.read_csv(csv_path)

    if uid_col not in query_df.columns:
        raise ValueError(f"Column '{uid_col}' not found in {csv_path}. "
                         f"Available columns: {list(query_df.columns)}")
    if smiles_col not in query_df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {csv_path}. "
                         f"Available columns: {list(query_df.columns)}")

    query_mols = []
    name_to_uid = {}
    n_skipped = 0
    n_defragged = 0

    for _, row in tqdm(query_df.iterrows(), total=len(query_df),
                       desc="  Loading query SMILES"):
        uid = str(row[uid_col])
        raw_smi = str(row[smiles_col]).strip()

        if not raw_smi or raw_smi == "nan":
            n_skipped += 1
            continue

        smi = canonicalize_smiles(raw_smi)
        if smi is None:
            print(f"  [WARNING] Failed to parse SMILES for {uid}: {raw_smi}")
            n_skipped += 1
            continue

        mol = Chem.MolFromSmiles(smi)

        # Keep largest fragment if multi-fragment
        num_frags = len(rdmolops.GetMolFrags(mol))
        if num_frags > 1:
            mol = get_largest_fragment(mol)
            n_defragged += 1

        mol = Chem.AddHs(mol)
        mol.SetProp("uid", uid)
        mol.SetProp("_Name", uid)  # roshambo2 keys by _Name
        name_to_uid[uid] = uid
        query_mols.append(mol)

    if n_defragged > 0:
        print(f"  [INFO] Extracted largest fragment for {n_defragged} molecule(s)")
    if n_skipped > 0:
        print(f"  [INFO] Skipped {n_skipped} invalid/empty SMILES entries")

    return query_mols, name_to_uid


def load_dataset_from_sdf(sdf_paths):
    """Load reference dataset molecules from one or more SDF files.

    Unlike ``load_dataset_from_csv`` (which generates conformers), this
    function loads molecules that already have 3D coordinates.  Each
    molecule's existing conformers are preserved as-is — no conformer
    generation or energy minimisation is performed.

    Args:
        sdf_paths (str or list[str]): Path to a single SDF file, or a
            list of SDF file paths.  All molecules are combined into a
            single reference set.

    Returns:
        list[rdkit.Chem.Mol]: RDKit mol objects with explicit Hs and
            their original conformers.
    """
    # Normalise to a list
    if isinstance(sdf_paths, str):
        sdf_paths = [sdf_paths]

    mols = []
    n_skipped = 0
    global_idx = 0  # running index across all files

    for sdf_path in sdf_paths:
        if not os.path.isfile(sdf_path):
            print(f"  [WARNING] SDF file not found, skipping: {sdf_path}")
            continue

        with SDMolSupplier(sdf_path, sanitize=True, removeHs=False) as supplier:
            for mol in tqdm(supplier, desc=f"  Loading {os.path.basename(sdf_path)}"):
                if mol is None:
                    n_skipped += 1
                    global_idx += 1
                    continue

                # Keep largest fragment if multi-fragment
                num_frags = len(rdmolops.GetMolFrags(mol))
                if num_frags > 1:
                    mol = get_largest_fragment(mol)

                # Ensure explicit Hs
                mol = Chem.AddHs(mol, addCoords=True)

                # Set a name if missing
                if not mol.HasProp("_Name") or not mol.GetProp("_Name").strip():
                    mol.SetProp("_Name", f"ref_{global_idx}")

                if mol.GetNumConformers() == 0:
                    print(f"  [WARNING] Molecule ref_{global_idx} has no conformers — skipping")
                    n_skipped += 1
                    global_idx += 1
                    continue

                mols.append(mol)
                global_idx += 1

    if n_skipped > 0:
        print(f"  [INFO] Skipped {n_skipped} invalid/conformer-less entries from SDF(s)")

    return mols


def load_dataset_from_csv(csv_path, smiles_col="smiles", name_col=None):
    """Load dataset SMILES from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        smiles_col (str): Column name containing SMILES strings.
        name_col (str, optional): Column name for molecule names.

    Returns:
        list[tuple[str, str]]: List of (smiles, name) tuples.
    """
    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {csv_path}. "
                         f"Available columns: {list(df.columns)}")

    results = []
    for idx, row in df.iterrows():
        raw_smi = str(row[smiles_col]).strip()
        if not raw_smi or raw_smi == "nan":
            continue
        smi = canonicalize_smiles(raw_smi)
        if smi is None:
            print(f"  [WARNING] Skipping invalid SMILES at row {idx}: {raw_smi}")
            continue
        if name_col and name_col in df.columns:
            name = str(row[name_col]).strip()
            if not name or name == "nan":
                name = f"mol_{idx}"
        else:
            name = f"mol_{idx}"
        results.append((smi, name))

    return results


# ---------------------------------------------------------------------------
# Results post-processing
# ---------------------------------------------------------------------------

def _strip_conf_suffix(name):
    """Strip trailing conformer suffix like '_0', '_1' from a molecule name."""
    if "_" in name and name.rsplit("_", 1)[-1].isdigit():
        return "_".join(name.rsplit("_", 1)[:-1])
    return name


def build_per_mol_dataframe(scores_dict, name_to_uid, best_fit_mols=None):
    """Build a results DataFrame from roshambo2 scores.

    For each (query UID, dataset molecule) pair, one row is produced.
    Optionally includes the aligned SDF block when *best_fit_mols* is provided.

    Args:
        scores_dict (dict[str, pd.DataFrame]): Output from roshambo2.compute().
        name_to_uid (dict[str, str]): name → UID mapping.
        best_fit_mols (dict[str, list[rdkit.Chem.Mol]], optional): If provided,
            an ``sdf`` column with aligned MolBlocks is added.

    Returns:
        pd.DataFrame: Results sorted by tanimoto_combination descending.
    """
    all_dfs = []

    for query_key, df in scores_dict.items():
        if df.empty:
            continue

        result_df = df.copy()

        # Resolve roshambo2 internal name → UID (handle conformer suffix)
        base_key = _strip_conf_suffix(query_key)
        query_uid = name_to_uid.get(query_key, name_to_uid.get(base_key, query_key))
        result_df.insert(0, "query_uid", query_uid)

        # Optionally attach SDF blocks
        if best_fit_mols is not None:
            sdf_lookup = {}
            for mol in best_fit_mols.get(query_key, []):
                hit_name = mol.GetProp("name") if mol.HasProp("name") else None
                if hit_name is not None:
                    sdf_lookup[hit_name] = Chem.MolToMolBlock(mol)
            result_df["sdf"] = result_df["name"].map(sdf_lookup)

        # Strip conformer suffix to group conformers of the same dataset mol
        result_df["mol_base"] = result_df["name"].apply(_strip_conf_suffix)

        # Keep only the best conformer per dataset molecule
        result_df = result_df.sort_values("tanimoto_combination", ascending=False)
        result_df = result_df.drop_duplicates(subset=["query_uid", "mol_base"], keep="first")

        # Clean up columns
        id_cols = ["query_uid"]
        if best_fit_mols is not None:
            id_cols.append("sdf")
        drop_cols = {*id_cols, "name", "mol_base"}
        score_cols = [c for c in result_df.columns if c not in drop_cols]
        result_df = result_df[id_cols + score_cols].copy()

        result_df = result_df.sort_values("tanimoto_combination", ascending=False).reset_index(drop=True)
        all_dfs.append(result_df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Dataset preparation helper
# ---------------------------------------------------------------------------

def prepare_dataset(ds_cfg, color, hardware_options=None, mmff_max_iters=500,
                    embed_chunk_size=2000):
    """Prepare a single reference dataset and return the H5 path.

    ``ds_cfg`` is a dict with the following keys (all optional except ``h5``):
      - ``h5``:         Path where the H5 will be written / read from.
      - ``sdf``:        Path (or list of paths) to SDF file(s) with
                        pre-computed poses.
      - ``csv``:        Path to a CSV of SMILES.
      - ``smiles_col``: SMILES column name in the CSV (default ``"smiles"``).
      - ``name_col``:   Optional molecule-name column.
      - ``n_confs``:    Conformers per molecule when generating from CSV.
      - ``prefix``:     Label for this dataset (used for output file naming).

    Priority:
      1. If the H5 already exists **and** no ``sdf`` is given → reuse.
      2. ``sdf`` → load poses as-is, (re)build H5.
      3. ``csv`` → generate conformers, build H5.

    Returns:
        str: Path to the ready-to-use H5 dataset.
    """
    h5_path = ds_cfg["h5"]
    sdf_path = ds_cfg.get("sdf")   # str, list[str], or None
    csv_path = ds_cfg.get("csv")
    smiles_col = ds_cfg.get("smiles_col", "smiles")
    name_col = ds_cfg.get("name_col")
    n_confs = ds_cfg.get("n_confs", 5)
    prefix = ds_cfg.get("prefix", "")

    label = f"'{prefix}'" if prefix else "default"

    # If SDF is explicitly provided, always (re)build H5 from it
    # sdf_path can be a single string or a list of strings
    if sdf_path:
        if isinstance(sdf_path, list):
            print(f"  [{label}] Loading reference poses from {len(sdf_path)} SDF file(s)")
        else:
            print(f"  [{label}] Loading reference poses from SDF: {sdf_path}")
        dataset_mols = load_dataset_from_sdf(sdf_path)
        if not dataset_mols:
            print(f"ERROR: No valid molecules in dataset SDF for {label}. Skipping.")
            return None
        print(f"  [{label}] {len(dataset_mols)} molecule(s) loaded with existing poses")
        prepare_from_rdkitmols(dataset_mols, color=color).save_to_h5(h5_path)
        print(f"  [{label}] Saved H5: {h5_path}")
        return h5_path

    # Reuse existing H5 if present
    if os.path.isfile(h5_path):
        print(f"  [{label}] H5 already exists, reusing: {h5_path}")
        return h5_path

    # Build from CSV
    if csv_path:
        smiles_pairs = load_dataset_from_csv(csv_path, smiles_col=smiles_col,
                                             name_col=name_col)
        if not smiles_pairs:
            print(f"ERROR: No valid SMILES in dataset CSV for {label}. Skipping.")
            return None

        dataset_mols = [
            m for m in (prepare_mol(smi, name=name) for smi, name in smiles_pairs)
            if m is not None
        ]
        if not dataset_mols:
            print(f"ERROR: No valid dataset molecules parsed for {label}. Skipping.")
            return None

        print(f"  [{label}] Generating {n_confs} conformer(s) for "
              f"{len(dataset_mols)} reference molecules...")
        dataset_mols = embed_and_optimize(
            dataset_mols, n_confs=n_confs,
            hardware_options=hardware_options, mmff_max_iters=mmff_max_iters,
            chunk_size=embed_chunk_size,
        )
        if not dataset_mols:
            print(f"ERROR: Conformer generation failed for all ref molecules "
                  f"in {label}. Skipping.")
            return None

        prepare_from_rdkitmols(dataset_mols, color=color).save_to_h5(h5_path)
        print(f"  [{label}] Saved H5: {h5_path}")
        return h5_path

    print(f"ERROR: No source (sdf/csv/h5) for dataset {label}.")
    return None


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

def run_roshambo2_batch(query_mols, dataset_input, args, name_to_uid,
                        output_csv, write_header):
    """Run roshambo2 for a batch of query molecules and append results to CSV.

    Each batch creates a fresh Roshambo2 object so peak memory stays
    proportional to ``batch_size × dataset`` instead of ``total_queries ×
    dataset``.

    Args:
        query_mols (list[rdkit.Chem.Mol]): Query molecules for this batch.
        dataset_input (str): Path to H5 dataset.
        args: Parsed argparse namespace.
        name_to_uid (dict): name → UID mapping.
        output_csv (str): Path to output CSV (append mode).
        write_header (bool): Whether to write the CSV header row.

    Returns:
        int: Number of result rows written.
    """
    optim_mode = args.optim_mode or ("combination" if args.color else "shape")

    roshambo2_calc = Roshambo2(
        query_mols,
        dataset_input,
        color=args.color,
        verbosity=args.verbosity,
    )

    compute_kwargs = dict(
        backend=args.backend,
        optim_mode=optim_mode,
        start_mode=args.start_mode,
        reduce_over_conformers=True,     # roshambo2 keeps best conformer
        write_scores=False,
        max_results=args.max_results,
    )
    if args.color and optim_mode == "combination":
        compute_kwargs["combination_param"] = args.combination_param

    scores = roshambo2_calc.compute(**compute_kwargs)

    # Optionally retrieve aligned 3D structures
    best_fit_mols = None
    if args.save_sdf:
        best_fit_mols = roshambo2_calc.get_best_fit_structures(top_n=args.max_results)

    batch_df = build_per_mol_dataframe(scores, name_to_uid, best_fit_mols=best_fit_mols)
    n_rows = len(batch_df)

    if n_rows > 0:
        batch_df.to_csv(output_csv, mode="a", header=write_header, index=False)

    # Free memory eagerly
    del roshambo2_calc, scores, best_fit_mols, batch_df
    gc.collect()

    return n_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate conformers from SMILES and run roshambo2 shape overlay."
    )

    # Query input — CSV with SDF paths OR SMILES, plus UIDs
    parser.add_argument(
        "--query_csv", required=True, type=str,
        help="Path to a CSV file with query molecules. "
             "Must have a UID column and either an SDF path column or a SMILES column."
    )
    parser.add_argument(
        "--uid_col", type=str, default="uid",
        help='Column name for the unique identifier (default: "uid").'
    )
    parser.add_argument(
        "--sdf_col", type=str, default="sdf_path",
        help='Column name for the SDF file path (default: "sdf_path"). '
             'Used when --query_smiles_col is not set.'
    )
    parser.add_argument(
        "--query_smiles_col", type=str, default=None,
        help="Column name for query SMILES strings. When set, queries are "
             "loaded from SMILES instead of SDF files, and conformers are "
             "generated via nvMolKit/RDKit. Overrides --sdf_col."
    )
    parser.add_argument(
        "--query_confs", type=int, default=10,
        help="Conformers to generate per query molecule (default: 10)."
    )

    # Dataset input (the reference library — typically smaller)
    # Option A: multiple reference sets via JSON
    parser.add_argument(
        "--datasets", type=str, default=None,
        help='JSON list of dataset configs, each with "h5", and optionally '
             '"sdf", "csv", "smiles_col", "n_confs", "prefix". '
             'Example: \'[{"h5":"a.h5","csv":"a.csv","prefix":"actives"}]\'. '
             'When set, the single-dataset flags below are ignored.'
    )
    # Option B: single reference set via individual flags (backward compat)
    parser.add_argument(
        "--dataset_h5", type=str, default=None,
        help="Path to the roshambo2 H5 dataset file. "
             "If it exists it is loaded directly. "
             "Otherwise --dataset_sdf or --dataset_csv must be given."
    )
    parser.add_argument(
        "--dataset_csv", type=str, default=None,
        help="CSV of dataset SMILES. Used only when --dataset_h5 doesn't exist."
    )
    parser.add_argument(
        "--dataset_sdf", type=str, nargs="+", default=None,
        help="One or more SDF files with pre-computed 3D poses. "
             "When provided, poses are used as-is (no conformer generation). "
             "Always rebuilds the H5. Takes priority over --dataset_csv."
    )
    parser.add_argument(
        "--smiles_col", type=str, default="smiles",
        help='SMILES column in dataset CSV (default: "smiles").'
    )
    parser.add_argument(
        "--name_col", type=str, default=None,
        help='Optional molecule-name column in dataset CSV.'
    )
    parser.add_argument(
        "--n_confs", type=int, default=5,
        help="Conformers per dataset molecule (default: 5)."
    )

    # Roshambo2 settings
    parser.add_argument(
        "--backend", type=str, default="cuda", choices=["cpp", "cuda"],
        help="Compute backend (default: cuda)."
    )
    parser.add_argument(
        "--pharmacophore", action="store_true", dest="color",
        help="Enable pharmacophore (color) features."
    )
    parser.add_argument(
        "--optim_mode", type=str, default=None,
        help="Optimization mode: 'shape', 'color', or 'combination'."
    )
    parser.add_argument(
        "--combination_param", type=float, default=0.5,
        help="Color weight for 'combination' optim_mode (default: 0.5)."
    )
    parser.add_argument(
        "--start_mode", type=int, default=1, choices=[0, 1, 2],
        help="Start mode: 0=fast, 1=balanced, 2=thorough (default: 1)."
    )
    parser.add_argument(
        "--max_results", type=int, default=100,
        help="Max results to keep per query molecule (default: 100)."
    )
    parser.add_argument(
        "--output_prefix", type=str, default="roshambo2_hits",
        help="Prefix for output files (default: roshambo2_hits)."
    )
    parser.add_argument(
        "--verbosity", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity: 0=quiet, 1=info, 2=debug (default: 1)."
    )

    # Batching / scaling
    parser.add_argument(
        "--query_batch_size", type=int, default=200,
        help="Queries per roshambo2 batch. With a 600-molecule reference "
             "library a batch of 200 keeps memory comfortable. (default: 200)."
    )
    parser.add_argument(
        "--embed_chunk_size", type=int, default=2000,
        help="Max molecules per nvMolKit GPU embedding call. Prevents GPU "
             "OOM when generating conformers for 10K+ queries. (default: 2000)."
    )
    parser.add_argument(
        "--save_sdf", action="store_true",
        help="Include aligned SDF MolBlocks in output CSV. "
             "Omit to save significant memory at large scale."
    )

    # nvMolKit GPU hardware options
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=None,
        help="GPU device IDs for conformer generation (default: auto)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=500,
        help="nvMolKit GPU batch size (default: 500)."
    )
    parser.add_argument(
        "--batches_per_gpu", type=int, default=4,
        help="Batches per GPU for nvMolKit (default: 4)."
    )
    parser.add_argument(
        "--preprocessing_threads", type=int, default=8,
        help="Preprocessing threads for nvMolKit (default: 8)."
    )
    parser.add_argument(
        "--mmff_max_iters", type=int, default=500,
        help="Max MMFF optimisation iterations (default: 500)."
    )

    args = parser.parse_args()

    t_start = time.perf_counter()

    # Build nvMolKit hardware options
    hardware_options = None
    try:
        from nvmolkit.types import HardwareOptions
        hw_kwargs = dict(
            batchSize=args.batch_size,
            batchesPerGpu=args.batches_per_gpu,
            preprocessingThreads=args.preprocessing_threads,
        )
        if args.gpu_ids is not None:
            hw_kwargs["gpuIds"] = args.gpu_ids
        hardware_options = HardwareOptions(**hw_kwargs)
    except ImportError:
        pass  # embed_and_optimize will fall back to RDKit

    # ── 1. Load query molecules from CSV ─────────────────────────────
    print("=" * 60)
    print("Step 1: Loading query molecules from CSV...")
    print("=" * 60)
    print(f"  CSV file:   {args.query_csv}")
    print(f"  UID column: {args.uid_col}")

    if args.query_smiles_col:
        # ---- SMILES-based queries (no 3D coords yet) ----
        print(f"  SMILES col: {args.query_smiles_col}")
        query_mols, query_name_to_uid = load_queries_from_smiles_csv(
            args.query_csv, uid_col=args.uid_col, smiles_col=args.query_smiles_col,
        )
    else:
        # ---- SDF-based queries (already have 3D coords) ----
        print(f"  SDF column: {args.sdf_col}")
        query_mols, query_name_to_uid = load_queries_from_csv(
            args.query_csv, uid_col=args.uid_col, sdf_col=args.sdf_col,
        )

    if not query_mols:
        print("ERROR: No valid query molecules found. Exiting.")
        sys.exit(1)

    n_unique = len(set(query_name_to_uid.values()))
    print(f"  Loaded {len(query_mols)} query molecule(s), {n_unique} unique UID(s)")

    # Generate conformers for all queries in GPU chunks
    # (SDF queries already have one conformer but benefit from additional ones;
    #  SMILES queries have zero conformers and require this step.)
    print(f"  Generating {args.query_confs} conformer(s) per query...")
    query_mols = embed_and_optimize(
        query_mols, n_confs=args.query_confs,
        hardware_options=hardware_options, mmff_max_iters=args.mmff_max_iters,
        chunk_size=args.embed_chunk_size,
    )
    if not query_mols:
        print("ERROR: Conformer generation failed for all queries. Exiting.")
        sys.exit(1)

    print(f"  {len(query_mols)} query molecule(s) with conformers ready")

    # ── 2. Build the list of reference datasets ────────────────────
    # Either from --datasets JSON or from the individual flags.
    if args.datasets:
        try:
            dataset_list = json.loads(args.datasets)
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse --datasets JSON: {args.datasets}")
            sys.exit(1)
    elif args.dataset_h5:
        # Backward-compatible single dataset from individual flags
        dataset_list = [{
            "h5":         args.dataset_h5,
            "sdf":        args.dataset_sdf,
            "csv":        args.dataset_csv,
            "smiles_col": args.smiles_col,
            "name_col":   args.name_col,
            "n_confs":    args.n_confs,
            "prefix":     "",
        }]
    else:
        print("ERROR: Provide --datasets or --dataset_h5. Exiting.")
        sys.exit(1)

    print(f"\n  Reference datasets to process: {len(dataset_list)}")
    for ds in dataset_list:
        pfx = ds.get("prefix", "")
        print(f"    • {pfx or '(default)'}: h5={ds.get('h5')}")

    all_output_csvs = []

    for ds_idx, ds_cfg in enumerate(dataset_list):
        prefix = ds_cfg.get("prefix", "")
        label = prefix or "default"

        # ── 2.x  Prepare the reference dataset ───────────────────────
        print()
        print("=" * 60)
        print(f"Step 2.{ds_idx + 1}: Preparing reference dataset '{label}'...")
        print("=" * 60)

        h5_path = prepare_dataset(
            ds_cfg, color=args.color,
            hardware_options=hardware_options,
            mmff_max_iters=args.mmff_max_iters,
            embed_chunk_size=args.embed_chunk_size,
        )
        if h5_path is None:
            print(f"  ⚠ Skipping dataset '{label}' (preparation failed).")
            continue

        # ── 3.x  Run roshambo2 in query batches ─────────────────────
        print()
        print("=" * 60)
        print(f"Step 3.{ds_idx + 1}: Running roshambo2 overlay vs '{label}'...")
        print("=" * 60)

        optim_mode = args.optim_mode or ("combination" if args.color else "shape")

        n_queries = len(query_mols)
        batch_size = min(args.query_batch_size, n_queries)
        n_batches = (n_queries + batch_size - 1) // batch_size

        print(f"  Backend:          {args.backend}")
        print(f"  Pharmacophore:    {args.color}")
        print(f"  Optim mode:       {optim_mode}")
        print(f"  Start mode:       {args.start_mode}")
        print(f"  Max results:      {args.max_results} per query")
        print(f"  Save SDF:         {args.save_sdf}")
        print(f"  Query batching:   {n_queries} queries → {n_batches} batch(es) of ≤{batch_size}")
        print()

        suffix = f"_{prefix}" if prefix else ""
        output_csv = f"{args.output_prefix}{suffix}_scores.csv"
        if os.path.isfile(output_csv):
            os.remove(output_csv)

        total_rows = 0
        write_header = True

        for batch_idx in tqdm(range(n_batches), desc=f"  Batches ({label})"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_queries)
            batch_mols = query_mols[start:end]

            t_batch = time.perf_counter()
            n_rows = run_roshambo2_batch(
                batch_mols, h5_path, args, query_name_to_uid,
                output_csv, write_header,
            )
            elapsed = time.perf_counter() - t_batch
            total_rows += n_rows
            write_header = False

            tqdm.write(f"    Batch {batch_idx + 1}/{n_batches}: "
                       f"{len(batch_mols)} queries → {n_rows} rows in {elapsed:.1f}s  "
                       f"(cumulative: {total_rows})")

        all_output_csvs.append((label, output_csv, total_rows))

    # ── 4. Summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    elapsed_total = time.perf_counter() - t_start
    for label, csv_path, rows in all_output_csvs:
        print(f"  [{label}]  {rows:,} rows  →  {csv_path}")
    print(f"  Elapsed time: {elapsed_total:.1f}s")

    return [csv for _, csv, _ in all_output_csvs]


if __name__ == "__main__":
    main()
