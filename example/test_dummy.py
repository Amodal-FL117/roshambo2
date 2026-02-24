#!/usr/bin/env python
"""Generate dummy test data and run smiles_to_roshambo2.py end-to-end.

Tests both the basic path and the batched query path.
"""

import os
import subprocess
import sys
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

TEST_DIR = os.path.join(tempfile.gettempdir(), "roshambo2_test")
os.makedirs(TEST_DIR, exist_ok=True)

# ── 1. Create query SDF files ────────────────────────────────────────
query_smiles = {
    "QUERY_001": "c1ccccc1",          # benzene
    "QUERY_002": "c1ccc(O)cc1",       # phenol
    "QUERY_003": "c1ccc(N)cc1",       # aniline
    "QUERY_004": "c1ccc(C)cc1",       # toluene
}

sdf_paths = {}
for uid, smi in query_smiles.items():
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    AllChem.MMFFOptimizeMolecule(mol)
    mol.SetProp("_Name", uid)
    sdf_path = os.path.join(TEST_DIR, f"{uid}.sdf")
    with Chem.SDWriter(sdf_path) as w:
        w.write(mol)
    sdf_paths[uid] = sdf_path
    print(f"  Wrote query SDF: {sdf_path}")

# ── 2. Create the query CSV ──────────────────────────────────────────
query_csv = os.path.join(TEST_DIR, "queries.csv")
with open(query_csv, "w") as f:
    f.write("uid,sdf_path\n")
    for uid, path in sdf_paths.items():
        f.write(f"{uid},{path}\n")
print(f"  Wrote query CSV: {query_csv}")

# ── 3. Create the dataset CSV (small set of SMILES) ──────────────────
dataset_smiles = [
    "c1ccccc1",          # benzene (identical to query 1)
    "c1ccc(O)cc1",       # phenol  (identical to query 2)
    "CCO",               # ethanol
    "c1ccc(N)cc1",       # aniline
    "c1ccc(C)cc1",       # toluene
    "CC(=O)O",           # acetic acid
]

dataset_csv = os.path.join(TEST_DIR, "dataset.csv")
with open(dataset_csv, "w") as f:
    f.write("smiles\n")
    for smi in dataset_smiles:
        f.write(f"{smi}\n")
print(f"  Wrote dataset CSV: {dataset_csv}")

# Remove any previously-cached H5 so we test the full pipeline
h5_path = os.path.join(TEST_DIR, "dataset.h5")
if os.path.isfile(h5_path):
    os.remove(h5_path)
    print(f"  Removed old H5: {h5_path}")

# ── 4. Run smiles_to_roshambo2.py (batched, --query_batch_size 2) ────
script = os.path.join(os.path.dirname(__file__), "smiles_to_roshambo2.py")
output_prefix = os.path.join(TEST_DIR, "test_hits")

cmd = [
    sys.executable, script,
    "--query_csv", query_csv,
    "--dataset_h5", h5_path,
    "--dataset_csv", dataset_csv,
    "--query_confs", "3",
    "--n_confs", "2",
    "--backend", "cpp",
    "--start_mode", "1",
    "--output_prefix", output_prefix,
    "--verbosity", "1",
    "--query_batch_size", "2",       # 4 queries in 2 batches of 2
    "--max_results", "10",
]

print()
print("=" * 60)
print("Running smiles_to_roshambo2.py (batched, batch_size=2) ...")
print("  " + " ".join(cmd))
print("=" * 60)
print()

result = subprocess.run(cmd, cwd=os.path.dirname(script))

# ── 5. Quick sanity-check on output ──────────────────────────────────
print()
print("=" * 60)
print("Checking outputs ...")
print("=" * 60)

scores_csv = f"{output_prefix}_scores.csv"
if os.path.isfile(scores_csv):
    import pandas as pd
    df = pd.read_csv(scores_csv)
    print(f"\n  ✓ Scores CSV found: {scores_csv}")
    print(f"    Rows: {len(df)},  Columns: {list(df.columns)}")

    # Show non-SDF columns
    display_cols = [c for c in df.columns if c != "sdf"]
    print(df[display_cols].head(20).to_string(index=False))

    # Check for all 4 query UIDs
    found_uids = sorted(df["query_uid"].unique())
    print(f"\n    Query UIDs found: {found_uids}")
    expected_uids = sorted(query_smiles.keys())
    if found_uids == expected_uids:
        print("    ✓ All expected UIDs present")
    else:
        print(f"    ✗ Expected {expected_uids}, got {found_uids}")

    # SDF column should NOT be present (--save_sdf not used)
    if "sdf" in df.columns:
        print("\n    ✗ Unexpected 'sdf' column in output (--save_sdf was not used)")
    else:
        print("\n    ✓ No 'sdf' column (correct — --save_sdf not used)")
else:
    print(f"\n  ✗ Scores CSV not found at: {scores_csv}")

if os.path.isfile(h5_path):
    print(f"\n  ✓ H5 cache created: {h5_path}")
else:
    print(f"\n  ✗ H5 cache not found at: {h5_path}")

sys.exit(result.returncode)
