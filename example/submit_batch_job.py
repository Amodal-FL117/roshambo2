"""
submit_batch_job.py – Submit a roshambo2 shape overlay job to the FSP batch API.

Sends a single GPU job that runs smiles_to_roshambo2.py inside the
roshambo2 Docker container.  All roshambo2 CLI flags can be customised
in the ROSHAMBO2_ARGS dict below.

Usage:
    python submit_batch_job.py
"""

import requests
import json
import time
from datetime import datetime, timezone

# ── API configuration ────────────────────────────────────────────────
BASE_URL = "https://fsp-api.data-platform.fsp-amodal.com"
ENDPOINT = "/job"

# ── Docker image ─────────────────────────────────────────────────────
ECR_ACCOUNT = "184892497286"
ECR_REGION = "us-east-1"
ECR_REPO = "roshambo2"
ECR_TAG = "latest"
IMAGE_URI = f"{ECR_ACCOUNT}.dkr.ecr.{ECR_REGION}.amazonaws.com/{ECR_REPO}:{ECR_TAG}"

# ── roshambo2 arguments (edit these) ─────────────────────────────────
ROSHAMBO2_ARGS = {
    # Query input
    "--query_csv":          "/fsx/amira/craig_handout_mols/fs_diverse_sm_results.csv",
    "--uid_col":            "uid",
    "--sdf_col":            "sdf_path",         # used when --query_smiles_col is None
    "--query_smiles_col":   "smiles",           # use SMILES column instead of SDF files
    "--query_confs":        50,

    # Dataset (reference library)
    "--dataset_h5":         "/fsx/amira/mc4r/pm03_data/MC4R_Database_small_molecules_new.h5",
    "--dataset_csv":        "/fsx/amira/mc4r/pm03_data/MC4R_Database_small_molecules_new.csv",   # only needed if H5 doesn't exist yet
    "--smiles_col":         "SMILES",
    "--n_confs":            100,

    # Roshambo2 settings
    "--backend":            "cuda",
    "--color":              True,           # set False to disable pharmacophore features
    "--optim_mode":         "combination",  # "shape", "color", or "combination"
    "--combination_param":  0.5,
    "--start_mode":         1,              # 0=fast, 1=balanced, 2=thorough
    "--max_results":        100,
    "--output_prefix":      "/fsx/amira/craig_handout_mols/fs_diverse_sm_roshambo2",

    # Batching / scaling
    "--query_batch_size":   200,
    "--embed_chunk_size":   2000,
    "--save_sdf":           False,          # set True to include aligned SDF blocks

    # nvMolKit GPU settings
    "--batch_size":         500,
    "--batches_per_gpu":    4,
    "--preprocessing_threads": 8,
    "--mmff_max_iters":     500,
}

# ── Build the command list ────────────────────────────────────────────
def build_command(args_dict):
    """Convert the args dict into a list of CLI tokens for smiles_to_roshambo2.py.

    Boolean True  → flag is included (e.g. --color)
    Boolean False → flag is omitted
    None          → flag is omitted
    Other values  → flag + str(value)
    """
    parts = []
    for flag, value in args_dict.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                parts.append(flag)
        else:
            parts.append(flag)
            parts.append(str(value))
    return parts


# The ENTRYPOINT is ["pixi", "run", "python", "example/smiles_to_roshambo2.py"]
# so the command list is appended as arguments to the script.
command = build_command(ROSHAMBO2_ARGS)

# ── Job parameters ───────────────────────────────────────────────────
params = {
    "image_uri": IMAGE_URI,
    "cpu":       "8",
    "memory_mb": "60000",
    "gpu":       "1",
    "command":   command,
    "num_nodes": "1",
}

# ── Submit ───────────────────────────────────────────────────────────
print("=" * 60)
print("Submitting roshambo2 batch job")
print("=" * 60)
print(f"  Image:   {IMAGE_URI}")
print(f"  CPU:     {params['cpu']}")
print(f"  Memory:  {params['memory_mb']} MB")
print(f"  GPU:     {params['gpu']}")
print(f"  Command: {' '.join(command)}")
print()

t_submit = time.perf_counter()
submit_time = datetime.now(timezone.utc)

response = requests.post(
    f"{BASE_URL}{ENDPOINT}",
    params=params,
)

resp_json = response.json()
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(resp_json, indent=2)}")

if response.status_code != 200 or "jobId" not in resp_json:
    print("\nJob submission failed — not polling.")
    exit(1)

job_id = resp_json["jobId"]
print(f"\nSubmitted at: {submit_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"Job ID:       {job_id}")
print(f"Console:      {resp_json.get('consoleUrl', 'N/A')}")

# ── Poll job status until terminal state ─────────────────────────────
STATUS_ENDPOINT = f"/job/{job_id}"
TERMINAL_STATES = {"SUCCEEDED", "FAILED"}
POLL_INTERVAL = 30  # seconds

print(f"\nPolling every {POLL_INTERVAL}s until job reaches a terminal state...")
print("-" * 60)

while True:
    time.sleep(POLL_INTERVAL)
    elapsed = time.perf_counter() - t_submit

    try:
        status_resp = requests.get(f"{BASE_URL}{STATUS_ENDPOINT}")
        status_json = status_resp.json()
        status = status_json.get("status", "UNKNOWN")
    except Exception as e:
        print(f"  [{elapsed:>7.0f}s] Error polling status: {e}")
        continue

    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    time_str = f"{hrs}h {mins:02d}m {secs:02d}s" if hrs else f"{mins}m {secs:02d}s"

    print(f"  [{time_str}] Status: {status}")

    if status in TERMINAL_STATES:
        end_time = datetime.now(timezone.utc)
        total_elapsed = time.perf_counter() - t_submit
        mins, secs = divmod(int(total_elapsed), 60)
        hrs, mins = divmod(mins, 60)
        final_time = f"{hrs}h {mins:02d}m {secs:02d}s" if hrs else f"{mins}m {secs:02d}s"

        print()
        print("=" * 60)
        print(f"  Job {status}")
        print(f"  Submitted:    {submit_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Finished:     {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Total time:   {final_time}")
        print("=" * 60)
        break