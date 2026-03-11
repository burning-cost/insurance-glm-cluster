"""
Upload project to Databricks and run tests via the Jobs API.
Run this from the local machine (Raspberry Pi) — it only uses the SDK.
"""
from __future__ import annotations

import os
import sys
import time
import base64
from pathlib import Path

# Load creds
env_path = Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service.jobs import Task, NotebookTask, JobCluster
from databricks.sdk.service.compute import ClusterSpec, AutoScale

w = WorkspaceClient()

PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_PATH = "/Workspace/insurance-glm-cluster"

def upload_file(local_path: Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    try:
        w.workspace.mkdirs(str(Path(remote_path).parent))
    except Exception:
        pass
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=ws_svc.ImportFormat.AUTO,
        overwrite=True,
    )

def upload_directory(local_dir: Path, remote_dir: str) -> None:
    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            if any(p in str(local_file) for p in [".git", "__pycache__", ".egg-info", ".pyc"]):
                continue
            relative = local_file.relative_to(local_dir)
            remote_path = f"{remote_dir}/{relative}".replace("\\", "/")
            print(f"  Uploading {relative} -> {remote_path}")
            upload_file(local_file, remote_path)

print("Uploading project files...")
upload_directory(PROJECT_ROOT, WORKSPACE_PATH)
print("Upload complete.")

# Create a notebook to run tests
NOTEBOOK_CONTENT = '''# Databricks notebook source
# MAGIC %pip install numpy pandas scipy statsmodels "scikit-learn>=1.3" pytest

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-glm-cluster", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:])
print(result.stderr[-1000:])

# COMMAND ----------
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-glm-cluster/tests/",
     "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-glm-cluster"
)
print(result.stdout[-10000:] if len(result.stdout) > 10000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-2000:])
print("Return code:", result.returncode)
assert result.returncode == 0, f"Tests failed! Return code: {result.returncode}"
'''

nb_path = f"{WORKSPACE_PATH}/notebooks/run_tests_nb"
content_b64 = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
try:
    w.workspace.mkdirs(f"{WORKSPACE_PATH}/notebooks")
except Exception:
    pass
w.workspace.import_(
    path=nb_path,
    content=content_b64,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print(f"Notebook uploaded to {nb_path}")

# Run the notebook as a job
print("Submitting test job...")
run = w.jobs.submit(
    run_name="insurance-glm-cluster-tests",
    tasks=[
        Task(
            task_key="run_tests",
            notebook_task=NotebookTask(
                notebook_path=nb_path,
            ),
            new_cluster=ClusterSpec(
                spark_version="15.4.x-scala2.12",
                node_type_id="m5d.large",
                num_workers=1,
            ),
        )
    ],
).result()

print(f"Job submitted. Run ID: {run.run_id}")

# Poll for completion
print("Waiting for job...")
while True:
    run_state = w.jobs.get_run(run_id=run.run_id)
    life_cycle = run_state.state.life_cycle_state
    result_state = run_state.state.result_state
    print(f"  Status: {life_cycle} / {result_state}")
    if str(life_cycle) in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(15)

# Get output
for task_run in (run_state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task_run.run_id)
        if output.notebook_output:
            print("\n=== NOTEBOOK OUTPUT ===")
            print(output.notebook_output.result)
        if output.error:
            print("\n=== ERROR ===")
            print(output.error)
    except Exception as e:
        print(f"Could not retrieve output: {e}")

success = str(result_state) == "ResultState.SUCCESS"
print(f"\nJob result: {result_state}")
sys.exit(0 if success else 1)
