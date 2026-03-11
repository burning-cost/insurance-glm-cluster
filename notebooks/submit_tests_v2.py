"""
Submit test job to Databricks serverless and capture output via file.
"""
from __future__ import annotations
import os
import sys
import time
import base64
from pathlib import Path

env_path = Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service.jobs import NotebookTask, SubmitTask, JobEnvironment
from databricks.sdk.service.compute import Environment

w = WorkspaceClient()

# Notebook writes output to a workspace file and returns summary via dbutils.notebook.exit()
NOTEBOOK_CONTENT = r'''# Databricks notebook source
# MAGIC %pip install "numpy>=1.24" "pandas>=2.0" "scipy>=1.12" "statsmodels>=0.14" "scikit-learn>=1.3" pytest --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess, sys

pip_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-glm-cluster", "--quiet"],
    capture_output=True, text=True
)

test_result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-glm-cluster/tests/",
     "-v", "--tb=short", "-p", "no:warnings"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-glm-cluster"
)

output = test_result.stdout
if test_result.stderr:
    output += "\nSTDERR:\n" + test_result.stderr[-500:]
output += f"\nRETURN CODE: {test_result.returncode}"

# Write to workspace file
output_path = "/Workspace/insurance-glm-cluster/test_output.txt"
with open(output_path, "w") as f:
    f.write(output)

# Return summary via notebook exit (captured by API)
lines = test_result.stdout.strip().split("\n")
# Find the summary line
summary = ""
for line in reversed(lines):
    if "passed" in line or "failed" in line or "error" in line:
        summary = line.strip()
        break

exit_msg = f"RC={test_result.returncode} | {summary}"
dbutils.notebook.exit(exit_msg)
'''

nb_path = "/Workspace/insurance-glm-cluster/notebooks/test_runner_v2"
content_b64 = base64.b64encode(NOTEBOOK_CONTENT.encode()).decode()
w.workspace.import_(
    path=nb_path,
    content=content_b64,
    format=ws_svc.ImportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    overwrite=True,
)
print(f"Notebook ready at {nb_path}")

env = JobEnvironment(
    environment_key="Default",
    spec=Environment(client="2"),
)

print("Submitting serverless job...")
run_response = w.jobs.submit(
    run_name="glm-cluster-tests-v2",
    tasks=[
        SubmitTask(
            task_key="run_tests",
            notebook_task=NotebookTask(notebook_path=nb_path),
            environment_key="Default",
        )
    ],
    environments=[env],
)
run_id = run_response.run_id
print(f"Run ID: {run_id}")

# Poll
start = time.time()
while True:
    state = w.jobs.get_run(run_id=run_id)
    lc = str(state.state.life_cycle_state)
    rs = str(state.state.result_state) if state.state.result_state else "..."
    elapsed = int(time.time() - start)
    print(f"  [{elapsed}s] {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break
    time.sleep(15)

# Get output
for task in (state.tasks or []):
    out = w.jobs.get_run_output(run_id=task.run_id)
    if out.notebook_output and out.notebook_output.result:
        print(f"\nSummary: {out.notebook_output.result}")
    if out.error:
        print(f"\nERROR: {out.error}")
    if out.error_trace:
        print(out.error_trace[-2000:])

# Read the output file from workspace
print("\nReading full test output from workspace...")
try:
    content = w.workspace.export(
        path="/Workspace/insurance-glm-cluster/test_output.txt",
        format=ws_svc.ExportFormat.AUTO,
    )
    if content.content:
        decoded = base64.b64decode(content.content).decode()
        print(decoded)
except Exception as e:
    print(f"Could not read output file: {e}")

final = str(state.state.result_state)
print(f"\nJob result: {final}")
sys.exit(0 if "SUCCESS" in final else 1)
