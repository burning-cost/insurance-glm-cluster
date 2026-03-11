# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-glm-cluster: Test Runner
# MAGIC
# MAGIC Runs the full test suite on Databricks serverless compute.

# COMMAND ----------

# MAGIC %pip install numpy pandas scipy statsmodels scikit-learn pytest

# COMMAND ----------

import subprocess
import sys
import os

# COMMAND ----------

# Install the package in editable mode
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-glm-cluster"],
    capture_output=True, text=True
)
print(result.stdout)
print(result.stderr)

# COMMAND ----------

# Run the tests
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-glm-cluster/tests/",
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-glm-cluster"
)
print(result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout)
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)
