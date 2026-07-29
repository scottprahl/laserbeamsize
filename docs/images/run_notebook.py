#!/usr/bin/env python3
"""Execute a Jupyter notebook from the command line."""

import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def strip_execution_metadata(nb):
    """
    Remove the per-cell execution timing recorded by nbclient.

    Those entries hold wall-clock timestamps, so leaving them in place makes
    every rerun rewrite the notebook even when nothing about it has changed.

    Args:
        nb: the notebook to clean, modified in place
    """
    for cell in nb.cells:
        cell.get("metadata", {}).pop("execution", None)


def run_notebook(notebook_path, timeout=600):
    """Execute a Jupyter notebook and save the results."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # record_timing=False keeps nbclient from stamping each cell with the time
    # it ran, so regenerating the images produces no spurious diff
    ep = ExecutePreprocessor(timeout=timeout, record_timing=False)

    # Execute the notebook
    ep.preprocess(nb, {"metadata": {"path": notebook_path.parent if hasattr(notebook_path, "parent") else "."}})

    # also drop timings left over from before record_timing was disabled
    strip_execution_metadata(nb)

    # Optionally save the executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Successfully executed {notebook_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    run_notebook(notebook_path)
