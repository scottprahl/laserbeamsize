#!/usr/bin/env python3
"""Execute a Jupyter notebook from the command line."""

import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_path, timeout=600):
    """Execute a Jupyter notebook and save the results."""
    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout)

    # Execute the notebook
    ep.preprocess(nb, {"metadata": {"path": notebook_path.parent if hasattr(notebook_path, "parent") else "."}})

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
