#!/bin/bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Install the laserbeamsize package in editable mode
pip install -e .
