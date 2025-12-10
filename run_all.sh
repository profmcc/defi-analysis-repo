#!/bin/bash
# Simple shell script to run all notebooks in order
# Double-click or run: ./run_all.sh

cd "$(dirname "$0")"
python3 run_all_notebooks.py
