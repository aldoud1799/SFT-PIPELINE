#!/usr/bin/env bash
set -e

python -m pip uninstall -y torcheo
python -m pip install -r requirements.txt
echo "Setup completed"