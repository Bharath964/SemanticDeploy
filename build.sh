#!/bin/bash
# Install Python 3.10
sudo apt-get update
sudo apt-get install python3.10
python3.10 -m pip install --upgrade pip
python3.10 -m pip install -r requirementsdeploy.txt
