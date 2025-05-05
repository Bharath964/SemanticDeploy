#!/bin/bash
# Upgrade pip
python3.10 -m pip install --upgrade pip

# Install requirements (make sure filename matches)
python3.10 -m pip install -r requirementsdeploy.txt

# Explicitly install Streamlit
python3.10 -m pip install streamlit==1.15.2