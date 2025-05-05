#!/bin/bash
# Disable Poetry
unset PIP_REQUIRE_VIRTUALENV
unset POETRY_VIRTUALENVS_CREATE
# Install Python 3.10
pyenv install 3.10.13
pyenv global 3.10.13

# Now use python3.10 explicitly
python -m pip install --upgrade pip
python -m pip install -r requirementsdeploy.txt
python -m pip install streamlit==1.15.2
