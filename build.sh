#!/bin/bash
# Disable Poetry
unset PIP_REQUIRE_VIRTUALENV
unset POETRY_VIRTUALENVS_CREATE
# Install Python 3.10
pyenv install 3.10.13
pyenv global 3.10.13
# Force Python version (if available)
python3 -m pip install --upgrade pip
python3 -m pip install -r requirementsdepoly.txt --no-cache-dir

