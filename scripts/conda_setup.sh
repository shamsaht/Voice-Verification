#!/bin/bash
# setup.sh

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit
fi

# Create the environment from the YAML file
conda env create -f paco_env.yml
echo "Environment 'paco' created successfully."
