# Euler ENV Setup

Source: mostly copied from [leggedrobotics - euler-cluster-guide](https://leggedrobotics.github.io/euler-cluster-guide/python-environments.html)

📦 Installing Miniconda3

To install Miniconda3, follow these steps:

# Create a directory for Miniconda
mkdir -p /cluster/project/rsl/$USER/miniconda3

# Download the installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /cluster/project/rsl/$USER/miniconda3/miniconda.sh

# Run the installer silently
bash /cluster/project/rsl/$USER/miniconda3/miniconda.sh -b -u -p /cluster/project/rsl/$USER/miniconda3/

# Clean up the installer
rm /cluster/project/rsl/$USER/miniconda3/miniconda.sh

# Initialize conda for bash
/cluster/project/rsl/$USER/miniconda3/bin/conda init bash

# Prevent auto-activation of the base environment
conda config --set auto_activate_base false

# Crate a conda environment with python 3.10
conda create -n isaaclab-env python=3.10
-> this should create a new env at /cluster/project/rsl/$USER/miniconda3/envs/isaaclab-env

# To activate the environment:

conda activate isaaclab-env

# To deactivate the environment:

conda deactivate



