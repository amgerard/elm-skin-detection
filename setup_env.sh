#!/bin/bash          

echo Starting 

# mkdir virtenvs

module load python/2.7.10

# virtualenv virtenvs/mpi4py

virtualenv --system-site-packages virtenvs/mpi4py

source virtenvs/mpi4py/bin/activate

pip install -U scikit-image

# deactivate

echo Ending
