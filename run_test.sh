#!/bin/bash          

echo setup environment
. setup_env.sh

# python save_superpxl_transform.py ../../../Original/train/ ../../../Skin/train/ ../../../Original/val/ ../../../Skin/val/

echo testing model on test dataset

python core/test_model.py

echo done
