#!/bin/bash          
echo Starting 
#python test.py ../../../Original/train/ ../../../Skin/train/ > output.txt

. setup_env.sh

# python save_superpxl_transform.py ../../../Original/train/ ../../../Skin/train/ ../../../Original/val/ ../../../Skin/val/

echo running model on test dataset

python core/test_model.py

echo Ending
