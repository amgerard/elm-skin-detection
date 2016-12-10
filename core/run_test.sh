#!/bin/bash          
echo Starting 
#python test.py ../../../Original/train/ ../../../Skin/train/ > output.txt
python save_superpxl_transform.py ../../../Original/train/ ../../../Skin/train/ ../../../Original/val/ ../../../Skin/val/
echo Ending
