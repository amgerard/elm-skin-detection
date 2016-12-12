#!/bin/bash          

imagePath=/Shared/bdagroup3/Fall2016/Original/test/
maskPath=/Shared/bdagroup3/Fall2016/Skin_test/test/

if [ $# -eq 2 ]
  then
    echo "2 arguments supplied"
    imagePath=$1
    maskPath=$2
fi

echo $imagePath
echo $maskPath
echo setup environment
. setup_env.sh

# python save_superpxl_transform.py ../../../Original/train/ ../../../Skin/train/ ../../../Original/val/ ../../../Skin/val/

echo testing model on test dataset

python core/test_model.py $imagePath $maskPath

echo done
