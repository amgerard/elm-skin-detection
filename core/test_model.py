import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import time

from elm.ELM import ELMRegressor
from transform.SuperPxlTransform import SuperPxlTransform
from print_superpxl_results import print_results 
from print_superpxl_results import predictY 
from save_superpxl_transform import transformImageCollection

def getModel(elmPkl):
	with open(elmPkl, 'rb') as inpu:
		return pickle.load(inpu)

def test_model(elmPkl, test_images, test_masks):
	
	start = time.time()
	
	# get ELM model
	ELM = getModel(elmPkl)

	print 'a'

	#testTransform = SuperPxlTransform(test_images, test_masks)	
	#testTransform.transform()
	#X_test = testTransform.X
	#y_test = testTransform.y

	X_test, y_test, _ = transformImageCollection(test_images, test_masks)

	print 'b'
	print X_test.shape, y_test.shape

	# predict skin
	pred = predictY(ELM, X_test)
	
	print np.bincount(pred)
	print_results(y_test, pred)
	print time.time() - start

if __name__ == '__main__':
	
	elmPath = 'elm/elm_train_and_val.pkl' # sys.argv[1]
	origPath = '../../../Original/test/*.jpg' # sys.argv[1]
	skinPath = '../../../Skin_test/test/*.bmp' # sys.argv[1]

	test_images = io.ImageCollection(origPath)
	test_masks = io.ImageCollection(skinPath)

	print len(test_images), len(test_masks)

	skin_im = test_model(elmPath, test_images, [np.zeros(x.shape) for x in test_images])
	# skin_im = test_model(elmPath, test_images, test_masks)
