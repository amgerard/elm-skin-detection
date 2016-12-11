import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import time
from multiprocessing.dummy import Pool as ThreadPool

from elm.ELM import ELMRegressor
from transform.SuperPxlTransform import SuperPxlTransform
from print_superpxl_results import print_results 
from print_superpxl_results import predictY 
from save_superpxl_transform import transformImageCollection

def getModel(elmPkl):
	with open(elmPkl, 'rb') as inpu:
		return pickle.load(inpu)

def transformImage2(pair):
	im = pair[0]
	mask = pair[1]
	train = SuperPxlTransform([im], [mask])
	train.transform()
	X_train = train.X
	y_train = train.y
	return X_train,y_train,train.y_stats

if __name__ == '__main__':
	
	elmPkl = 'core/elm/elm_train_and_val.pkl' # sys.argv[1]
	origPath = '../../Original/test/*.jpg' # sys.argv[1]
	skinPath = '../../Skin_test/test/*.bmp' # sys.argv[1]

	test_images = io.ImageCollection(origPath)
	# test_masks = io.ImageCollection(skinPath)
	test_masks = [np.zeros(x.shape) for x in test_images]

	print len(test_images), len(test_masks)

	start = time.time()
	
	# get ELM model
	ELM = getModel(elmPkl)

	print 'a'

	#testTransform = SuperPxlTransform(test_images, test_masks)	
	#testTransform.transform()
	#X_test = testTransform.X
	#y_test = testTransform.y

	# X_test, y_test, _ = transformImageCollection2(test_images, test_masks)

	pl = ThreadPool(16)
        results = pl.map(transformImage2, zip(test_images, test_masks))

        X_test = np.concatenate([x[0] for x in results])
        y_test = np.concatenate([x[1] for x in results])
        y_stats = np.concatenate([x[2] for x in results])

	print 'b', time.time() - start
	print X_test.shape, y_test.shape

	# predict skin
	pred = predictY(ELM, X_test)

	print 'c', time.time() - start
	print_results(y_test, pred)
	print np.bincount(pred.astype(int))
