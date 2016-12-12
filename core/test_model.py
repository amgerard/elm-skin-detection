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
    #elmPkl = 'elm/elm_train_and_val.pkl' # sys.argv[1]
# elmPkl = 'elm/elm.pkl' # sys.argv[1]
    origPath = '../../../Original/test/*.jpg' # sys.argv[1]
    skinPath = '../../../Skin_test/*.bmp' # sys.argv[1]
#skinPath = '../../../Skin/val/*.bmp' # sys.argv[1]

    if len(sys.argv) > 1:
        origPath = sys.argv[1] + '*.jpg'
    if len(sys.argv) > 2:
        skinPath = sys.argv[2] + '*.bmp'

    print origPath, skinPath

    test_images = []
    test_masks = []
    try:
        test_images = io.ImageCollection(origPath)
        test_masks = io.ImageCollection(skinPath)
    except:
        print 'error loading test masks'

    no_masks_found = False
    if (len(test_masks) == 0):
        no_masks_found = True
        test_masks = [np.zeros(x.shape) for x in test_images]

    print len(test_images), len(test_masks)
    print 'loading model...'

    start = time.time()

# get ELM model
    ELM = getModel(elmPkl)

    print 'transforming all test images (in parallel)...'

#testTransform = SuperPxlTransform(test_images, test_masks)	
#testTransform.transform()
#X_test = testTransform.X
#y_test = testTransform.y

# X_test, y_test, _ = transformImageCollection2(test_images, test_masks)

    pl = ThreadPool(16)
    results = pl.map(transformImage2, zip(test_images, test_masks))

    X_test = np.concatenate([x[0] for x in results])
    y_test = np.zeros(X_test.shape[0]) # in case permissions to Skin_test are not granted
    y_test[0:100] = 1 # AUC doesn't like all 0's
    if no_masks_found == False:
        y_test = np.concatenate([x[1] for x in results])
    # y_stats = np.concatenate([x[2] for x in results])

    print 'done transforming: ', time.time() - start
    print X_test.shape, y_test.shape

# predict skin
    print 'predicting skin pixels...'
    pred = predictY(ELM, X_test)

    print 'done predicting: ', time.time() - start
    print np.bincount(pred.astype(int))
    print ''
    print 'results:'
    print_results(y_test, pred)
