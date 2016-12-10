import time
start = time.time()

import sys
import Transform
import skimage.io as io
import numpy as np
from sklearn.metrics import mean_absolute_error
from ELM import ELMRegressor

from multiprocessing.dummy import Pool as ThreadPool

def transformImage(pair):
    im = pair[0]
    mask = pair[1]
    train = Transform.SuperPxlTransform([im], [mask])
    train.transform()
    X_train = train.X
    y_train = train.y
    return X_train,y_train

if __name__ == '__main__':
    print 'get validation weights, #pixels/image'

    path = '' if len(sys.argv) < 4 else str(sys.argv[3])
    all_test_imgs = io.ImageCollection(path + '*.jpg')
    
    tot_pxls = sum([im.shape[0]*im.shape[1] for im in all_test_imgs])

    weights = np.concatenate([[im.shape[0]*im.shape[1]/float(tot_pxls)] for im in all_test_imgs])

# same transformed data
    np.savetxt("weights.csv", weights, delimiter=",")

    print 'done'

    end = time.time()
    print 'time elapsed: ' + str(end-start)
