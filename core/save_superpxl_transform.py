import time
start = time.time()

import sys
import skimage.io as io
import numpy as np
from sklearn.metrics import mean_absolute_error
from multiprocessing.dummy import Pool as ThreadPool

from elm.ELM import ELMRegressor
from transform.SuperPxlTransform import SuperPxlTransform

def transformImage(pair):
    im = pair[0]
    mask = pair[1]
    train = SuperPxlTransform([im], [mask])
    train.transform()
    X_train = train.X
    y_train = train.y
    return X_train,y_train,train.y_stats

def transformImageCollection(images, masks):
    pl = ThreadPool(16)
    results = pl.map(transformImage, zip(images, masks))

    X = np.concatenate([x[0] for x in results])
    y = np.concatenate([x[1] for x in results])
    y_stats = np.concatenate([x[2] for x in results])
    return X, y, y_stats

def get_image_collection(path):
    return io.ImageCollection(path)

if __name__ == '__main__':
    print 'load train data'

    # path to training data (can be passed in as argument)
    path = '../../../Original/train/' if len(sys.argv) < 2 else str(sys.argv[1])

    # load all training images @ path
    all_train_imgs = io.ImageCollection(path + '*.jpg')

    # path to training masks (can be passed in as argument)
    path = '../../../Skin/train/' if len(sys.argv) < 3 else str(sys.argv[2])

    # load all training images @ path
    all_train_masks = io.ImageCollection(path + '*.bmp')

    print 'begin transform train data'
    
    pl = ThreadPool(16)
    results = pl.map(transformImage, zip(all_train_imgs,all_train_masks))

    X_train = np.concatenate([x[0] for x in results])
    y_train = np.concatenate([x[1] for x in results])
    y_train_stats = np.concatenate([x[2] for x in results])

    print 'done, transforming training data: ' + str(time.time()-start)
    print 'load test data'

    path = '../../../Original/val/' if len(sys.argv) < 4 else str(sys.argv[3])
    all_test_imgs = io.ImageCollection(path + '*.jpg')
    
    path = '../../../Skin/val/' if len(sys.argv) < 5 else str(sys.argv[4])
    all_test_masks = io.ImageCollection(path + '*.bmp')

    print 'begin transform test data'
    pl2 = ThreadPool(16)
    results = pl2.map(transformImage, zip(all_test_imgs,all_test_masks))

    X_test = np.concatenate([x[0] for x in results])
    y_test = np.concatenate([x[1] for x in results])
    y_test_stats = np.concatenate([x[2] for x in results])
    
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    print 'done, transforming test data: ' + str(time.time()-start)

    # same transformed data
    path = '' # '../../results/_300s_1sig_skew_30compact/'
    np.savetxt(path + "X_train.csv", X_train, delimiter=",")
    np.savetxt(path + "y_train.csv", y_train, delimiter=",")
    np.savetxt(path + "X_test.csv", X_test, delimiter=",")
    np.savetxt(path + "y_test.csv", y_test, delimiter=",")
    np.savetxt(path + "y_test_stats.csv", y_test_stats, delimiter=",")
    np.savetxt(path + "y_train_stats.csv", y_train_stats, delimiter=",")

    print 'done saving transformed data' + str(time.time()-start)

    ELM = ELMRegressor(1300)
    ELM.fit(X_train, y_train)
    prediction = ELM.predict(X_train)

    print 'train error: ' + str(mean_absolute_error(y_train, prediction))

    prediction = ELM.predict(X_test)
    print 'test error: ' + str(mean_absolute_error(y_test, prediction))

    end = time.time()
    print 'time elapsed: ' + str(end-start)
