import time
import sys
from multiprocessing import Pool, Lock, freeze_support
import itertools
import numpy as np
import skimage.io as io
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score
from ELM import ELMRegressor
import Transform


def func(pair):
    test_imgs = [pair[0]]
    test_masks = [pair[1]]
    ELM = pair[2]

    
    test = Transform.SuperPxlParallelTestTransform(test_imgs,test_masks)
    test.transform()
    X_test = test.X
    y_test = test.y

    prediction = ELM.predict(X_test)
    thresh = 0.3
    skin = prediction >= thresh
    not_skin = prediction < thresh
    prediction[skin] = 1
    prediction[not_skin] = 0

    accuracy = accuracy_score(y_test, prediction)
    conf = confusion_matrix(y_test,prediction)
    auc = roc_auc_score(y_test,prediction)
    results = [accuracy,conf[0,0],conf[0,1],conf[1,0],conf[1,1],auc]
    return results

if __name__ == "__main__":
    
    start = time.time()

    # load transformed training data
    path = 'results/_300s_1sig_entropy/'
    X_train = np.loadtxt(path + "X_train.csv", delimiter=",")
    y_train = np.loadtxt(path + "y_train.csv", delimiter=",").astype(int)

    # train model
    x = 1300
    ELM = ELMRegressor(x)
    ELM.fit(X_train, y_train)

    # load test data
    path = '' if len(sys.argv) < 4 else str(sys.argv[3])
    test_imgs = io.ImageCollection(path + '*.jpg')
    path = '' if len(sys.argv) < 5 else str(sys.argv[4])
    test_masks = io.ImageCollection(path + '*.bmp')

    elms = [ELM.copy() for _ in range(700)]
    ims_and_masks = zip(test_imgs,test_masks,elms)
    
    # launch threads
    pl = Pool(16)
    results = pl.map(func, ims_and_masks)
     
    #results = [func(trip) for trip in ims_and_masks]

    for r in results:
        print ','.join(str(x) for x in r)

    end = time.time()
    print 'time elapsed: ' + str(end-start)
