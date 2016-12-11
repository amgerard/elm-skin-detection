import time
start = time.time()

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score
import pickle
from elm.ELM import ELMRegressor

def predictY(ELM, train):
    predict = ELM.predict(train)
    thresh = 0.3 # 0.5
    skin = predict >= thresh
    not_skin = predict < thresh
    predict[skin] = 1
    predict[not_skin] = 0
    return predict

# same transformed data
path = '../../results/_300s_1sig_entropy/'
X = np.loadtxt(path + "X_train_and_val.csv", delimiter=",")
y = np.loadtxt(path + "y_train_and_val.csv", delimiter=",").astype(int)

print 'a'

x = 1300
ELM = ELMRegressor(x)
ELM.fit(X, y)

print 'b'

with open(path + 'elm_train_and_val.pkl', 'wb') as output:
    pickle.dump(ELM,output,pickle.HIGHEST_PROTOCOL)

# predict training set
prediction = predictY(ELM,X)

print 'train accuracy: ' + str(accuracy_score(y_train, prediction))

prediction = predictY(ELM,X_test)

print 'test accuracy: ' + str(accuracy_score(y_test, prediction))

print y_test.shape, prediction.shape

conf = confusion_matrix(y_test,prediction).astype(float)
auc = roc_auc_score(y_test,prediction)
tn = conf[0,0] # true negative
fp = conf[0,1] # false positive
fn = conf[1,0] # false negative
tp = conf[1,1] # true positive
all_pos = tp + fn
all_neg = fp + tn
all_good = tp + tn
all_bad = fp + fn
all_ = all_good + all_bad
print ' true pos rate: ' + str(tp/all_pos)
print ' false pos rate: ' + str(fp/all_neg)
print ' auc: ' + str(auc)

end = time.time()
print 'time elapsed: ' + str(end-start)
