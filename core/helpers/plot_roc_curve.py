import time
start = time.time()

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
from ELM import ELMRegressor

def predictY(ELM, train):
    predict = ELM.predict(train)
    skin = predict >= 0.5
    not_skin = predict < 0.5
    predict[skin] = 1
    predict[not_skin] = 0
    return predict

def savePlot(traces, traceNames, yLabel, useYLim, title, filename):
        plt.figure()
        colors = ['b', 'r', 'g']
        for i, y in enumerate(traces):
            plt.plot(y, 'o-', c=colors[i], label=traceNames[i])
        plt.legend(loc='upper left', fontsize='small')
        plt.ylabel(yLabel)
        plt.xlabel("Number of Features (d)")
        plt.grid(which='both')
        if useYLim:
                plt.ylim([0,20])
        plt.title(title)
        with PdfPages(filename) as pdf:
                pdf.savefig()
                plt.close()

def frange(x, y, jump):
	while x < y:
		yield x
		x += jump

# same transformed data
path = 'results/_300s_1sig_entropy/'
X_train = np.loadtxt(path + "X_train.csv", delimiter=",")
y_train = np.loadtxt(path + "y_train.csv", delimiter=",").astype(int)
X_test = np.loadtxt(path + "X_test.csv", delimiter=",")
y_test = np.loadtxt(path + "y_test.csv", delimiter=",").astype(int)

for x in range(400, 1400, 100):
	if x < 1300:
		continue

	# train model
	ELM = ELMRegressor(x)
	ELM.fit(X_train, y_train)

	predict = predictY(ELM,X_train)
	prediction = np.ones(predict.shape)
	thresh = 0.3
	skin = predict >= thresh
	not_skin = predict < thresh
	prediction[skin] = 1
	prediction[not_skin] = 0
	print 'train accuracy: ' + str(accuracy_score(y_train, prediction))


	# print
	#print x
	#print np.bincount(prediction.astype(int))
	#print np.bincount(y_test.astype(int))

	# print 'train error: ' + str(mean_absolute_error(y_train, prediction))
	#print 'train accuracy: ' + str(accuracy_score(y_train, prediction))

	# prediction = predictY(ELM,X_test)
    	predict = ELM.predict(X_test)

	fpr = []
	tpr = []	

	for thresh in frange(0,1.0,0.05):
		prediction = np.ones(predict.shape)
		skin = predict >= thresh
		not_skin = predict < thresh
		prediction[skin] = 1
		prediction[not_skin] = 0

		print ''
		print thresh

		print ' test error: ' + str(mean_absolute_error(y_test, prediction))
		print ' test accuracy: ' + str(accuracy_score(y_test, prediction))

		#print y_test.shape, prediction.shape

		conf = confusion_matrix(y_test,prediction).astype(float)
		auc2 = roc_auc_score(y_test,prediction)
		# print conf
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
		print ' auc: ' + str(auc2)

		false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, prediction, pos_label=1)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		
		fpr.append(false_positive_rate[1])
		tpr.append(true_positive_rate[1])
	
	roc_auc = 0.89
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, '-o', label='AUC = %0.2f'% roc_auc)
	plt.legend(loc='upper right', fontsize='small')
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([0.0,1.0])
	plt.ylim([0.0,1.0])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.grid(which='both')
	plt.show()

	end = time.time()
	print 'time elapsed: ' + str(end-start)
