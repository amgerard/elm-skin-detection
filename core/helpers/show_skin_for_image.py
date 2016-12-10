import Transform
from ELM import ELMRegressor
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import pickle

# slic
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# elm
# import elm

def getModel(useSaved = False):
	if useSaved == True:
		with open('elm.pkl', 'rb') as inpu:
			return pickle.load(inpu)

	# pre-transformed data
	X_train = np.loadtxt("results/_300s_1sig_minMax/X_train.csv", delimiter=",")
	y_train = np.loadtxt("results/_300s_1sig_minMax/y_train.csv", delimiter=",").astype(int)
        print np.bincount(y_train.astype(int))
        #print X_train.shape,y_train.shape
	#params = ["sigmoid", 0.5, 1300, False]
        #elmk = elm.ELMRandom(params)
	#tr_result = elmk.train(np.column_stack([X_train,y_train]))
	#return elmk
	ELM = ELMRegressor(1300)
        ELM.fit(X_train, y_train)
        #prediction = np.round(ELM.predict(X_train))
	return ELM

def seg_2_stats(seg):
	return np.array([np.mean(seg[:,0]), np.std(seg[:,0]), np.median(seg[:,0]),
		np.mean(seg[:,1]), np.std(seg[:,1]), np.median(seg[:,1]),
		np.mean(seg[:,2]), np.std(seg[:,2]), np.median(seg[:,2]),
		np.min(seg[:,0]), np.min(seg[:,1]), np.min(seg[:,2]),
		np.max(seg[:,0]), np.max(seg[:,1]), np.max(seg[:,2])])

def idxs_by_label(im_labels, x):
        return np.argwhere(im_labels == x)

def pixels_by_label(im, idxs):
	return im[idxs[:,0],idxs[:,1],:]

if __name__ == '__main__':
	# get ELM model
	ELM = getModel(True)
	numSegments = 300

	path = '../../../Original/train/'
	images = io.ImageCollection(path + '*.jpg')	
	# images = [io.imread(path + 'im03017.jpg')]
	for image in images:	
		# load the image and convert it to a floating point data type
		#image = io.imread(path + 'im03022.jpg')
		#image = io.imread('../../../Original/test/im03998.jpg')
		image = img_as_float(image)
		
		fig = plt.figure("orig")
		ax = fig.add_subplot(1, 1, 1)
		#ax.imshow(mark_boundaries(image, segments))
		ax.imshow(image)
		plt.axis("off")
		plt.show()

		# apply SLIC and extract (approximately) the supplied number
		# of segments
		segments = slic(image, n_segments = numSegments, sigma = 1.0)
		#segments = slic(image, n_segments = numSegments, sigma = 1, compactness=20)
		
		unique_labels = np.unique(segments)
		idxs = {x:idxs_by_label(segments,x) for x in unique_labels}
		segs = {x:pixels_by_label(image,idxs[x]) for x in unique_labels}
		X_test = np.vstack([seg_2_stats(segs[x]) for x in unique_labels])

		# show the output of SLIC
		fig = plt.figure("Superpixels -- %d segments" % (numSegments))
		ax = fig.add_subplot(1, 1, 1)
		ax.imshow(mark_boundaries(image, segments))
		plt.axis("off")	
		plt.show()

		# predict skin
		pred = ELM.predict(X_test)
		print pred
		av = 0.3 # pred.mean()
		skin = pred >= av
		not_skin = pred < av
		pred[skin] = 1
		pred[not_skin] = 0
		for i,x in enumerate(pred):
			#print x
			if x == 0:
				idx = idxs[unique_labels[i]]
				image[idx[:,0],idx[:,1],:] = 1

		# show skin
		fig = plt.figure("skin")
		ax = fig.add_subplot(1, 1, 1)
		#ax.imshow(mark_boundaries(image, segments))
		ax.imshow(image)
		plt.axis("off")
		plt.show()

	# save
	#with open('elm.pkl', 'wb') as output:
	#    pickle.dump(ELM, output, pickle.HIGHEST_PROTOCOL)
