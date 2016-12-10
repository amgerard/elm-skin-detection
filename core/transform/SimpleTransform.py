import numpy as np
import ITransform

class SimpleTransform(ITransform):
	def transform(self):
		self.X = np.concatenate([self.im_reshape(im) for im in self.images], axis=0)
		self.y = np.concatenate([self.im_2_mask(im) for im in self.masks], axis=0)
	
    def im_reshape(self,im):
		'''
		reshape so each pixel is a (r,b,g) row
		'''
		x,y,z = im.shape
		return im.reshape(x*y,z)
	
    def im_2_mask(self,im):
		mask2d = ITransform.im_2_mask(self,im)
		x,y = mask2d.shape
		return mask2d.reshape(x*y)
