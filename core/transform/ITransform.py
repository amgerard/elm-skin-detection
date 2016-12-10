from abc import ABCMeta, abstractmethod
import numpy as np
import skimage.io as io
from skimage.util import img_as_float

#class ITransform(metaclass=ABCMeta):
class ITransform():
    def __init__(self, images, masks):
        self.images = [img_as_float(im) for im in images]
        self.masks = [img_as_float(mk) for mk in masks]
        self.X = np.ones(1);
        self.y = np.ones(1);
        self.y_stats = np.ones(1);

    @abstractmethod
    def transform(self):
        pass

    def im_2_mask(self,im):
        '''
        covert mask to 1's and 0's
        '''
        mu = np.mean(im, axis=2)
        mask2d = mu < 1 # 255
        return mask2d.astype(int)
