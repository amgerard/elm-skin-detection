import numpy as np
import skimage.io as io
from scipy import stats

# super-pixels
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import color

from ITransform import ITransform

class SuperPxlTransform(ITransform):
    
    def transform(self):
        im_labels = [self.get_im_labels(im) for im in self.images]
        ims_and_labels = zip(self.images, im_labels)

        results = [self.im_superpixels(pair) for pair in ims_and_labels]
        self.X = np.concatenate(results, axis=0)

        labels_and_masks = zip(im_labels, [self.im_2_mask(m) for m in self.masks])
        yy = [self.im_mask(pair) for pair in labels_and_masks]
        self.y = np.concatenate(yy, axis=0)
        
        y_stats = [self.im_mask_stats(pair) for pair in labels_and_masks]
        self.y_stats = np.concatenate(y_stats, axis=0)

    def get_im_labels(self,im):
        return slic(im, n_segments = 300, sigma = 1.0) #  compactness = 30.0) # 500, 5

    def im_superpixels(self,pair):
        im = pair[0]
        im_labels = pair[1]
        unique_labels = np.unique(im_labels)
        segs = {x:self.pixels_by_label(im,im_labels,x) for x in unique_labels}
        
        gray_im = color.rgb2gray(im);
        ent = entropy(gray_im, disk(5))
        entropy_by_lbl = {x:self.entropy_by_label(ent,im_labels,x) for x in unique_labels}
        
        total = float(im.shape[0] * im.shape[1])
        return np.vstack([self.seg_2_stats(segs[x],total,entropy_by_lbl[x][0],entropy_by_lbl[x][1]) for x in unique_labels])
    
    def pixels_by_label(self, im, im_labels, x):
        idxs = np.argwhere(im_labels == x)
        return im[idxs[:,0],idxs[:,1],:]
    
    def entropy_by_label(self, im, im_labels, x):
        idxs = np.argwhere(im_labels == x)
        return [np.mean(im[idxs[:,0],idxs[:,1]]),np.std(im[idxs[:,0],idxs[:,1]])]

    def seg_2_stats(self, seg, total, entrpy, entrpy_sd):
        return np.array([np.mean(seg[:,0]), np.std(seg[:,0]), np.median(seg[:,0]),
            np.mean(seg[:,1]), np.std(seg[:,1]), np.median(seg[:,1]),
            np.mean(seg[:,2]), np.std(seg[:,2]), np.median(seg[:,2]),
            np.min(seg[:,0]), np.min(seg[:,1]), np.min(seg[:,2]),
            np.max(seg[:,0]), np.max(seg[:,1]), np.max(seg[:,2]), 
            # (stats.skew(seg[:,0])+50)/100.0,(stats.skew(seg[:,1])+50)/100.0,(stats.skew(seg[:,2])+50)/100.0, # skew
            seg.shape[0]/total, entrpy/7.0, entrpy_sd/7.0]) # /3.0
    
    def im_mask(self,pair):
        im_labels = pair[0]
        mask = pair[1]
        unique_labels = np.unique(im_labels)
        return [self.mask_by_label(mask,im_labels,x) for x in unique_labels]

    def mask_stats(self,msk_by_lbl):
        bin_cnt = np.bincount(msk_by_lbl) # mode.mode[0]
        total = bin_cnt.sum()
        skin = total - bin_cnt[0]
        return float(skin) / total # mode.mode[0]
    
    def mask_by_label(self, mask, im_labels, x):
        idxs = np.argwhere(im_labels == x)
        msk_by_lbl = mask[idxs[:,0],idxs[:,1]]
        msk_percent = self.mask_stats(msk_by_lbl)
        return 1 if msk_percent > .5 else 0
        #mode = stats.mode(msk_by_lbl)
        #return mode.mode[0]

    def im_mask_stats(self,pair):
        im_labels = pair[0]
        mask = pair[1]
        unique_labels = np.unique(im_labels)
        return [self.stats_by_label(mask,im_labels,x) for x in unique_labels]
    
    def stats_by_label(self, mask, im_labels, x):
        idxs = np.argwhere(im_labels == x)
        msk_by_lbl = mask[idxs[:,0],idxs[:,1]]
        msk_percent = self.mask_stats(msk_by_lbl)
        return msk_percent
