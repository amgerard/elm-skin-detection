import numpy as np
import SuperPxlTransform

class SuperPxlParallelTestTransform(SuperPxlTransform):
    
    def seg_2_stats(self, seg, total, entrpy, entrpy2):
        statArr = SuperPxlTransform.seg_2_stats(self,seg,total,entrpy,entrpy2)
        return statArr.reshape(1,statArr.shape[0]).repeat(seg.shape[0],0)
    
    def im_mask(self,pair):
        im_labels = pair[0]
        mask = pair[1]
        unique_labels = np.unique(im_labels)
        return np.concatenate([self.mask_by_label(mask,im_labels,x) for x in unique_labels])
    
    def mask_by_label(self, mask, im_labels, x):
        idxs = np.argwhere(im_labels == x)
        return mask[idxs[:,0],idxs[:,1]]
