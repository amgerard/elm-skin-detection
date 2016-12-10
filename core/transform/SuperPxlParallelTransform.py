import numpy as np
import SuperPxlTransform

class SuperPxlParallelTransform(SuperPxlTransform):
    def transform(self):
        im_labels = [SuperPxlTransform.get_im_labels(im) for im in self.images]
        
        ims_and_labels = zip(self.images, im_labels)
        results = [SuperPxlTransform.im_superpixels(pair) for pair in ims_and_labels]
        self.X = np.concatenate(results, axis=0)
        
        labels_and_masks = zip(im_labels, [self.im_2_mask(m) for m in self.masks])
        yy = [self.im_mask(pair) for pair in labels_and_masks]
        self.y = np.concatenate(yy, axis=0)
