"""Methods for data augmentation
"""

import random
import cv2
import numpy as np
from itertools import izip

def perturb_image(img, enable_flip=False, noise_mag=None):
    if not (isinstance(img, tuple) or isinstance(img, list)):
        img = [img]
    N = len(img)
    shape = list(img[0].shape)

    # Flip it
    if enable_flip:
        flip_ind = np.random.randint(-1,1,N)
        img = [cv2.flip(i, fi) for i, fi in izip(img, flip_ind)]

    # Add noise
    if noise_mag is not None:
        noise = np.random.normal(scale=noise_mag, size=[N] + shape)
        img = [i + n for i, n in izip(img, noise)]
    
    return img

def perturb_vector(vec, flip_mask=None, noise_mag=None):
    vec = np.atleast_2d(vec)
    N = len(vec)
    shape = len(vec[0])

    # Flip signs
    if flip_mask is not None:
        flips = np.random.choice([-1, 1], size=(N, shape))
        flips[:,np.array(flip_mask, dtype=bool)] = 1
        vec = np.multiply(flips, vec)

    # Add noise
    if noise_mag is not None:
        noise = np.random.normal(scale=noise_mag, size=(N, shape))
        vec = vec + noise

    return vec

class DataAugmenter(object):
    """Provides methods for perturbing data randomly
    """
    def __init__(self, image=None, vector=None):
        
        if image is None:
            self.img_args = None
        else:
            self.img_args = image
        
        if vector is None:
            self.vec_args = None
        else:
            self.vec_args = vector

    def augment_data(self, data):
        if self.img_args is not None and self.vec_args is not None:
            vecs, imgs = zip(*data)
            vecs = perturb_vector(vec=vecs, **self.vec_args)
            imgs = perturb_image(img=imgs, **self.img_args)
            return zip(vecs, imgs)
        elif self.img_args is not None and self.vec_args is None:
            return perturb_image(img=data, **self.img_args)
        elif self.img_args is None and self.vec_args is not None:
            return perturb_vector(vec=data, **self.vec_args)
        else:
            raise RuntimeError('Must use vec or img')