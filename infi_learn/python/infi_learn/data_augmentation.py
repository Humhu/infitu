"""Methods for data augmentation
"""

import random
import cv2
import tensorflow as tf
import numpy as np
from itertools import izip


# def perturb_image(img, enable_flip=False, noise_mag=None):
#     if not (isinstance(img, tuple) or isinstance(img, list)):
#         img = [img]
#     N = len(img)
#     shape = list(img[0].shape)

#     # Flip it
#     # 0 corresponds to x-flip
#     # 1 corresponds to y-flip
#     # -1 corresponds to x and y flip
#     # 2 corresponds to no flip
#     if enable_flip:
#         flip_inds = np.random.randint(-1, 2, N)
#         img = [cv2.flip(i, fi) if fi != 2 else i
#                for i, fi in izip(img, flip_inds)]

#     # Add noise
#     if noise_mag is not None:
#         noise = np.random.normal(scale=noise_mag, size=[N] + shape)
#         img = [i + n for i, n in izip(img, noise)]

#     return img

# def perturb_vector(vec, flip_mask=None, noise_mag=None):
#     vec = np.atleast_2d(vec)
#     N = len(vec)
#     shape = len(vec[0])

#     # Flip signs
#     if flip_mask is not None:
#         flips = np.random.choice([-1, 1], size=(N, shape))
#         flips[:, np.array(flip_mask, dtype=bool)] = 1
#         vec = np.multiply(flips, vec)

#     # Add noise
#     if noise_mag is not None:
#         noise = np.random.normal(scale=noise_mag, size=(N, shape))
#         vec = vec + noise

#     return vec


class DataAugmenter(object):
    """Provides methods for perturbing data randomly
    """

    def __init__(self, image_ph, vector_ph, labels_ph,
                 image_flip_vert=False, image_flip_hor=False, image_noise_sd=0.0,
                 vector_flip_mask=None, vector_noise_sd=0.0,
                 label_noise_sd=0.0):

        self.image = image_ph
        self.vector = vector_ph
        self.labels = labels_ph

        # TODO More operations?
        if self.image is not None:
            if image_noise_sd > 0.0:
                self.image = self.image + tf.truncated_normal(shape=tf.shape(self.image),
                                                              stddev=image_noise_sd)
            if image_flip_vert:
                self.image = tf.map_fn(lambda img: tf.image.random_flip_up_down(img),
                                       self.image)
            if image_flip_hor:
                self.image = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                                       self.image)

        if self.vector is not None:
            if vector_noise_sd > 0.0:
                self.vector = self.vector + tf.truncated_normal(shape=tf.shape(self.vector),
                                                                stddev=vector_noise_sd)
            if vector_flip_mask is not None:
                # TODO test this...
                flips = tf.round(tf.random_uniform(shape=tf.shape(self.vector),
                                                   minval=0,
                                                   maxval=vector_flip_mask))
                signs = tf.pow(tf.constant(-1.0, dtype=tf.float32), flips)
                self.vector = tf.multiply(signs, self.vector)

        if self.labels is not None:
            if label_noise_sd > 0.0:
                self.labels = self.labels + tf.truncated_normal(shape=tf.shape(self.labels),
                                                                stddev=label_noise_sd)

    def augment_data(self, sess, feed):
        """Retrieve the vector and image tensors depending on if we are training or not
        """
        if self.image is not None and self.vector is not None:
            vec, img, lab = sess.run([self.vector, self.image, self.labels], feed_dict=feed)
            state = zip(vec, img)
            return state, lab
        elif self.image is None and self.vector is not None:
            return sess.run([self.vector, self.labels], feed_dict=feed)
        elif self.image is not None and self.vector is None:
            return sess.run([self.image, self.labels], feed_dict=feed)
        else:
            raise RuntimeError('Must specify either image or vector')

    # def augment_data(self, data):
    #     # Use synchronized flip indices for both perturbations to preserve relations
    #     # 0 corresponds to x-flip, 1 corresponds to y-flip, -1 corresponds to x and y flip,
    #     # 2 corresponds to no flip
    #     if self.img_args is not None and self.vec_args is not None:
    #         vecs, imgs = zip(*data)
    #         vecs = perturb_vector(vec=vecs, **self.vec_args)
    #         imgs = perturb_image(img=imgs, **self.img_args)
    #         return zip(vecs, imgs)
    #     elif self.img_args is not None and self.vec_args is None:
    #         return perturb_image(img=data, **self.img_args)
    #     elif self.img_args is None and self.vec_args is not None:
    #         return perturb_vector(vec=data, **self.vec_args)
    #     else:
    #         raise RuntimeError('Must use vec or img')
