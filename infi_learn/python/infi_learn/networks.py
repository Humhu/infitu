#! /usr/bin/env python

import abc
import numpy as np

import tensorflow as tf

import adel
import utils as rru


class NetworkBase(object):
    """Convenience class for wrapping/training tensorflow networks. Simplifies
    enabling/disabling batch normalization and dropout and
    initializing feed dicts for tensorflow.

    Parameters
    ----------
    scope : string
        This network's base scope
    use_batch_norm : bool (default False)
        If true, enables batch normalization
    dropout_rate : float (default 0.0)
        If > 0, enables dropout
    kwargs : dict
        Architecture specifications to pass to network constructor
    """

    def __init__(self, scope, use_batch_norm=False, dropout_rate=0.0, **kwargs):

        self.network_spec = kwargs
        self.network_spec['scope'] = scope

        self.batch_training = None
        if use_batch_norm:
            self.batch_training = tf.placeholder(
                tf.bool, name='%s/batch_training' % scope)
            self.network_spec['batch_training'] = self.batch_training

        self.training_dropout = dropout_rate
        self.dropout_rate = None
        if self.training_dropout > 0.0:
            self.dropout_rate = tf.placeholder(
                tf.float32, name='%s/dropout_rate' % scope)
            self.network_spec['dropout_rate'] = self.dropout_rate

    @property
    def scope(self):
        return self.network_spec['scope']

    @property
    def network_args(self):
        """Returns a kwarg dict to construct a neural network. Note that a copy
        of the internal field is returned.
        """
        return dict(self.network_spec)

    def init_feed(self, training):
        """Initializes a feed dict for training or execution.

        Parameters
        ----------
        training : bool
            Whether or not in training mode
        """
        feed = {}

        if self.batch_training is not None:
            feed[self.batch_training] = training

        if self.dropout_rate is not None:
            if training:
                feed[self.dropout_rate] = self.training_dropout
            else:
                feed[self.dropout_rate] = 0.0

        return feed


class VectorImageNetwork(NetworkBase):
    """Base implementation of a network with optional vector/image inputs

    Parameters
    ----------
    img_size : 2-tuple or list of ints or None (default None)
        The input image width and height, or None if not using image
    vec_size : int or None (default None)
        The input vector size, or None if not using vector
    kwargs   : keyword args for NetworkBase
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, img_size=None, vec_size=None, **kwargs):
        NetworkBase.__init__(self, **kwargs)
        self.img_size = img_size
        self.vec_size = vec_size

        if not (self.using_image or self.using_vector):
            raise ValueError('Must use image and/or vector')

        #self.image_ph = self.make_img_input(name='image')
        #self.vector_ph = self.make_vec_input(name='vector')
        #self.net, self.params, self.state, self.ups = self.make_net(img_in=self.image_ph,
        #                                                            vec_in=self.vector_ph)

    #def initialize(self, sess):
    #    sess.run([p.initializer for p in self.params], feed_dict={})

    def parse_states(self, states):
        if self.using_vector and self.using_image:
            bel, img = zip(*states)
            bel = rru.shape_data_vec(bel)
            img = rru.shape_data_2d(img)
        elif self.using_vector and not self.using_image:
            bel = rru.shape_data_vec(states)
            img = None
        elif not self.using_vector and self.using_image:
            bel = None
            img = rru.shape_data_2d(states)
        else:
            raise ValueError('Must use vector and/or image')
        return bel, img

    @property
    def using_image(self):
        return not self.img_size is None

    @property
    def using_vector(self):
        return not self.vec_size is None

    def make_img_input(self, name):
        """Creates an image input placeholder
        """
        if not self.using_image:
            return None
        else:
            return tf.placeholder(tf.float32,
                                  shape=[None, self.img_size[0],
                                         self.img_size[1], 1],
                                  name='%s/%s' % (self.scope, name))

    def make_vec_input(self, name):
        """Creates a vector input placeholder
        """
        if not self.using_vector:
            return None
        else:
            return tf.placeholder(tf.float32,
                                  shape=[None, self.vec_size],
                                  name='%s/%s' % (self.scope, name))

    @abc.abstractmethod
    def make_net(self, img_in=None, vec_in=None, reuse=False):
        """Creates the model network with the specified parameters. To be 
        implemented by a derived class.
        """
        pass

    def get_ops(self, inputs, img_ph, vec_ph, feed):
        """Populates a feed dict and returns ops to perform an embedding
        using this model.

        Parameters
        ----------
        inputs : iterable of state tuples
        feed : dict
            Tensorflow feed dict to add fields to

        Returns
        -------
        ops : tensorflow operation
            Operation to run to get output
        """
        if len(inputs) == 0:
            return None

        if self.using_image and not self.using_vector:
            feed[img_ph] = rru.shape_data_2d(inputs)
        elif not self.using_image and self.using_vector:
            feed[vec_ph] = rru.shape_data_vec(inputs)
        else:
            # By convention (vecs, imgs)
            vecs, imgs = zip(*inputs)
            feed[img_ph] = rru.shape_data_2d(imgs)
            feed[vec_ph] = rru.shape_data_vec(vecs)
