#! /usr/bin/env python

import abc
import adel
import infi_learn as rr
import tensorflow as tf
import numpy as np


class NetworkWrapper(object):
    """Convenience class for training tensorflow networks. Simplifies
    enabling/disabling batch normalization and dropout and
    initializing feed dicts for tensorflow.

    Parameters
    ----------
    use_batch_norm : bool (default False)
        If true, enables batch normalization
    dropout_rate : float (default 0.0)
        If > 0, enables dropout
    kwargs : dict
        Architecture specifications to pass to network constructor
    """

    def __init__(self, use_batch_norm=False, dropout_rate=0.0, **kwargs):
        self.network_spec = kwargs
        self.batch_training = None
        if use_batch_norm:
            self.batch_training = tf.placeholder(
                tf.bool, name='batch_training')
            self.network_spec['batch_training'] = self.batch_training

        self.training_dropout = dropout_rate
        self.dropout_rate = None
        if self.training_dropout > 0.0:
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            self.network_spec['dropout_rate'] = self.dropout_rate

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
