"""Classes for learning state-value functions
"""

import numpy as np
import tensorflow as tf

import adel

import networks as rrn
import utils as rru

# NOTE This is pretty much identical to EmbeddingNetwork... consolidate them!


class ValueNetwork(rrn.NetworkBase):
    """Wraps and provides methods for querying a state-value network
    """

    def __init__(self, img_size, vec_size, **kwargs):
        rrn.NetworkBase.__init__(self, **kwargs)
        self.img_size = img_size
        self.vec_size = vec_size

        if not (self.using_image or self.using_vector):
            raise ValueError('Must use image and/or vector')

        self.image_ph = self.make_img_input(name='image')
        self.vector_ph = self.make_vec_input(name='vector')
        net, params, state, ups = self.make_net(img_in=self.image_ph,
                                                vec_in=self.vector_ph)
        self.net = net
        self.params = params

    def __repr__(self):
        s = 'State-value network:'
        for e in self.net:
            s += '\n\t%s' % str(e)
        return s

    def print_filters(self, sess):
        s = 'L0 filters:'
        filters = sess.run(self.params[0])
        print filters.shape
        for f in filters:
            s += '\n\t%s' % str(f)
        return s


    def initialize(self, sess):
        sess.run([p.initializer for p in self.params], feed_dict={})

    def parse_states(self, states):
        if self.using_vector and self.using_image:
            vec, img = zip(*states)
            vec = rru.shape_data_vec(vec)
            img = rru.shape_data_2d(img)
        elif self.using_vector and not self.using_image:
            vec = rru.shape_data_vec(states)
            img = None
        elif not self.using_vector and self.using_image:
            vec = None
            img = rru.shape_data_2d(states)
        else:
            raise ValueError('Must use vector and/or image')
        return vec, img

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

    def make_net(self, img_in=None, vec_in=None, reuse=False):
        """Creates the model network with the specified parameters
        """
        if self.using_image and not self.using_vector:
            return adel.make_conv2d_fc_net(img_in=img_in,
                                           reuse=reuse,
                                           **self.network_spec)
        elif not self.using_image and self.using_vector:
            return adel.make_fullycon(input=vec_in,
                                      reuse=reuse,
                                      **self.network_args)
        else:
            return adel.make_conv2d_joint_net(img_in=img_in,
                                              vector_in=vec_in,
                                              reuse=reuse,
                                              **self.network_args)

    def get_value_ops(self, states, feed):
        """Populates a feed dict and returns ops to perform an embedding
        using this model.

        Parameters
        ----------
        states : iterable of states
        feed : dict
            Tensorflow feed dict to add fields to

        Returns
        -------
        ops : tensorflow operation
            Operation to run to get embeddings
        """
        if len(states) == 0:
            return None

        if self.using_image and not self.using_vector:
            feed[self.image_ph] = rru.shape_data_2d(states)
        elif not self.using_image and self.using_vector:
            feed[self.vector_ph] = rru.shape_data_vec(states)
        else:
            # By convention (vecs, imgs)
            vecs, imgs = zip(*states)
            feed[self.image_ph] = rru.shape_data_2d(imgs)
            feed[self.vector_ph] = rru.shape_data_vec(vecs)
        return self.net[-1]


class BanditValueProblem(object):
    """Wraps a value network and learning optimization.
    """

    def __init__(self, model, **kwargs):
        self.model = model

        self.image_ph = model.image_ph #model.make_img_input(name='image')
        self.vec_ph = model.vector_ph #model.make_vec_input(name='vector')
        self.value_ph = tf.placeholder(tf.float32,
                                       shape=[None, 1],
                                       name='%s/values' % model.scope)
        self.net, params, state, ups = model.make_net(img_in=self.image_ph,
                                                      vec_in=self.vec_ph,
                                                      reuse=True)

        self.loss = tf.losses.mean_squared_error(labels=self.value_ph,
                                                 predictions=self.net[-1])

        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        with tf.control_dependencies(ups):
            self.train = opt.minimize(self.loss, var_list=params)

        self.initializers = [s.initializer for s in state] + \
            [adel.optimizer_initializer(opt, params)]

    def initialize(self, sess):
        sess.run(self.initializers)

    def run_training(self, sess, data):
        feed = self.model.init_feed(training=True)
        self._fill_feed(data=data, feed=feed)
        return sess.run([self.loss, self.train], feed_dict=feed)[0]

    def run_loss(self, sess, data):
        feed = self.model.init_feed(training=False)
        self._fill_feed(data=data, feed=feed)
        return sess.run(self.loss, feed_dict=feed)

    def run_output(self, sess, data):
        feed = self.model.init_feed(training=False)
        ops = self.model.get_value_ops(states=data.all_inputs,
                                               feed=feed)
        return sess.run(ops, feed_dict=feed)

    def _fill_feed(self, data, feed):
        """Helper to create class permutations and populate
        a feed dict.
        """

        vec, img = self.model.parse_states(data.all_inputs)
        if self.model.using_vector:
            feed[self.vec_ph] = vec
        if self.model.using_image:
            feed[self.image_ph] = img

        feed[self.value_ph] = np.atleast_2d(data.all_labels).T
