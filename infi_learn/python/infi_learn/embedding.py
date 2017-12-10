"""Classes for learning embeddings
"""
import itertools

import numpy as np
import tensorflow as tf

import adel
import rospy

import utils as rru
import networks as rrn


class EmbeddingNetwork(rrn.NetworkBase):
    """Wraps and provides methods for querying an embedding network.

    Parameters
    ----------
    img_size : 2-tuple or list of ints
        The input image width and height
    vec_size : int
        The input vector size
    kwargs   : keyword args for NetworkBase
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
        s = 'Embedding network:'
        for e in self.net:
            s += '\n\t%s' % str(e)
        return s

    def initialize(self, sess):
        sess.run([p.initializer for p in self.params], feed_dict={})

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

    # TODO Possibly separate embed out from the network itself?
    def get_embed_ops(self, states, feed):
        """Populates a feed dict and returns ops to perform an embedding
        using this model.

        Parameters
        ----------
        states : iterable of state tuples
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

    def embed_dataset(self, data, feed):
        """Populates a feed dict with all states from a dataset and returns 
        ops to perform an embedding using this model.

        Parameters
        ----------
        model : EmbeddingNetwork
            The model to use
        data : adel.SARSDataset
            Dataset to fully embed. Assumes states are (belief, img) tuples
        feed : dict
            Tensorflow feed dict to add fields to

        Returns
        -------
        ops : tensorflow operation
            Operation to run to get embeddings
        labels : list of bools
        """
        states = data.all_states + data.all_terminal_states
        embed = self.get_embed_ops(states=states, feed=feed)
        labels = [True] * data.num_sars + [False] * data.num_terminals
        return embed, np.array(labels)


class EmbeddingProblem(object):
    """Wraps an embedding network and learning optimization. Provides methods for
    using the embedding and sampling the dataset for aggregate learning.

    Parameters
    ----------
    model : EmbeddingNetwork
        The model to learn on
    sep_dist : float
        Desired squared min embedding separation distance between disparate classes
    """
    # TODO Make loss a parameter?

    def __init__(self, model, separation_distance):
        self.model = model

        self.p_image_ph = model.make_img_input(name='p_image')
        self.p_belief_ph = model.make_vec_input(name='p_belief')
        self.n_image_ph = model.make_img_input(name='n_image')
        self.n_belief_ph = model.make_vec_input(name='n_belief')

        self.p_net, params, state, ups = model.make_net(img_in=self.p_image_ph,
                                                        vec_in=self.p_belief_ph,
                                                        reuse=True)
        self.n_net = model.make_net(img_in=self.n_image_ph,
                                    vec_in=self.n_belief_ph,
                                    reuse=True)[0]
        self.state = state

        # Loss function penalty for different classes being near each other
        delta = self.p_net[-1] - self.n_net[-1]
        delta_sq = tf.reduce_sum(delta * delta, axis=-1)

        # Squared exponential with sep_dist as bandwidth
        self.sep_dist = separation_distance
        # self.loss = tf.reduce_mean(-tf.exp(delta_sq / self.sep_dist))

        # Huber with sep_dist as horizontal offset
        self.loss = tf.losses.huber_loss(labels=-delta_sq + self.sep_dist,
                                         predictions=[0])

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

    def run_embedding(self, sess, data):
        feed = self.model.init_feed(training=False)
        ops, labels = self.model.embed_dataset(data=data, feed=feed)
        embed = sess.run(ops, feed_dict=feed)
        return embed, labels

    def _fill_feed(self, data, feed):
        """Helper to create class permutations and populate
        a feed dict.
        """
        sinds = range(data.num_sars)
        stinds = range(data.num_terminals)
        neg_combos = list(itertools.product(sinds, stinds))
        neg_i, neg_is = zip(*neg_combos)

        bel, img = self.model.parse_states([data.all_states[i] for i in neg_i])
        bel_t, img_t = self.model.parse_states(
            [data.all_terminal_states[i] for i in neg_is])

        if self.model.using_vector:
            feed[self.p_belief_ph] = bel
            feed[self.n_belief_ph] = bel_t

        if self.model.using_image:
            feed[self.p_image_ph] = img
            feed[self.n_image_ph] = img_t
