"""Classes for learning embeddings
"""

import tensorflow as tf
import adel
import itertools
import utils as rru
import numpy as np


def get_embed_validation(model, dataset, feed):
    """Generates tf ops to produce embeddings and labels for a dataset

    Parameters
    ----------
    model : EmbeddingModel
        The model to use
    dataset : adel.SARSDataset
        Dataset to fully embed. Assumes states are (belief, img) tuples
    feed : dict
        Tensorflow feed dict to add fields to

    Returns
    -------
    ops : tensorflow operation
        Operation to run to get embeddings
    labels : list of bools
        True if positive class, false if negative, corresponding to op outputs
    """
    ops = model.embed(states=dataset.all_states + dataset.all_terminal_states,
                      feed=feed)
    labels = [True] * dataset.num_tuples + [False] * dataset.num_terminals
    return ops, np.array(labels)


class EmbeddingModel(object):
    """Wraps and provides methods for querying an embedding model.

    Parameters
    ----------
    img_size : 2-tuple or list of ints
        The input image width and height
    vec_size : int
        The input vector size
    scope    : string
        The scope in which to construct the model
    spec     : keyword args for adel.make_conv2d_joint_net
    """

    def __init__(self, img_size, vec_size, scope, spec):

        self.img_size = img_size
        self.vec_size = vec_size

        if not (self.using_image or self.using_vector):
            raise ValueError('Must use image and/or vector')

        self.scope = scope
        self.spec = spec

        self.image_ph = self.make_img_input(name='image')
        self.vector_ph = self.make_vec_input(name='vector')
        net, params, state, ups = self.make_net(img_in=self.image_ph,
                                                vec_in=self.vector_ph)
        self.net = net

    def __repr__(self):
        s = 'Embedding network:'
        for e in self.net:
            s += '\n\t%s' % str(e)
        return s

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
                                  name='%s%s' % (self.scope, name))

    def make_vec_input(self, name):
        """Creates a vector input placeholder
        """
        if not self.using_vector:
            return None
        else:
            return tf.placeholder(tf.float32,
                                  shape=[None, self.vec_size],
                                  name='%s%s' % (self.scope, name))

    def make_net(self, img_in=None, vec_in=None, reuse=False):
        """Creates the model network with the specified parameters
        """
        if self.using_image and not self.using_vector:
            return adel.make_conv2d_fc_net(img_in=img_in,
                                           scope=self.scope,
                                           reuse=reuse,
                                           **self.spec)
        elif not self.using_image and self.using_vector:
            return adel.make_fullycon(input=vec_in,
                                      scope=self.scope,
                                      reuse=reuse,
                                      **self.spec)
        else:
            return adel.make_conv2d_joint_net(img_in=img_in,
                                              vector_in=vec_in,
                                              scope=self.scope,
                                              reuse=reuse,
                                              **self.spec)

    def embed(self, states, feed):
        """Populates a feed dict and returns ops to perform an embedding
        using this model.

        Parameters
        ----------
        states : iterable of state tuples
        feed : dict
            Tensorflow feed dict to add fields to
        """
        if self.using_image and not self.using_vector:
            feed[self.image_ph] = rru.shape_data_2d(states)
        elif not self.using_image and self.using_vector:
            feed[self.vector_ph] = rru.shape_data_vec(states)
        else:
            # By convention (vecs, imgs)
            vecs, imgs = zip(*states)
            feed[self.image_ph] = rru.shape_data_2d(imgs)
            feed[self.vector_ph] = rru.shape_data_vec(vecs)
        return [self.net[-1]]


class EmbeddingTask(object):
    """Wraps an embedding network and learning optimization. Provides methods for
    using the embedding and sampling the dataset for aggregate learning.

    Parameters
    ----------
    model : EmbeddingModel
        The model to learn on
    dataset : adel.SARSDataset
        Training dataset object
    sep_dist : float
        Desired squared min embedding separation distance between disparate classes
    """

    def __init__(self, model, train_data, sep_dist):
        self.model = model

        self.train_dataset = train_data
        # TODO Not sure how we would change the sampling mode if we wanted to
        self.training_sampler = adel.SamplingInterface(self.train_dataset)

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
        self.loss = tf.reduce_mean(-tf.exp(delta_sq / sep_dist))

        # Huber with sep_dist as horizontal offset
        # self.loss = tf.losses.huber_loss(labels=-delta_sq + sep_dist,
        #                                 predictions=[0])

        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        with tf.control_dependencies(ups):
            self.train = opt.minimize(self.loss, var_list=params)

        self.initializers = [s.initializer for s in state] + \
            [adel.optimizer_initializer(opt, params)]

    def get_feed_training(self, feed, k):
        """Sample the dataset and add fields to a tensorflow feed dict.

        Parameters
        ----------
        feed : dict
            Feed dict to add fields to
        k    : int
            Number of datapoints to sample. If not enough datapoints,
            doesn't populate the feed dict. If -1, uses whole dataset.

        Returns
        -------
        ops : list of tensorflow operations
            List of the loss and training operation
        """
        return self.get_feed_dataset(feed=feed,
                                     k=k,
                                     dataset=self.train_dataset,
                                     train=True)

    def get_feed_dataset(self, feed, dataset, k=-1, train=False):
        """Populate a feed dict to compute loss and/or train on
        a dataset.

        Parameters
        ----------
        feed: dict
            Feed dict to add fields to
        dataset: adel.SARSDataset
            Dataset to use data from
        k : int (default -1)
            Number of datapoints to sample. If not enough datapoints,
            doesn't populate the feed dict. If -1, uses the whole dataset.
        train: bool (default False)
            Whether or not to include the learner train op

        Returns
        -------
        ops : list of tensorflow operations
            List of operations to run with a session to compute loss, and
            optionally step the learner
        """
        if k == -1:
            if dataset.num_tuples == 0 or dataset.num_terminals == 0:
                return []
            s = self.train_dataset.all_states
            st = self.train_dataset.all_terminal_states
        else:
            if dataset.num_tuples < k or dataset.num_terminals < k:
                return []
            s = self.training_sampler.sample_sars(k)[0]
            st = self.training_sampler.sample_terminals(k)[0]

        self._fill_feed(s, st, feed)
        out = [self.loss]
        if train:
            out.append(self.train)
        return out

    def _fill_feed(self, s, st, feed):
        """Helper to create class permutations and populate
        a feed dict.
        """
        sinds = range(len(s))
        stinds = range(len(st))
        neg_combos = list(itertools.product(sinds, stinds))
        neg_i, neg_is = zip(*neg_combos)

        bel, img = zip(*[s[i] for i in neg_i])
        feed[self.p_belief_ph] = rru.shape_data_vec(bel)
        feed[self.p_image_ph] = rru.shape_data_2d(img)
        bel_t, img_t = zip(*[st[i] for i in neg_is])
        feed[self.n_belief_ph] = rru.shape_data_vec(bel_t)
        feed[self.n_image_ph] = rru.shape_data_2d(img_t)
