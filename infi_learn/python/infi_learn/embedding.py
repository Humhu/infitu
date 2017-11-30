"""Classes for learning embeddings
"""

import tensorflow as tf
import adel
import itertools
import utils as rru
import numpy as np

# TODO How would we generalize the input architecture?
class EmbeddingLearner(object):
    """Wraps an embedding network and learning optimization. Provides methods for
    using the embedding and sampling the dataset for aggregate learning.

    Parameters
    ----------
    dataset : adel.SARSDataset
        Training dataset object
    img_size : 2-tuple or list of int
        Painted laser image width and height
    vec_size : int
        Belief state dimensionality
    scope : string
        Scope for this learner's tensorflow components
    spec : dict
        adel.conv2d_joint_net network specification dict
    sep_dist : float
        Desired squared min embedding separation distance between disparate classes
    validation : adel.SARSDataset (optional, default None)
        Another dataset to use for validation
    """

    def __init__(self, train_data, img_size, vec_size, scope, spec, sep_dist,
                 val_data=None):
        self.train_dataset = train_data
        # TODO Not sure how we would change the sampling mode if we wanted to
        self.training_sampler = adel.SamplingInterface(self.train_dataset)
        self.val_dataset = val_data
        self.image_ph = tf.placeholder(tf.float32,
                                       shape=[None, img_size[0],
                                              img_size[1], 1],
                                       name='%simage' % scope)
        self.belief_ph = tf.placeholder(tf.float32,
                                        shape=[None, vec_size],
                                        name='%sbelief' % scope)
        self.n_image_ph = tf.placeholder(tf.float32,
                                         shape=[None, img_size[0],
                                                img_size[1], 1],
                                         name='%sn_image' % scope)
        self.n_belief_ph = tf.placeholder(tf.float32,
                                          shape=[None, vec_size],
                                          name='%sn_belief' % scope)
        net, params, state, ups = adel.make_conv2d_joint_net(img_in=self.image_ph,
                                                             vector_in=self.belief_ph,
                                                             scope=scope,
                                                             reuse=False,
                                                             **spec)
        n_net = adel.make_conv2d_joint_net(img_in=self.n_image_ph,
                                           vector_in=self.n_belief_ph,
                                           scope=scope,
                                           reuse=True,
                                           **spec)[0]
        self.net = net
        self.n_net = n_net
        self.state = state

        delta = net[-1] - n_net[-1]
        delta_sq = tf.reduce_sum(delta * delta, axis=-1)
        self.loss = tf.losses.huber_loss(labels=-delta_sq + sep_dist,
                                         predictions=[0])

        opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        with tf.control_dependencies(ups):
            self.train = opt.minimize(self.loss, var_list=params)

        self.initializers = [s.initializer for s in state] + \
            [adel.optimizer_initializer(opt, params)]

    def __repr__(self):
        s = 'Embedding network:'
        for e in self.net:
            s += '\n\t%s' % str(e)
        return s

    def embed(self, imgs, vecs, feed):
        """Populates a feed dict and returns ops to perform an embedding
        using this model.

        Parameters
        ----------
        imgs : 2D, 3D, or 4D numpy array of floats
            Single or multiple channeled or unchanneled image inputs
        vecs : 1 or 2D numpy array or floats
            Single or multiple vector inputs
        feed : dict
            Tensorflow feed dict to add fields to
        """
        feed[self.image_ph] = rru.shape_data_2d(imgs)
        feed[self.belief_ph] = rru.shape_data_vec(vecs)

        return [self.net[-1]]

    def get_embed_validation(self, feed):
        """Generates embeddings for the validation dataset

        Parameters
        ----------
        feed : dict
            Tensorflow feed dict to add fields to

        Returns
        -------
        ops : tensorflow operation
            Operation to run to get embeddings
        labels : list of bools
            True if positive class, false if negative, corresponding to op outputs
        """
        if self.val_dataset is None or self.val_dataset.num_tuples == 0 \
            or self.val_dataset.num_terminals == 0:
            return []
        sb, si = zip(*self.val_dataset.all_states)
        stb, sti = zip(*self.val_dataset.all_terminal_states)
        feed[self.belief_ph] = rru.shape_data_vec(sb + stb)
        feed[self.image_ph] = rru.shape_data_2d(si + sti)

        labels = [True] * self.val_dataset.num_tuples \
        + [False] * self.val_dataset.num_terminals
        return self.net[-1], np.array(labels)

    def get_feed_training(self, feed, k):
        """Sample the dataset and add fields to a tensorflow feed dict.

        Parameters
        ----------
        feed : dict
            Feed dict to add fields to
        k    : int
            Number of datapoints to sample. If not enough datapoints,
            doesn't populate the dict.

        Returns
        -------
        ops : list of tensorflow operations
            List of the loss and training operation
        """
        if self.train_dataset.num_tuples < k or self.train_dataset.num_terminals < k:
            return []
        s = self.train_dataset.sample_sars(k)[0]
        st = self.train_dataset.sample_terminals(k)[0]
        self._fill_feed(s, st, feed)
        out = [self.loss, self.train]
        return out

    def get_feed_validation(self, feed):
        """Ports the validation dataset to a tensorflow feed dict.

        Parameters
        ----------
        feed : dict
            Feed dict to add fields to
        
        Returns
        -------
        ops : list of tensorflow operations
            List of the loss
        """
        if self.val_dataset is None or self.val_dataset.num_tuples == 0 \
            or self.val_dataset.num_terminals == 0:
            return []
        s = self.val_dataset.all_states
        st = self.val_dataset.all_terminal_states
        self._fill_feed(s, st, feed)
        out = [self.loss]
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
        feed[self.belief_ph] = rru.shape_data_vec(bel)
        feed[self.image_ph] = rru.shape_data_2d(img)
        bel_t, img_t = zip(*[st[i] for i in neg_is])
        feed[self.n_belief_ph] = rru.shape_data_vec(bel_t)
        feed[self.n_image_ph] = rru.shape_data_2d(img_t)
