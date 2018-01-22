"""Classes for learning embeddings
"""
import itertools

import numpy as np
import tensorflow as tf

import adel
import rospy

import utils as rru
import networks as rrn

from infi_learn.learning import Learner
from infi_learn.plotting import LineSeriesPlotter, FilterPlotter, ScatterPlotter
from infi_learn.data_augmentation import DataAugmenter


class EmbeddingNetwork(rrn.VectorImageNetwork):
    """Derived VI network for learning embeddings

    Parameters
    ----------
    img_size : 2-tuple or list of ints
        The input image width and height
    vec_size : int
        The input vector size
    kwargs   : keyword args for NetworkBase
    """

    # TODO Have embedding_dimension as an argument and add a final
    # layer on the tail end of the created networks to bring it to that
    # dimension with the appropriate rectification?
    def __init__(self, **kwargs):
        rrn.VectorImageNetwork.__init__(self, **kwargs)

    # def __repr__(self):
    #     s = 'Embedding network:'
    #     for e in self.net:
    #         s += '\n\t%s' % str(e)
    #     return s

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


class EmbeddingProblem(object):
    """Wraps an embedding network and learning optimization. Provides methods for
    using the embedding.

    Parameters
    ==========
    model : EmbeddingNetwork
        The model to learn on
    bandwidth : float
        Desired squared min embedding separation distance between disparate classes
    """
    # TODO Make loss a parameter?

    def __init__(self, model, loss_type, augmentation, learning, embedding_loss_clip=1.0,
                 label_dim=1, distance_scale=1.0, attraction_k=1e-3, anchor_weight=1e-6):
        self.model = model

        self.p_image_ph = model.make_img_input(name='p_image')
        self.p_belief_ph = model.make_vec_input(name='p_belief')
        self.net, self.params, state, ups = model.make_net(img_in=self.p_image_ph,
                                                           vec_in=self.p_belief_ph,
                                                           reuse=False)

        self.augmenter = DataAugmenter(image_ph=self.p_image_ph,
                                       vector_ph=self.p_belief_ph,
                                       **augmentation)

        self.p_labels_ph = tf.placeholder(tf.float32,
                                          name='%s/p_labels' % self.model.scope,
                                          shape=[None, label_dim])

        # Generate combinations, courtesy of stackoverflow
        X = self.net[-1]
        r = tf.reduce_sum(X * X, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        x_dist = r - 2 * tf.matmul(X, tf.transpose(X)) + tf.transpose(r)
        # x_dist = tf.sqrt(x_dist + 1e-16)

        y = self.p_labels_ph
        q = tf.reduce_sum(y * y, 1)
        q = tf.reshape(q, [-1, 1])
        y_dist = q - 2 * tf.matmul(y, tf.transpose(y)) + tf.transpose(q)
        # y_dist = tf.sqrt(y_dist + 1e-16)

        self.state = state
        if loss_type == 'relu':
            zero_point = y_dist * float(distance_scale)
            err = x_dist - zero_point

            # squared penalty for dissimilar points being too close
            close_loss = tf.nn.relu(-err)

            # penalty for similar points being too far
            far_loss = float(attraction_k) * tf.nn.relu(err)
            embedding_loss = tf.reduce_mean(close_loss + far_loss)
        elif loss_type == 'huber':
            # Huber with y_delta_sq * bandwidth as horizontal offset
            embedding_loss = tf.losses.huber_loss(labels=y_dist * float(distance_scale),
                                                  predictions=x_dist)
        elif loss_type == 'repel':
            # Penalize all points, scaled by label difference
            # Epsilon in denominator to avoid division by zero
            embedding_loss = tf.reduce_mean(y_dist / (x_dist + 1e-6))
        else:
            raise ValueError('Unknown loss type: %s' % loss_type)

        # Clip embedding loss norm
        clipped_embed_loss = tf.clip_by_norm(t=embedding_loss,
                                             clip_norm=float(embedding_loss_clip))

        # Penalize points for being too far from zero
        anchor_loss = tf.reduce_mean(tf.norm(X, axis=-1))

        self.loss = float(anchor_weight) * anchor_loss + clipped_embed_loss

        self.learner = Learner(variables=self.params, updates=ups, loss=self.loss,
                               **learning)

        self.initializers = [s.initializer for s in state] + self.learner.init

    def initialize(self, sess):
        sess.run(self.initializers)

    def run_training(self, sess, ins, outs):
        # TODO Make this cleaner somehow
        feed = self.model.init_feed(training=True)
        self._fill_feed(ins=ins, outs=outs, training=True, feed=feed)
        ins = self.augmenter.augment_data(sess=sess, feed=feed)
        self._fill_feed(ins=ins, outs=outs, training=True, feed=feed)
        return self.learner.step(sess=sess, feed=feed)
        # return sess.run([self.loss, self.train], feed_dict=feed)[0]

    def run_loss(self, sess, ins, outs):
        feed = self.model.init_feed(training=False)
        self._fill_feed(ins=ins, outs=outs, training=False, feed=feed)
        return sess.run(self.loss, feed_dict=feed)

    def run_embedding(self, sess, ins):
        feed = self.model.init_feed(training=False)
        self.model.get_ops(inputs=ins,
                           img_ph=self.p_image_ph,
                           vec_ph=self.p_belief_ph,
                           feed=feed)
        return sess.run(self.net[-1], feed_dict=feed)

    def _fill_feed(self, ins, outs, training, feed):
        """Helper to create class permutations and populate
        a feed dict.
        """

        self.model.get_ops(inputs=ins,
                           img_ph=self.p_image_ph,
                           vec_ph=self.p_belief_ph,
                           feed=feed)
        # TODO Needed .T for 1D case
        feed[self.p_labels_ph] = np.atleast_2d(outs)


class EmbeddingLearner(object):
    """Maintains dataset splits and plotting around an EmbeddingProblem
    """

    def __init__(self, problem, holdout,
                 batch_size=10, iters_per_spin=10, validation_period=10):
        self.problem = problem

        self.train_base = adel.BasicDataset()
        self.val_base = adel.BasicDataset()
        self.holdout = adel.HoldoutWrapper(training=self.train_base,
                                           holdout=self.val_base,
                                           **holdout)

        self.train_sampler = adel.DatasetSampler(base=self.train_base,
                                                 method='uniform')
        self.train_sampled = adel.LabeledDatasetTranslator(
            base=self.train_sampler)
        self.val_sampler = adel.DatasetSampler(base=self.val_base,
                                               method='uniform')
        self.val_sampled = adel.LabeledDatasetTranslator(base=self.val_sampler)

        self.train_data = adel.LabeledDatasetTranslator(base=self.train_base)
        self.val_data = adel.LabeledDatasetTranslator(base=self.val_base)
        self.reporter = adel.LabeledDatasetTranslator(base=self.holdout)

        self.iters_per_spin = iters_per_spin
        self.val_period = validation_period
        self.batch_size = batch_size

        self.spin_counter = 0

        self.error_plotter = LineSeriesPlotter(title='Embedding losses over time %s' % self.scope,
                                               xlabel='Spin iter',
                                               ylabel='Embedding loss')
        self.embed_plotter = ScatterPlotter(title='Validation embeddings %s' %
                                            self.problem.model.scope)
        self.plottables = [self.error_plotter, self.embed_plotter]

        if self.problem.model.using_image:
            # TODO HACK
            self.filters = self.problem.params[0]
            n_filters = int(self.filters.shape[-1])
            self.filter_plotter = FilterPlotter(n_filters)
            self.plottables.append(self.filter_plotter)

    @property
    def scope(self):
        return self.problem.model.scope

    def report_state_value(self, state, value):
        self.reporter.report_label(x=state, y=value)

    def get_plottables(self):
        """Return a list of the plottables used by this learner
        """
        return self.plottables

    def get_embedding(self, sess, on_training_data=True):
        """Returns the embedding values for training or validation data
        """
        if on_training_data:
            data = self.train_base
        else:
            data = self.val_base

        chunker = adel.DatasetChunker(base=data, block_size=500)  # TODO
        x = [self.problem.run_embedding(sess=sess, ins=adel.LabeledDatasetTranslator(chunk).all_inputs)
             for chunk in chunker.iter_subdata(key=None)]  # HACK key
        if len(x) == 0:
            x = []
        else:
            x = np.vstack(x)

        y = adel.LabeledDatasetTranslator(data).all_labels
        return x, y

    def get_status(self):
        """Returns a string describing the status of this learner
        """
        # TODO More information
        status = 'Embedding %s has %d/%d (train/validation) datapoints' % \
            (self.scope, self.train_data.num_data, self.val_data.num_data)
        return status

    def spin(self, sess):
        """Executes a train/validate cycle
        """

        train_loss = None
        for _ in range(self.iters_per_spin):
            if self.train_data.num_data < self.batch_size:
                print 'Not enough data to train, skipping...'
                break

            self.train_sampler.sample_data(key=None, k=self.batch_size)
            train_loss = self.problem.run_training(sess=sess,
                                                   ins=self.train_sampled.all_inputs,
                                                   outs=self.train_sampled.all_labels)

        self.spin_counter += 1
        if train_loss is None:
            return train_loss

        print 'Training iter: %d loss: %f ' % \
            (self.spin_counter * self.iters_per_spin, train_loss)
        self.error_plotter.add_line('training', self.spin_counter, train_loss)

        if self.spin_counter % self.val_period == 0:

            # Plot filters if using them
            if self.problem.model.using_image:
                fil_vals = np.squeeze(sess.run(self.filters))
                fil_vals = np.rollaxis(fil_vals, -1)
                self.filter_plotter.set_filters(fil_vals)

            # If not enough validation data for a full batch, bail
            if self.val_data.num_data < self.batch_size:
                return

            # HACK NOTE Can't compute loss for the whole validation dataset
            # so we take the mean of 100 samplings
            val_losses = []
            for _ in range(100):
                self.val_sampler.sample_data(key=None, k=self.batch_size)
                vl = self.problem.run_loss(sess=sess,
                                           ins=self.val_sampled.all_inputs,
                                           outs=self.val_sampled.all_labels)
                val_losses.append(vl)
            mean_val_loss = np.mean(val_losses)
            print 'Validation loss: %f ' % mean_val_loss

            self.error_plotter.add_line('validation',
                                        self.spin_counter,
                                        mean_val_loss)
            val_embed, val_labels = self.get_embedding(sess=sess,
                                                       on_training_data=False)
            # HACK for 1D, don't need this
            val_labels = np.mean(val_labels, axis=-1)
            self.embed_plotter.set_scatter('val',
                                           x=val_embed[:, 0],
                                           y=val_embed[:, 1],
                                           c=val_labels,
                                           marker='o')
