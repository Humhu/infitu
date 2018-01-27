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
    def __init__(self, dual_net, **kwargs):
        rrn.VectorImageNetwork.__init__(self, **kwargs)
        self.dual_net = dual_net

    def make_net(self, img_in=None, vec_in=None, reuse=False):
        """Creates the model network with the specified parameters
        """
        # TODO Decide if we want this
        # if self.using_image:
        #     pre_layers = self.get_img_preprocess(img_in=img_in)
        #     img_in = pre_layers.pop()
        # else:
        pre_layers = []

        if self.using_image and not self.using_vector:
            if not self.dual_net:
                if self.preprocessing_spec is not None:
                    pre_layers = self.get_img_preprocess(img_in)
                    img_in = pre_layers[-1]
                net, train, state, ups = adel.make_conv2d_fc_net(img_in=img_in,
                                                                 reuse=reuse,
                                                                 **self.network_spec)
            else:
                if self.preprocessing_spec is None:
                    raise ValueError('Must specify preprocessing spec if using dual net')
                pre_layers = self.get_img_preprocess(img_in)
                net, train, state, ups = adel.make_conv2d_parallel_net(img1=img_in,
                                                                       img2=pre_layers[-1],
                                                                       reuse=reuse,
                                                                       **self.network_spec)
        elif not self.using_image and self.using_vector:
            net, train, state, ups = adel.make_fullycon(input=vec_in,
                                                        reuse=reuse,
                                                        **self.network_args)
        else:
            net, train, state, ups = adel.make_conv2d_joint_net(img_in=img_in,
                                                                vector_in=vec_in,
                                                                reuse=reuse,
                                                                **self.network_args)
        return pre_layers + net, train, state, ups


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

    def __init__(self, model, dist_type, temporal_dist_type, loss_type, augmentation, learning,
                 loss_clip=1.0, label_dim=1, label_scale=1.0, ball_size=1.0,
                 attraction_k=1e-3, anchor_weight=1e-6, temporal_weight=0):
        self.model = model

        self.p_image_ph = model.make_img_input(name='p_image')
        self.p_belief_ph = model.make_vec_input(name='p_belief')
        self.net, self.params, state, ups = model.make_net(img_in=self.p_image_ph,
                                                           vec_in=self.p_belief_ph,
                                                           reuse=False)

        self.n_image_ph = model.make_img_input(name='n_image')
        self.n_belief_ph = model.make_vec_input(name='n_belief')
        self.n_net = model.make_net(img_in=self.n_image_ph,
                                    vec_in=self.n_belief_ph,
                                    reuse=True)[0]

        self.p_labels_ph = tf.placeholder(tf.float32,
                                          name='%s/p_labels' % self.model.scope,
                                          shape=[None, label_dim])

        if self.p_image_ph is not None:
            self.pn_image_ph = tf.concat([self.p_image_ph, self.n_image_ph],
                                         axis=3)
        else:
            self.pn_image_ph = None

        if self.p_belief_ph is not None:
            self.pn_belief_ph = tf.concat([self.p_belief_ph, self.n_belief_ph],
                                          axis=2)
        else:
            self.pn_belief_ph = None
        self.pn_augmenter = DataAugmenter(image_ph=self.pn_image_ph,
                                          vector_ph=self.pn_belief_ph,
                                          labels_ph=self.p_labels_ph,
                                          **augmentation)

        # self.p_augmenter = DataAugmenter(image_ph=self.p_image_ph,
        #                                  vector_ph=self.p_belief_ph,
        #                                  labels_ph=self.p_labels_ph,
        #                                  **augmentation)
        # self.n_augmenter = DataAugmenter(image_ph=self.n_image_ph,
        #                                  vector_ph=self.n_belief_ph,
        #                                  labels_ph=self.p_labels_ph,
        #                                  **augmentation)

        # Generate combinations, courtesy of stackoverflow
        X = self.net[-1]
        Xn = self.n_net[-1]

        if dist_type == 'euclidean':
            r = tf.reshape(tf.reduce_sum(X * X, 1), [-1, 1])
            # sum of squares for each vector
            XXT = tf.matmul(X, tf.transpose(X))
            x_pdist = r - 2 * XXT + tf.transpose(r)
            x_pdist = x_pdist / tf.cast(tf.shape(X)[1], tf.float32)

            # Less than L2 norm 1
            x_cond = tf.nn.relu(tf.diag_part(XXT) - ball_size)

        elif dist_type == 'cosine':
            X_norm = tf.nn.l2_normalize(X, dim=-1)
            x_pdist = -0.5 * tf.matmul(X_norm, tf.transpose(X_norm)) + 0.5

            # Between L2 norm 0.1 and 1
            XXT = tf.matmul(X, tf.transpose(X))
            XXTd = tf.diag_part(XXT)
            x_cond = tf.nn.relu(0.1 - XXTd) + tf.nn.relu(XXTd - ball_size)
        else:
            raise ValueError('Unrecognized dist type: %s' % dist_type)

        if temporal_dist_type == 'euclidean':
            dXn = Xn - X
            dXnXnT = tf.matmul(dXn, tf.transpose(dXn))
            x_xn_dist = tf.diag_part(dXnXnT)

        elif temporal_dist_type == 'cosine':
            X_norm = tf.nn.l2_normalize(X, dim=-1)
            Xn_norm = tf.nn.l2_normalize(Xn, dim=-1)
            x_xn_dist = -0.5 * tf.matmul(X_norm, tf.transpose(Xn_norm)) + 0.5
            x_xn_dist = tf.diag_part(x_xn_dist)
        else:
            raise ValueError(
                'Unrecognized temporal distance type: %s' % temporal_dist_type)

        y = self.p_labels_ph * float(label_scale)
        q = tf.reshape(tf.reduce_sum(y * y, 1), [-1, 1])
        y_dist = q - 2 * tf.matmul(y, tf.transpose(y)) + tf.transpose(q)
        y_dist = y_dist / label_dim

        self.state = state
        if loss_type == 'margin':
            zero_point = y_dist
            err = x_pdist - zero_point

            # squared penalty for dissimilar points being too close
            close_loss = tf.nn.relu(-err)
            
            # penalty for similar points being too far
            far_loss = float(attraction_k) * tf.log(tf.nn.relu(err) + 1)
            embedding_loss = tf.reduce_mean(close_loss + far_loss)
        elif loss_type == 'product':
            embedding_loss = tf.reduce_mean(-x_pdist * y_dist)
        else:
            raise ValueError('Unknown loss type: %s' % loss_type)

        # Add temporal smoothness
        time_loss = float(temporal_weight) * tf.reduce_mean(x_xn_dist, axis=-1)

        # Penalize points for being too far from zero
        # TODO Set desired ball size
        anchor_loss = float(anchor_weight) * tf.reduce_mean(x_cond, axis=-1)

        self.loss = tf.clip_by_norm(embedding_loss + anchor_loss + time_loss,
                                    clip_norm=float(loss_clip))

        self.learner = Learner(variables=self.params, updates=ups,
                               loss=self.loss, **learning)
        self.initializers = [s.initializer for s in state] + self.learner.init

    def initialize(self, sess):
        sess.run(self.initializers)

    def run_training(self, sess, ins, outs):
        feed = self.model.init_feed(training=True)
        self._fill_feed(ins=ins, outs=outs, feed=feed)

        # p_ins, outs = self.p_augmenter.augment_data(sess=sess, feed=feed)
        # n_ins, outs = self.n_augmenter.augment_data(sess=sess, feed=feed)

        ins, outs = self.pn_augmenter.augment_data(sess=sess, feed=feed)
        # TODO Make this more efficient somehow?
        if self.model.using_image and self.model.using_vector:
            vec, img = zip(*ins)
            vdim = vec.shape[1] / 2
            p_vec = vec[:, :vdim]
            n_vec = vec[:, vdim:]
            p_img = img[:, :, :, 0]
            n_img = img[:, :, :, 1]
            p_ins = zip(p_vec, p_img)
            n_ins = zip(n_vec, n_img)
        elif self.model.using_image and not self.model.using_vector:
            p_ins = ins[:, :, :, 0]
            n_ins = ins[:, :, :, 1]
        elif not self.model.using_image and self.model.using_vector:
            vdim = ins.shape[1] / 2
            p_ins = ins[:, :vdim]
            n_ins = ins[:, vdim:]
        else:
            raise ValueError('Must use image or vector')
        ins = zip(p_ins, n_ins)

        # ins, outs = self.p_augmenter.augment_data(sess=sess, feed=feed)

        self._fill_feed(ins=ins, outs=outs, feed=feed)
        return self.learner.step(sess=sess, feed=feed)

    def run_loss(self, sess, ins, outs):
        feed = self.model.init_feed(training=False)
        self._fill_feed(ins=ins, outs=outs, feed=feed)
        return sess.run(self.loss, feed_dict=feed)

    def run_embedding(self, sess, ins):
        feed = self.model.init_feed(training=False)

        # TODO Hardcoded block size
        block_size = 1000
        blocks = []
        for i in range((len(ins) / block_size) + 1):
            start = i * block_size
            stop = min(len(ins), (i + 1) * block_size)
            self.model.get_ops(inputs=ins[start:stop],
                               img_ph=self.p_image_ph,
                               vec_ph=self.p_belief_ph,
                               feed=feed)
            blocks.append(sess.run(self.net[-1], feed_dict=feed))
        return np.vstack(blocks)

    def _fill_feed(self, ins, outs, feed):
        """Helper to create class permutations and populate
        a feed dict.
        """
        c, n = zip(*ins)
        self.model.get_ops(inputs=c,
                           img_ph=self.p_image_ph,
                           vec_ph=self.p_belief_ph,
                           feed=feed)
        self.model.get_ops(inputs=n,
                           img_ph=self.n_image_ph,
                           vec_ph=self.n_belief_ph,
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
                                               ylabel='Embedding loss',
                                               log_y=True)
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
        x = [self.problem.run_embedding(sess=sess,
                                        ins=zip(*adel.LabeledDatasetTranslator(chunk).all_inputs)[0])
             # ins=adel.LabeledDatasetTranslator(chunk).all_inputs)
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
            s = min(self.val_data.num_data, self.batch_size)

            # HACK NOTE Can't compute loss for the whole validation dataset
            # so we take the mean of 100 samplings
            val_losses = []
            for _ in range(100):
                self.val_sampler.sample_data(key=None, k=s)
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
