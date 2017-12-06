#! /usr/bin/env python

import abc
import adel
import infi_learn as rr
import rospy
import tensorflow as tf
import numpy as np
import dill


class NetworkInterface(object):
    """Convenience class for training tensorflow networks. Simplifies
    enabling/disabling batch normalization and dropout, as well as
    initializing feed dicts for tensorflow.
    """

    def __init__(self):
        self.network_spec = rospy.get_param('~network')
        self.batch_training = None
        if rospy.get_param('~learning/batch_normalization', False):
            rospy.loginfo('Using batch normalization')
            self.batch_training = tf.placeholder(
                tf.bool, name='batch_training')
            self.network_spec['batch_training'] = self.batch_training

        self.training_dropout = rospy.get_param('~learning/dropout_rate', 0.0)
        self.dropout_rate = None
        if self.training_dropout > 0.0:
            rospy.loginfo('Using dropout rate of %f', self.training_dropout)
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            self.network_spec['dropout_rate'] = self.dropout_rate

    @property
    def network_args(self):
        """Returns a kwarg dict to construct a neural network. Note that a copy
        of the internal field is returned, so it may be modified.
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


class BaseEmbeddingLearner(object):
    """Base class for all action-split embedding learner nodes
    """

    def __init__(self, frontend_class):
        self.trainer = NetworkInterface()

        self.backend = rr.ActionSplitBackend()
        self.frontend = frontend_class(self.backend)

        self.sep_dist = rospy.get_param('~learning/separation_distance')
        self.batch_size = rospy.get_param('~learning/batch_size')
        self.iters_per_spin = rospy.get_param('~learning/iters_per_spin')

        spin_rate = rospy.get_param('~spin_rate')
        spin_dt = 1.0 / spin_rate
        self.spin_timer = rospy.Timer(rospy.Duration(spin_dt),
                                      callback=self.spin)
        self.spin_iter = 0

        self.validation_period = rospy.get_param(
            '~learning/validation_period', 0)
        self.dummy_loss = tf.constant('n/a')

        self.embeddings = {}
        self.sess = tf.Session()

        # Plotting and visualization
        # TODO One plotter per embedding
        self.error_plotter = rr.LineSeriesPlotter(title='Embedding losses over time',
                                                  xlabel='Spin iter (%f s/spin)' % spin_dt,
                                                  ylabel='Embedding loss')
        self.embed_plotter = rr.LineSeriesPlotter(title='Validation embeddings',
                                                  xlabel='Embedding dimension 1',
                                                  ylabel='Embedding dimension 2')

        self.plot_group = rr.PlottingGroup()
        self.plot_group.add_plottable(self.error_plotter)
        self.plot_group.add_plottable(self.embed_plotter)
        for p in self.frontend.get_plottables():
            self.plot_group.add_plottable(p)
        self.plot_rate = rospy.get_param('~plot_rate', 10.0)

    def spin_plot(self):
        self.plot_group.spin(self.plot_rate)

    @abc.abstractmethod
    def create_model(self, scope):
        pass

    def spin(self, event):
        self.frontend.spin(event.current_real.to_sec())

        # See if we have any new actions to initialize embeddings for
        for a in self.backend.get_splits():
            if a in self.embeddings:
                continue

            rospy.loginfo('Creating new learner for action: %s' % str(a))
            with self.sess.graph.as_default():
                model = self.create_model(scope='embed_%d/' % len(self.embeddings))
                learner = rr.EmbeddingTask(model=model,
                                           train_data=self.backend.get_training(a),
                                           sep_dist=self.sep_dist)
                rospy.loginfo('Created embedding: %s', str(model))

                self.sess.run(learner.initializers)
            # Store model index, model, and learner
            self.embeddings[a] = (len(self.embeddings), model, learner)

        if len(self.embeddings) == 0:
            # rospy.loginfo('No embeddings, skipping')
            return

        self._spin_training()
        if self.validation_period > 0 and (self.spin_iter % self.validation_period) == 0:
            self._spin_validation()

        self.spin_iter += 1

    def _spin_validation(self):
        """Runs validation
        """
        feed = self.trainer.init_feed(training=False)
        ops = []
        labels = []
        for i, item in enumerate(self.embeddings.iteritems()):
            a, l = item
            ind, model, learner = l
            val = self.backend.get_validation(a)
            # NOTE We're doing double work to compute the loss and the embedding
            # separately here, but it keeps the interfaces simpler
            loss_ops = learner.get_feed_dataset(feed=feed, dataset=val)
            emb_ops, labs = rr.get_embed_validation(feed=feed,
                                                    model=model,
                                                    dataset=val)
            if len(loss_ops) == 0:
                loss_ops = [self.dummy_loss]
            ops += loss_ops + emb_ops
            labels.append(labs)

        res = self.sess.run(ops, feed_dict=feed)
        losses = res[0::2]
        embeds = res[1::2]

        for i, item in enumerate(self.embeddings.iteritems()):
            a, l = item
            ind, model, learner = l
            val = self.backend.get_validation(a)
            rospy.loginfo('Validation: action %d has (steps/terms) (%d/%d) and loss %s',
                          ind, val.num_tuples, val.num_terminals, str(losses[i]))

            if isinstance(losses[i], float):
                self.error_plotter.add_line('val_%d' % ind, self.spin_iter, losses[i])

            pos_embeds = embeds[i][labels[i]]
            neg_embeds = embeds[i][np.logical_not(labels[i])]
            self.embed_plotter.set_line('val+_%d' % ind, pos_embeds[:, 0], pos_embeds[:, 1],
                                        marker='o', linestyle='none')
            self.embed_plotter.set_line('val-_%d' % ind, neg_embeds[:, 0], neg_embeds[:, 1],
                                        marker='x', linestyle='none')

    def _spin_training(self):
        """Runs training and prints stats out after iterations
        """
        for _ in range(self.iters_per_spin):
            feed = self.trainer.init_feed(training=True)
            ops = []
            for i, item in enumerate(self.embeddings.iteritems()):
                a, l = item
                ind, model, learner = l
                i_ops = learner.get_feed_training(feed=feed, k=self.batch_size)
                if len(i_ops) == 0:
                    i_ops = [self.dummy_loss, self.dummy_loss]
                ops += i_ops

            res = self.sess.run(ops, feed_dict=feed)

        losses = res[::2]
        for i, item in enumerate(self.embeddings.iteritems()):
            a, l = item
            ind, model, learner = l
            dataset = self.backend.get_training(a)
            rospy.loginfo('Training: action %d has (steps/terms) (%d/%d) and loss %s',
                          ind, dataset.num_tuples, dataset.num_terminals, str(losses[i]))
            if isinstance(losses[i], float):
                self.error_plotter.add_line('train_%d' % ind, self.spin_iter, losses[i])
