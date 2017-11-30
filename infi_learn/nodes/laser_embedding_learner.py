#! /usr/bin/env python

import adel
import infi_learn as rr
import rospy
import tensorflow as tf
import itertools
import numpy as np
import matplotlib.pyplot as plt

# TODO Generalize to use different frontends?


class LaserEmbeddingLearner(object):
    """Combines a laser frontend with an action-split backend to learn an embedding
    for each action-split dataset.
    """

    def __init__(self):
        self.backend = rr.ActionSplitBackend()

        belief_dim = rospy.get_param('~frontend/belief_dim')
        self.action_dim = rospy.get_param('~frontend/action_dim')
        laser_dim = rospy.get_param('~frontend/laser_dim')
        laser_fov = rospy.get_param('~frontend/laser_fov')
        max_range = rospy.get_param('~frontend/laser_max_range')
        resolution = rospy.get_param('~frontend/laser_paint_resolution')
        dt_tol = rospy.get_param('~frontend/time_tolerance')

        self.belief_source = rr.VectorSource(dim=belief_dim, topic='belief')
        self.laser_source = rr.LaserSource(laser_dim=laser_dim, topic='scan',
                                           enable_painting=True, fov=laser_fov,
                                           max_range=max_range, resolution=resolution,
                                           enable_vis=True)
        self.state_source = rr.MultiDataSource([self.belief_source, self.laser_source],
                                               tol=dt_tol)

        self.frontend = rr.DataSourceFrontend(source=self.state_source,
                                              backend=self.backend)

        self.sep_dist = rospy.get_param('~learning/separation_distance')
        self.batch_size = rospy.get_param('~learning/batch_size')
        self.iters_per_spin = rospy.get_param('~learning/iters_per_spin')

        spin_rate = rospy.get_param('~spin_rate')
        self.spin_timer = rospy.Timer(rospy.Duration(1.0 / spin_rate),
                                      callback=self.spin)
        self.spin_iter = 0

        self.validation_period = rospy.get_param(
            '~learning/validation_period', 0)

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

        self.embeddings = {}
        self.sess = tf.Session()

        self.error_plotter = rr.LineSeriesPlotter()
        self.embed_plotter = rr.LineSeriesPlotter()

        self.plot_group = rr.PlottingGroup()
        self.plot_group.add_plottable(self.error_plotter)
        self.plot_group.add_plottable(self.embed_plotter)
        self.plot_group.add_plottable(self.laser_source)

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

    def spin(self, event):
        self.frontend.spin(event.current_real.to_sec())
        for action, dataset in self.backend.datasets.iteritems():
            if action not in self.embeddings:
                rospy.loginfo(
                    'Creating new learner for action: %s' % str(action))
                with self.sess.graph.as_default():
                    learner = rr.EmbeddingLearner(dataset=dataset,
                                                  validation=self.backend.validations[action],
                                                  img_size=self.laser_source.painter.img_size,
                                                  vec_size=self.belief_source.dim,
                                                  sep_dist=self.sep_dist,
                                                  scope='embed_%d/' %
                                                  len(self.embeddings),
                                                  spec=self.network_spec)
                    rospy.loginfo('Created embedding: %s', str(learner))
                    self.sess.run(learner.initializers)
                self.embeddings[action] = (len(self.embeddings), learner)

        if len(self.embeddings) == 0:
            rospy.loginfo('No embeddings, skipping')
            return

        self._spin_training()
        if self.validation_period > 0 and (self.spin_iter % self.validation_period) == 0:
            self._spin_validation()

        self.spin_iter += 1

    def _spin_validation(self):
        """Runs validation
        """
        feed = self.init_feed(training=False)
        ops = []
        valid_inds = []
        for i, item in enumerate(self.embeddings.iteritems()):
            k, l = item
            ind, net = l
            i_ops = net.get_feed_validation(feed)
            if len(i_ops) > 0:
                valid_inds.append(i)
            ops += i_ops

        if len(ops) == 0:
            rospy.loginfo('No learners ready, skipping validation')
            return

        losses = [None] * len(self.embeddings)
        res = self.sess.run(ops, feed_dict=feed)
        for i, vi in enumerate(valid_inds):
            losses[vi] = res[i]

        for i, item in enumerate(self.embeddings.iteritems()):
            k, l = item
            ind, net = l
            validation = self.backend.validations[k]
            rospy.loginfo('Validation: action %d has (steps/terms) (%d/%d) and loss %s',
                          ind, validation.num_tuples, validation.num_terminals, str(losses[i]))

            self.error_plotter.add_line(
                'val_%d' % ind, self.spin_iter, losses[i])

            feed = self.init_feed(training=False)
            op, labels = net.get_embed_validation(feed)
            embed = self.sess.run(op, feed_dict=feed)
            pos_embeds = embed[labels]
            neg_embeds = embed[np.logical_not(labels)]
            self.embed_plotter.set_line('val+_%d' % ind, pos_embeds[:, 0], pos_embeds[:, 1],
                                        marker='o', linestyle='none')
            self.embed_plotter.set_line('val-_%d' % ind, neg_embeds[:, 0], neg_embeds[:, 1],
                                        marker='x', linestyle='none')

    def _spin_training(self):
        """Runs training
        """
        for _ in range(self.iters_per_spin):
            feed = self.init_feed(training=True)
            ops = []
            valid_inds = []
            for i, item in enumerate(self.embeddings.iteritems()):
                k, l = item
                ind, net = l
                i_ops = net.get_feed_training(feed, self.batch_size)
                if len(i_ops) > 0:
                    valid_inds.append(i)
                ops += i_ops

            if len(ops) == 0:
                rospy.loginfo('No learners ready, skipping training')
                return

            losses = [None] * len(self.embeddings)
            res = self.sess.run(ops, feed_dict=feed)
            res_losses = res[0::2]
            for i, vi in enumerate(valid_inds):
                losses[vi] = res_losses[i]

        for i, item in enumerate(self.embeddings.iteritems()):
            k, l = item
            ind, net = l
            dataset = self.backend.datasets[k]
            rospy.loginfo('Training: action %d has (steps/terms) (%d/%d) and loss %s',
                          ind, dataset.num_tuples, dataset.num_terminals, str(losses[i]))
            print losses[i]
            self.error_plotter.add_line(
                'train_%d' % ind, self.spin_iter, losses[i])


if __name__ == '__main__':
    rospy.init_node('laser_embedding_learner')
    lel = LaserEmbeddingLearner()
    try:
        lel.plot_group.spin(10.0)
    except rospy.ROSInterruptException:
        pass
