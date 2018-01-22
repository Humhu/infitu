"""Classes for learning ANN classifiers
"""

import numpy as np
import tensorflow as tf

import adel

import networks as rrn
import utils as rru
from infi_learn.plotting import LineSeriesPlotter, FilterPlotter, ScatterPlotter
from infi_learn.classification import compute_threshold_roc
from infi_learn.data_augmentation import DataAugmenter


class BinaryClassificationNetwork(rrn.VectorImageNetwork):
    """Derived VI network for discrete binary classification
    """
    # TODO Have output be automatically set to dimension 2

    def __init__(self, **kwargs):
        rrn.VectorImageNetwork.__init__(self, **kwargs)

    # def __repr__(self):
    #     s = 'Classifier network:'
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


class BinaryClassificationProblem(object):
    """Wraps a value network and learning optimization. Provides methods for
    using the value network.

    Parameters
    ==========
    model : ValueNetwork
        The model to learn
    """

    def __init__(self, model, learning_rate=1e-3, **kwargs):
        self.model = model

        self.image_ph = model.make_img_input(name='image')
        self.vec_ph = model.make_vec_input(name='vector')
        self.binary_ph = tf.placeholder(tf.int32,
                                        shape=[None, 1],
                                        name='%s/binary' % model.scope)
        self.net, self.params, state, ups = model.make_net(img_in=self.image_ph,
                                                           vec_in=self.vec_ph,
                                                           reuse=False)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.binary_ph,
                                                           logits=self.net[-1])

        opt = tf.train.AdamOptimizer(learning_rate=float(learning_rate))
        with tf.control_dependencies(ups):
            self.train = opt.minimize(self.loss, var_list=self.params)

        self.initializers = [s.initializer for s in state] + \
            [adel.optimizer_initializer(opt, self.params)]

    def initialize(self, sess):
        sess.run(self.initializers)

    def run_training(self, sess, ins, outs):
        feed = self.model.init_feed(training=True)
        self._fill_feed(ins=ins, outs=outs, feed=feed)
        return sess.run([self.loss, self.train], feed_dict=feed)[0]

    def run_loss(self, sess, ins, outs):
        feed = self.model.init_feed(training=False)
        self._fill_feed(ins=ins, outs=outs, feed=feed)
        return sess.run(self.loss, feed_dict=feed)

    def run_logits(self, sess, ins):
        feed = self.model.init_feed(training=False)
        self.model.get_ops(inputs=ins,
                           img_ph=self.image_ph,
                           vec_ph=self.vec_ph,
                           feed=feed)
        return sess.run(self.net[-1], feed_dict=feed)

    def _fill_feed(self, ins, outs, feed):
        self.model.get_ops(inputs=ins,
                           img_ph=self.image_ph,
                           vec_ph=self.vec_ph,
                           feed=feed)
        feed[self.binary_ph] = np.atleast_2d(outs).T

# This is very similar to EmbeddingLearner, consolidate


class BinaryClassificationLearner(object):
    def __init__(self, problem, holdout, augmentation,
                 batch_size=30, iters_per_spin=10, validation_period=10):
        self.problem = problem
        self.augmenter = DataAugmenter(**augmentation)

        self.train_base = adel.BasicDataset()
        self.val_base = adel.BasicDataset()
        self.holdout = adel.HoldoutWrapper(training=self.train_base,
                                           holdout=self.val_base,
                                           **holdout)

        self.train_sampler = adel.DatasetSampler(base=self.train_base,
                                                 method='uniform')
        self.train_sampled = adel.LabeledDatasetTranslator(
            base=self.train_sampler)

        self.train_data = adel.LabeledDatasetTranslator(base=self.train_base)
        self.val_data = adel.LabeledDatasetTranslator(base=self.val_base)
        self.reporter = adel.LabeledDatasetTranslator(base=self.holdout)

        self.iters_per_spin = iters_per_spin
        self.val_period = validation_period
        self.batch_size = batch_size

        self.spin_counter = 0

        self.error_plotter = LineSeriesPlotter(title='Log error over time %s' % self.scope,
                                               xlabel='Spin iter',
                                               ylabel='log loss')
        self.roc_plotter = LineSeriesPlotter(title='ROC for %s' % self.scope,
                                             xlabel='FPR',
                                             ylabel='TPR')
        self.class_plotter = ScatterPlotter(title='Value parities %s' % self.scope,
                                                  xlabel='Classes',
                                                  ylabel='Probability of nominal')
        self.class_inited = False
        self.plottables = [self.error_plotter,
                           self.roc_plotter, self.class_plotter]

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
        return self.plottables

    def get_status(self):
        """Returns a string describing the status of this learner
        """
        status = 'Classification %s has %d/%d (train/val) datapoints' % \
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
            aug_s = self.augmenter.augment_data(self.train_sampled.all_inputs)

            train_loss = self.problem.run_training(sess=sess,
                                                   ints=aug_s,
                                                   outs=self.train_sampled.all_labels)

        self.spin_counter += 1
        print 'Training iter: %d loss: %f ' % \
            (self.spin_counter * self.iters_per_spin, train_loss)
        self.error_plotter.add_line('training',
                                    self.spin_counter,
                                    train_loss)

        if self.spin_counter % self.val_period == 0:

            # Plot filters if using them
            if self.problem.model.using_image:
                fil_vals = np.squeeze(sess.run(self.filters))
                fil_vals = np.rollaxis(fil_vals, -1)
                self.filter_plotter.set_filters(fil_vals)

            val_loss = self.problem.run_loss(sess=sess,
                                             data=self.val_data)
            print 'Validation loss: %f ' % val_loss
            self.error_plotter.add_line('validation',
                                        self.spin_counter,
                                        val_loss)

            logits = self.problem.run_logits(sess=sess, data=self.val_data)
            probs = np.exp(logits)
            probs = probs[:, 1] / np.sum(probs, axis=1)

            self.class_plotter.set_scatter('val',
                                           x=self.val_data.all_labels,
                                           y=probs,
                                           c=probs,
                                           marker='o')
            if not self.class_inited:
                # TODO Change this to a histogram over the logits
                self.class_plotter.ax.set_xticks(
                    [0, 1], ['failure', 'nominal'])
                self.class_inited = True

            tprs, fprs = compute_threshold_roc(probs, self.val_data.all_labels)
            self.roc_plotter.set_line(name='roc',
                                      x=fprs,
                                      y=tprs,
                                      marker='o')
