"""Classes for learning state-value functions
"""

import numpy as np
import tensorflow as tf

import adel

import networks as rrn
import utils as rru
from infi_learn.plotting import LineSeriesPlotter, FilterPlotter, ScatterPlotter


class BanditValueNetwork(rrn.VectorImageNetwork):
    """Derived VI network for learning values
    """
    # TODO Have output be automatically set to dimension 1

    def __init__(self, **kwargs):
        rrn.VectorImageNetwork.__init__(self, **kwargs)

    def __repr__(self):
        s = 'State-value network:'
        for e in self.net:
            s += '\n\t%s' % str(e)
        return s

    @property
    def output_op(self):
        return self.net[-1]

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


class BanditValueProblem(object):
    """Wraps a value network and learning optimization. Provides methods for
    using the value network.

    Parameters
    ==========
    model : ValueNetwork
        The model to learn
    """

    def __init__(self, model, loss_type, learning_rate=1e-3, **kwargs):
        self.model = model

        self.image_ph = model.make_img_input(name='p_image')
        self.vec_ph = model.make_vec_input(name='p_vector')
        self.value_ph = tf.placeholder(tf.float32,
                                       shape=[None, 1],
                                       name='%s/values' % model.scope)
        self.net, params, state, ups = model.make_net(img_in=self.image_ph,
                                                      vec_in=self.vec_ph,
                                                      reuse=True)
        if loss_type == 'squared_error':
            self.loss = tf.losses.mean_squared_error(labels=self.value_ph,
                                                     predictions=self.net[-1])
        elif loss_type == 'absolute_error':
            self.loss = tf.losses.absolute_difference(labels=self.value_ph,
                                                      predictions=self.net[-1])
        else:
            raise ValueError('Unknown loss type: %s' % loss_type)

        opt = tf.train.AdamOptimizer(learning_rate=float(learning_rate))
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

    def run_value(self, sess, data):
        feed = self.model.init_feed(training=False)
        ops = self.model.get_ops(states=data.all_inputs,
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

# This is very similar to EmbeddingLearner, consolidate


class BanditValueLearner(object):
    def __init__(self, problem, holdout,
                 batch_size=30, iters_per_spin=10, validation_period=10):
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

        self.train_data = adel.LabeledDatasetTranslator(base=self.train_base)
        self.val_data = adel.LabeledDatasetTranslator(base=self.val_base)
        self.reporter = adel.LabeledDatasetTranslator(base=self.holdout)

        self.iters_per_spin = iters_per_spin
        self.val_period = validation_period
        self.batch_size = batch_size

        self.spin_counter = 0

        self.error_plotter = LineSeriesPlotter(title='Value error over time %s' % self.scope,
                                               xlabel='Spin iter',
                                               ylabel='Mean squared loss')
        self.value_plotter = ScatterPlotter(title='Value parities %s' % self.scope,
                                                  xlabel='Target value',
                                                  ylabel='Estimated value')
        self.plottables = [self.error_plotter, self.value_plotter]

        if self.problem.model.using_image:
            # TODO HACK
            self.filters = self.problem.model.params[0]
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

    def get_values(self, sess, on_training_data=True):
        if on_training_data:
            data = self.train_data
        else:
            data = self.val_data
        return self.problem.run_output(sess=sess, data=data)

    def get_status(self):
        """Returns a string describing the status of this learner
        """
        status = 'Value %s has %d/%d (train/val) datapoints' % \
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
                                                   data=self.train_sampled)

        self.spin_counter += 1
        print 'Training iter: %d loss: %f ' % \
            (self.spin_counter * self.iters_per_spin, train_loss)
        self.error_plotter.add_line('training', self.spin_counter, train_loss)

        if self.spin_counter % self.val_period == 0:

            # Plot filters if using them
            if self.problem.model.using_image:
                fil_vals = np.squeeze(sess.run(self.filters))
                fil_vals = np.rollaxis(fil_vals, -1)
                self.filter_plotter.set_filters(fil_vals)

            val_loss = self.problem.run_loss(sess=sess,
                                             data=self.val_data)
            print 'Validation loss: %f ' % val_loss
            self.error_plotter.add_line(
                'validation', self.spin_counter, val_loss)

            values = self.problem.run_value(sess=sess, data=self.val_data)
            self.value_plotter.set_scatter('val',
                                           x=self.val_data.all_labels,
                                           y=values,
                                           c=self.val_data.all_labels,
                                           marker='o')
