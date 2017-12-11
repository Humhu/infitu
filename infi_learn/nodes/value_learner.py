#! /usr/bin/env python

import adel
import rospy
import tensorflow as tf

import infi_learn as rr
from infi_msgs.msg import EpisodeBreak
import matplotlib.pyplot as plt


class DataSources(object):
    def __init__(self, image=None, vector=None, dt_tol=0.1):
        self.use_image = image is not None
        if self.use_image:
            self.image_source = rr.ImageSource(**image)

        self.use_vec = vector is not None
        if self.use_vec:
            self.vec_source = rr.VectorSource(**vector)

        if self.use_image and self.use_vec:
            # By convention (vecs, imgs)
            self.state_source = rr.MultiDataSource([self.vec_source, self.image_source],
                                                   tol=dt_tol)
        elif self.use_image and not self.use_vec:
            self.state_source = self.image_source
        elif not self.use_image and self.use_vec:
            self.state_source = self.vec_source
        else:
            raise ValueError('Must use image and/or vector')

    @property
    def img_size(self):
        if self.use_image:
            return self.image_source.dim
        else:
            return None

    @property
    def vec_size(self):
        if self.use_vec:
            return self.vec_source.dim
        else:
            return None


class ValueLearner(object):
    def __init__(self, model, holdout, batch_size=30, iters_per_spin=10,
                 validation_period=10):
        self.problem = rr.BanditValueProblem(model=model)

        self.tr_base = adel.BasicDataset()
        self.val_base = adel.BasicDataset()
        self.holdout = adel.HoldoutWrapper(training=self.tr_base,
                                           holdout=self.val_base,
                                           **holdout)

        self.tr_sampler = adel.DatasetSampler(base=self.tr_base,
                                              method='uniform')
        self.tr_sampled = adel.LabeledDatasetTranslator(base=self.tr_sampler)
        self.tr_data = adel.LabeledDatasetTranslator(base=self.tr_base)
        self.val_data = adel.LabeledDatasetTranslator(base=self.val_base)
        self.reporter = adel.LabeledDatasetTranslator(base=self.holdout)

        self.iters_per_spin = iters_per_spin
        self.val_period = validation_period
        self.batch_size = batch_size

        self.spin_counter = 0

        self.error_plotter = rr.LineSeriesPlotter(title='Value error over time %s' % model.scope,
                                                  xlabel='Spin iter',
                                                  ylabel='Mean squared loss')
        self.value_plotter = rr.LineSeriesPlotter(title='Values %s' % model.scope,
                                                  xlabel='Spin iter',
                                                  ylabel='Value')

    def initialize(self, sess):
        self.problem.initialize(sess)

    @property
    def scope(self):
        return self.problem.model.scope

    def report_state_value(self, state, value):
        self.reporter.report_label(x=state, y=value)

    def get_plottables(self):
        return [self.error_plotter, self.value_plotter]

    def get_values(self, sess, on_training_data=True):
        if on_training_data:
            data = self.tr_data
        else:
            data = self.val_data
        return self.problem.run_output(sess=sess, data=data)

    def spin(self, sess):
        rospy.loginfo('Value %s has %d/%d (train/val) datapoints',
                      self.scope,
                      self.tr_data.num_data,
                      self.val_data.num_data)

        if self.tr_data.num_data < self.batch_size:
            rospy.loginfo('Num data %d less than batch size %d',
                          self.tr_data.num_data, self.batch_size)
            return

        for _ in range(self.iters_per_spin):
            self.tr_sampler.sample_data(key=None, k=self.batch_size)
            train_loss = self.problem.run_training(sess=sess,
                                                   data=self.tr_sampled)

        self.spin_counter += 1
        rospy.loginfo('Training iter: %d loss: %f ' %
                      (self.spin_counter * self.iters_per_spin, train_loss))
        self.error_plotter.add_line('training', self.spin_counter, train_loss)

        if self.spin_counter % self.val_period == 0:
            val_loss = self.problem.run_loss(sess=sess,
                                             data=self.val_data)
            rospy.loginfo('Validation loss: %f ' % val_loss)
            self.error_plotter.add_line(
                'validation', self.spin_counter, val_loss)

            values = self.problem.run_output(sess=sess, data=self.val_data)
            self.value_plotter.set_line('predicted', x=range(len(values)),
                                        y=values, color='r')
            self.value_plotter.set_line('true', x=range(len(values)),
                                        y=self.val_data.all_labels, color='b')

            rospy.loginfo(self.problem.model.print_filters(sess))


class ValueLearnerNode(object):
    def __init__(self):
        self.sources = DataSources(**rospy.get_param('~sources'))
        self.frontend = rr.SARSFrontend(source=self.sources.state_source,
                                        break_mode='action_change',
                                        **rospy.get_param('~sars_frontend'))
        self.frontend.register_callback(self.data_callback)

        self.network_args = rospy.get_param('~value/network')
        self.learner_args = rospy.get_param('~value/learning')
        self.registry = {}

        self.sess = tf.Session()

        self.plot_group = rr.PlottingGroup()

        spin_rate = rospy.get_param('~value_spin_rate')
        spin_dt = 1.0 / spin_rate
        self.embedding_timer = rospy.Timer(rospy.Duration(spin_dt),
                                           callback=self.spin_embedding)

    def initialize_new(self, scope):
        with self.sess.graph.as_default():
            model = rr.ValueNetwork(img_size=self.sources.img_size,
                                    vec_size=self.sources.vec_size,
                                    scope=scope,
                                    **self.network_args)
            learner = ValueLearner(model=model,
                                   **self.learner_args)

            # Initialize tensorflow variables
            model.initialize(self.sess)
            learner.initialize(self.sess)
            print 'Created value network:\n%s' % str(model)

        for p in learner.get_plottables():
            self.plot_group.add_plottable(p)

        return model, learner

    def data_callback(self, is_active, savs):
        if not is_active:
            return

        s, a, v, sn = savs
        if a not in self.registry:
            scope = 'value_%d' % len(self.registry)
            self.registry[a] = self.initialize_new(scope=scope)

        _, learner = self.registry[a]
        learner.report_state_value(state=s, value=v)

    def spin_embedding(self, event):
        now = event.current_real.to_sec()
        self.frontend.spin(now)
        for val in self.registry.itervalues():
            model, learner = val
            learner.spin(sess=self.sess)


if __name__ == '__main__':
    rospy.init_node('value_learner')

    vln = ValueLearnerNode()

    plot_rate = rospy.get_param('~plot_rate', 10.0)
    try:
        vln.plot_group.spin(plot_rate)
    except rospy.ROSInterruptException:
        pass
