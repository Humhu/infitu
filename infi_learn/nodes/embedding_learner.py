#! /usr/bin/env python

import adel
import optim
import infi_learn as rr
import rospy
import tensorflow as tf
import numpy as np
import dill
from itertools import izip
import matplotlib.pyplot as plt


class DataSources(object):
    def __init__(self, image=None, belief=None, dt_tol=0.1):
        self.use_image = image is not None
        if self.use_image:
            self.image_source = rr.ImageSource(**image)

        self.use_belief = belief is not None
        if self.use_belief:
            self.belief_source = rr.VectorSource(**belief)

        if self.use_image and self.use_belief:
            # By convention (vecs, imgs)
            self.state_source = rr.MultiDataSource([self.belief_source, self.image_source],
                                                   tol=dt_tol)
        elif self.use_image and not self.use_belief:
            self.state_source = self.image_source
        elif not self.use_image and self.use_belief:
            self.state_source = self.belief_source
        else:
            raise ValueError('Must use image and/or belief')

    @property
    def img_size(self):
        if self.use_image:
            return self.image_source.dim
        else:
            return None

    @property
    def belief_size(self):
        if self.use_belief:
            return self.belief_source.dim
        else:
            return None


class EmbeddingLearner(object):
    def __init__(self, model, holdout, loss,
                 batch_size=10, iters_per_spin=10, validation_period=10):
        self.problem = rr.EmbeddingProblem(model=model, **loss)

        self.tr_base = adel.BasicDataset()
        self.val_base = adel.BasicDataset()
        self.holdout = adel.HoldoutWrapper(training=self.tr_base,
                                           holdout=self.val_base,
                                           **holdout)

        self.tr_sampler = adel.DatasetSampler(base=self.tr_base,
                                              method='uniform')
        self.tr_sampled = adel.LabeledDatasetTranslator(base=self.tr_sampler)
        self.val_sampler = adel.DatasetSampler(base=self.val_base,
                                               method='uniform')
        self.val_sampled = adel.LabeledDatasetTranslator(base=self.val_sampler)

        self.tr_data = adel.LabeledDatasetTranslator(base=self.tr_base)
        self.val_data = adel.LabeledDatasetTranslator(base=self.val_base)
        self.reporter = adel.LabeledDatasetTranslator(base=self.holdout)

        self.iters_per_spin = iters_per_spin
        self.validation_period = validation_period
        self.batch_size = batch_size

        self.spin_counter = 0

        self.error_plotter = rr.LineSeriesPlotter(title='Embedding losses over time %s' % model.scope,
                                                  xlabel='Spin iter',
                                                  ylabel='Embedding loss')
        self.embed_plotter = rr.ScatterPlotter(
            title='Validation embeddings %s' % model.scope)
        
        self.filters = model.params[0]
        n_filters = int(self.filters.shape[-1])
        self.filter_plotter = rr.FilterPlotter(n_filters)

    def initialize(self, sess):
        self.problem.initialize(sess)

    @property
    def scope(self):
        return self.problem.model.scope

    def report_state_value(self, state, value):
        self.reporter.report_label(x=state, y=value)

    def get_plottables(self):
        return [self.error_plotter, self.embed_plotter, self.filter_plotter]

    def get_embedding(self, sess, on_training_data=True):
        if on_training_data:
            data = self.tr_base
        else:
            data = self.val_base

        chunker = adel.DatasetChunker(base=data, block_size=500)  # TODO
        x = [self.problem.run_embedding(sess=sess, data=adel.LabeledDatasetTranslator(chunk))
             for chunk in chunker.iter_subdata(key=None)]  # HACK key
        if len(x) == 0:
            x = []
        else:
            x = np.vstack(x)

        y = adel.LabeledDatasetTranslator(data).all_labels
        return x, y

    def spin(self, sess):
        rospy.loginfo('Embedding %s has %d/%d (train/validation) datapoints',
                      self.scope, self.tr_data.num_data, self.val_data.num_data)

        train_loss = None
        for _ in range(self.iters_per_spin):
            if self.tr_data.num_data < self.batch_size:
                rospy.loginfo('Not enough data to train, skipping...')
                break

            self.tr_sampler.sample_data(key=None, k=self.batch_size)
            train_loss = self.problem.run_training(
                sess=sess, data=self.tr_sampled)

        self.spin_counter += 1
        if train_loss is None:
            return
        rospy.loginfo('Training iter: %d loss: %f ' %
                      (self.spin_counter * self.iters_per_spin, train_loss))
        self.error_plotter.add_line('training', self.spin_counter, train_loss)

        if self.spin_counter % self.validation_period == 0:

            fil_vals = np.squeeze(sess.run(self.filters))
            fil_vals = np.rollaxis(fil_vals, -1)
            self.filter_plotter.set_filters(fil_vals)

            if self.val_data.num_data < self.batch_size:
                return
            val_losses = []
            # HACK NOTE Can't compute loss for the whole validation dataset
            for _ in range(100):
                self.val_sampler.sample_data(key=None, k=self.batch_size)
                vl = self.problem.run_loss(sess=sess,
                                           data=self.val_sampled)
                val_losses.append(vl)
            mean_val_loss = np.mean(val_losses)
            rospy.loginfo('Validation loss: %f ' % mean_val_loss)
            self.error_plotter.add_line(
                'validation', self.spin_counter, mean_val_loss)
            val_embed, val_labels = self.get_embedding(sess=sess,
                                                       on_training_data=False)
            self.embed_plotter.set_scatter('val',
                                           x=val_embed[:, 0],
                                           y=val_embed[:, 1],
                                           c=val_labels,
                                           marker='o')


class ClassifierLearner(object):
    def __init__(self, embedding, classifier, holdout, optimizer, min_value,
                 visualize=True, vis_res=10):
        self.classifier = rr.ParzenNeighborsClassifier(**classifier)
        self.embedding = embedding
        self.optimizer = optim.parse_optimizers(**optimizer)

        self.training_base = adel.BasicDataset()
        self.tuning_base = adel.BasicDataset()
        self.validation_base = adel.BasicDataset()

        self.holdout = adel.HoldoutWrapper(training=self.training_base,
                                           holdout=self.tuning_base,
                                           **holdout)

        self.report_binary = adel.BinaryDatasetTranslator(self.holdout)
        self.training_binary = adel.BinaryDatasetTranslator(self.training_base)
        self.tuning_binary = adel.BinaryDatasetTranslator(self.tuning_base)
        self.validation_binary = adel.BinaryDatasetTranslator(
            self.validation_base)

        self.min_value = min_value

        self.visualize = visualize
        self.vis_res = vis_res

        if self.visualize:
            # TODO Put scope in name
            self.heat_plotter = rr.ImagePlotter(
                vmin=0, vmax=1, title='Classifier')
            self.point_plotter = rr.LineSeriesPlotter(other=self.heat_plotter)

    def get_plottables(self):
        if self.visualize:
            return [self.heat_plotter, self.point_plotter]
        else:
            return []

    def update_dataset(self, sess):
        self.holdout.clear()
        self.validation_base.clear()
        embed, labels = self.embedding.get_embedding(sess=sess,
                                                     on_training_data=True)
        for x, y in izip(embed, labels):
            if y > self.min_value:
                self.report_binary.report_positive(x)
            else:
                self.report_binary.report_negative(x)

        embed, labels = self.embedding.get_embedding(sess=sess,
                                                     on_training_data=False)
        for x, y in izip(embed, labels):
            if y > self.min_value:
                self.validation_binary.report_positive(x)
            else:
                self.validation_binary.report_negative(x)

        rospy.loginfo('Classifier %s has %d/%d (pos/neg) training, %d/%d tuning, %d/%d validation',
                      self.embedding.scope,
                      self.training_binary.num_positives, self.training_binary.num_negatives,
                      self.tuning_binary.num_positives, self.tuning_binary.num_negatives,
                      self.validation_binary.num_positives, self.validation_binary.num_negatives)

    def update_classifier(self):
        X = self.training_binary.all_positives + self.training_binary.all_negatives
        labels = [True] * self.training_binary.num_positives \
            + [False] * self.training_binary.num_negatives
        self.classifier.update_model(X=X, labels=labels)

    def visualize_classifier(self, data):
        if not self.visualize:
            return

        pos, neg = data.all_data
        all_data = np.array(pos + neg)
        pos = np.array(pos)
        neg = np.array(neg)

        mins = np.min(all_data, axis=0)
        maxs = np.max(all_data, axis=0)
        vis_lims = [np.linspace(start=l, stop=u, num=self.vis_res)
                    for l, u in zip(mins, maxs)]
        vis_pts = np.meshgrid(*vis_lims)
        vis_x = np.vstack([vi.flatten() for vi in vis_pts]).T

        probs = self.classifier.query(vis_x)
        pimg = np.reshape(probs, (self.vis_res, self.vis_res))

        self.heat_plotter.set_image(img=pimg,
                                    extents=(mins[0], maxs[0], mins[1], maxs[1]))
        if len(pos) > 0:
            self.point_plotter.set_line(name='pos', x=pos[:, 0], y=pos[:, 1],
                                        color='k', marker='o', markerfacecolor='none',
                                        linestyle='none')
        if len(neg) > 0:
            self.point_plotter.set_line(name='neg', x=neg[:, 0], y=neg[:, 1],
                                        color='k', marker='x', linestyle='none')

    def optimize(self):
        rr.optimize_parzen_neighbors(classifier=self.classifier,
                                     dataset=self.tuning_binary,
                                     optimizer=self.optimizer)
        rospy.loginfo('Classifier %s params: %s', self.embedding.scope,
                      self.classifier.print_params())

        tr_x = self.tuning_binary.all_positives + self.tuning_binary.all_negatives
        tr_y = [True] * self.tuning_binary.num_positives + \
            [False] * self.tuning_binary.num_negatives
        tr_loss = rr.compute_classification_loss(classifier=self.classifier,
                                                 x=tr_x, y=tr_y)

        val_x = self.validation_binary.all_positives + \
            self.validation_binary.all_negatives
        val_y = [True] * self.validation_binary.num_positives \
            + [False] * self.validation_binary.num_negatives
        val_loss = rr.compute_classification_loss(classifier=self.classifier,
                                                  x=val_x, y=val_y)
        rospy.loginfo('Classifier %s training loss: %f validation loss: %f',
                      self.embedding.scope, tr_loss, val_loss)

    def spin(self, sess):
        self.update_dataset(sess=sess)

        if self.training_binary.num_data > 0:
            self.update_classifier()

            if self.tuning_binary.num_data > 0:
                self.optimize()
                self.visualize_classifier(data=self.validation_binary)


class EmbeddingLearnerNode(object):
    def __init__(self):
        self.sources = DataSources(**rospy.get_param('~sources'))
        self.frontend = rr.SARSFrontend(source=self.sources.state_source,
                                        **rospy.get_param('~sars_frontend'))
        self.frontend.register_callback(self.data_callback)

        self.network_args = rospy.get_param('~embedding/network')
        self.learner_args = rospy.get_param('~embedding/learning')
        self.classifier_args = rospy.get_param('~classification')
        self.registry = {}

        self.sess = tf.Session()

        self.plot_group = rr.PlottingGroup()

        spin_rate = rospy.get_param('~embedding_spin_rate')
        spin_dt = 1.0 / spin_rate
        self.embedding_timer = rospy.Timer(rospy.Duration(spin_dt),
                                           callback=self.spin_embedding)

        spin_rate = rospy.get_param('~classifier_spin_rate')
        spin_dt = 1.0 / spin_rate
        self.classifier_timer = rospy.Timer(rospy.Duration(spin_dt),
                                            callback=self.spin_classifier)

    def initialize_new(self, scope):
        with self.sess.graph.as_default():
            model = rr.EmbeddingNetwork(img_size=self.sources.img_size,
                                        vec_size=self.sources.belief_size,
                                        scope=scope,
                                        **self.network_args)
            learner = EmbeddingLearner(model=model,
                                       **self.learner_args)
            classifier = ClassifierLearner(embedding=learner,
                                           **self.classifier_args)

            # Initialize tensorflow variables
            model.initialize(self.sess)
            learner.initialize(self.sess)
            print 'Created net:\n%s' % str(model)

        for p in learner.get_plottables() + classifier.get_plottables():
            self.plot_group.add_plottable(p)

        return model, learner, classifier

    def data_callback(self, is_active, payload):
        if is_active:
            s, a, r, sn = payload
        else:
            s, a = payload
            r = 0

        if a not in self.registry:
            scope = 'embedding_%d' % len(self.registry)
            self.registry[a] = self.initialize_new(scope=scope)

        _, learner, _ = self.registry[a]
        learner.report_state_value(state=s, value=r)

    def spin_embedding(self, event):
        now = event.current_real.to_sec()
        self.frontend.spin(now)
        for val in self.registry.itervalues():
            model, learner, classifier = val
            learner.spin(sess=self.sess)

    def spin_classifier(self, event):
        for val in self.registry.itervalues():
            model, learner, classifier = val
            classifier.spin(sess=self.sess)


if __name__ == '__main__':
    rospy.init_node('image_embedding_learner')

    eln = EmbeddingLearnerNode()

    plot_rate = rospy.get_param('~plot_rate', 10.0)
    try:
        eln.plot_group.spin(plot_rate)
    except rospy.ROSInterruptException:
        pass
