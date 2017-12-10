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

    def get_plottables(self):
        if self.use_image:
            return [self.image_source]
        else:
            return []

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
                 batch_num_sars=10, batch_num_terminal=10,
                 iters_per_spin=10, validation_period=10):
        self.problem = rr.EmbeddingProblem(model=model, **loss)

        self.training_base = adel.BasicDataset()
        self.validation_base = adel.BasicDataset()
        self.holdout = adel.HoldoutWrapper(training=self.training_base,
                                           holdout=self.validation_base,
                                           **holdout)

        self.training_sampler = adel.DatasetSampler(base=self.training_base,
                                                    method='uniform')
        self.training_sampled_sars = adel.SARSDatasetTranslator(
            base=self.training_sampler)
        self.training_sars = adel.SARSDatasetTranslator(
            base=self.training_base)
        self.validation_sars = adel.SARSDatasetTranslator(
            base=self.validation_base)

        self.iters_per_spin = iters_per_spin
        self.validation_period = validation_period
        self.sars_k = batch_num_sars
        self.term_k = batch_num_terminal

        self.spin_counter = 0

        self.error_plotter = rr.LineSeriesPlotter(title='Embedding losses over time %s' % model.scope,
                                                  xlabel='Spin iter',
                                                  ylabel='Embedding loss')
        self.embed_plotter = rr.LineSeriesPlotter(title='Validation embeddings %s' % model.scope,
                                                  xlabel='Embedding dimension 1',
                                                  ylabel='Embedding dimension 2')

    def initialize(self, sess):
        self.problem.initialize(sess)

    @property
    def scope(self):
        return self.problem.model.scope

    def report_data(self, key, data):
        self.holdout.report_data(key=key, data=data)

    def get_plottables(self):
        return [self.error_plotter, self.embed_plotter]

    def get_embedding(self, sess, on_training_data=True):
        if on_training_data:
            data = self.training_sars
        else:
            data = self.validation_sars
        return self.problem.run_embedding(sess=sess, data=data)

    def spin(self, sess):
        rospy.loginfo('Embedding %s has %d/%d (act/term) training, %d/%d validation',
                      self.scope, self.training_sars.num_sars, self.training_sars.num_terminals,
                      self.validation_sars.num_sars, self.validation_sars.num_terminals)

        train_loss = None
        for _ in range(self.iters_per_spin):
            if self.training_sars.num_sars < self.sars_k or self.training_sars.num_terminals < self.term_k:
                rospy.loginfo('Not enough data to train, skipping...')
                break

            self.training_sampler.sample_data(key=True, k=self.sars_k)
            self.training_sampler.sample_data(key=False, k=self.term_k)
            train_loss = self.problem.run_training(sess=sess,
                                                   data=self.training_sampled_sars)

        self.spin_counter += 1
        if train_loss is None:
            return
        rospy.loginfo('Training iter: %d loss: %f ' %
                      (self.spin_counter * self.iters_per_spin, train_loss))
        self.error_plotter.add_line('training', self.spin_counter, train_loss)

        if self.spin_counter % self.validation_period == 0:
            val_loss = self.problem.run_loss(sess=sess,
                                             data=self.validation_sars)
            rospy.loginfo('Validation loss: %f ' % val_loss)
            self.error_plotter.add_line(
                'validation', self.spin_counter, val_loss)
            val_embed, val_labels = self.problem.run_embedding(sess=sess,
                                                               data=self.validation_sars)
            pos_embeds = val_embed[val_labels]
            neg_embeds = val_embed[np.logical_not(val_labels)]
            self.embed_plotter.set_line('val+', pos_embeds[:, 0], pos_embeds[:, 1],
                                        marker='o', markerfacecolor='none', linestyle='none')
            self.embed_plotter.set_line('val-', neg_embeds[:, 0], neg_embeds[:, 1],
                                        marker='x', linestyle='none')


class ClassifierLearner(object):
    def __init__(self, embedding, classifier, holdout, optimizer,
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

        self.visualize = visualize
        self.vis_res = vis_res

        if self.visualize:
            # TODO Put scope in name
            self.heat_plotter = rr.ImagePlotter(vmin=0, vmax=1, title='Classifier')
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
            if y:
                self.report_binary.report_positive(x)
            else:
                self.report_binary.report_negative(x)

        embed, labels = self.embedding.get_embedding(sess=sess,
                                                     on_training_data=False)
        for x, y in izip(embed, labels):
            if y:
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

    def data_callback(self, key, payload):
        action = payload[1]
        if action not in self.registry:
            scope = 'embedding_%d' % len(self.registry)
            self.registry[action] = self.initialize_new(scope=scope)

        _, learner, _ = self.registry[action]
        learner.report_data(key=key, data=payload)

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
    plt.figure()
    plot_rate = rospy.get_param('~plot_rate', 10.0)
    try:
        eln.plot_group.spin(plot_rate)
    except rospy.ROSInterruptException:
        pass
