"""Learner setup for embedding task
"""
import numpy as np
import adel
from infi_learn.embedding import EmbeddingProblem
from infi_learn.plotting import LineSeriesPlotter, FilterPlotter, ScatterPlotter


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
        x = [self.problem.run_embedding(sess=sess, data=adel.LabeledDatasetTranslator(chunk))
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
                                                   data=self.train_sampled)

        self.spin_counter += 1
        if train_loss is None:
            return
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
                                           data=self.val_sampled)
                val_losses.append(vl)
            mean_val_loss = np.mean(val_losses)
            print 'Validation loss: %f ' % mean_val_loss

            self.error_plotter.add_line('validation',
                                        self.spin_counter,
                                        mean_val_loss)
            val_embed, val_labels = self.get_embedding(sess=sess,
                                                       on_training_data=False)
            self.embed_plotter.set_scatter('val',
                                           x=val_embed[:, 0],
                                           y=val_embed[:, 1],
                                           c=val_labels,
                                           marker='o')
