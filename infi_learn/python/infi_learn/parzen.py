"""Classes and functions for Parzen-window classification
"""

from itertools import izip
import numpy as np
import sklearn.neighbors as skn
from infi_learn.classification import compute_classification_loss


def rbf_window(x, xq, bw):
    """A vectorized radial basis function kernel
    """
    #dists = ssd.cdist(x, xq, metric='euclidean')
    # return np.exp(-dists / (2 * bw * bw))

    # return np.atleast_1d(ss.multivariate_normal.pdf(x=x-xq, cov=np.diag(bw)))

    D = np.diag(bw)
    deltas = (x - xq).T
    ips = np.einsum('ji,jk,ki->i', deltas, D, deltas)
    return np.exp(-ips * 0.5)


def regularized_parzen_estimate(xq, x, y, window, epsilon=0):
    """Returns probability of positive class at query xq

    Parameters
    ----------
    xq : numpy 1D or list of numpy 1D
        Query input
    x : numpy 1D or list of numpy 1D
        Training inputs
    y : numpy 1D boolean
        Training classes
    window : windowing function
        Parzen kernel function
    epsilon : float (default 0)
        Regularization

    Returns
    -------
    pos_probs : probabilities of positive class at query
    """
    xq = np.atleast_2d(xq)
    x = np.atleast_2d(x)

    K = window(x, xq)
    pos_factor = np.sum(K[y], axis=0) + epsilon
    tot_factor = np.sum(K, axis=0) + 2 * epsilon
    return np.atleast_1d(pos_factor / tot_factor)


class ParzenNeighborsClassifier(object):
    """A binary classifier using Parzen windows on nearest-neighbors.
    """

    def __init__(self, bandwidth, epsilon, radius=None, window=None):
        self.nn_model = skn.NearestNeighbors()

        if np.iterable(bandwidth):
            self.bw = [float(bi) for bi in bandwidth]
        else:
            self.bw = float(bandwidth)

        self.epsilon = float(epsilon)

        if window is None:
            self.window = rbf_window
        else:
            self.window = window
        if radius is None:
            # Rule of thumb
            self.radius = 3 * np.mean(self.bw)
        else:
            self.radius = radius

        self.X = None
        self.labels = None

    @property
    def log_params(self):
        p = np.hstack([self.epsilon, self.bw])
        return np.log(p)

    @log_params.setter
    def log_params(self, p):
        """Set the bandwidth and regularization log-parameters.
        Assumes [eps, rad, bw] ordering for p.
        """
        params = np.exp(p)
        self.epsilon = params[0]
        self.bw = params[1:]
        self.radius = 3 * max(self.bw)

    def print_params(self):
        return 'epsilon: %f radius: %f bandwidth: %s ' \
            % (self.epsilon, self.radius, str(self.bw))

    def update_model(self, X, labels):
        """Regenerates the dataset embedding and retrains the
        nearest neighbor model

        Parameters
        ----------
        X : array of vectors, or numpy 2D
        """
        self.X = np.array(X)
        self.labels = np.array(labels)
        self.nn_model.fit(X=X)

    def query(self, x):
        """Predicts the positive class probability at the specified
        embedding query points

        Parameters
        ----------
        x : 1D or 2D numpy
            Input(s) to query

        Returns
        -------
        probs : 1D array
            Positive class probabilities
        """
        x = np.atleast_2d(x)
        # NOTE there's a slight speedup here if we use L2 distances for RBF
        # kernel
        inds = self.nn_model.radius_neighbors(x,
                                              radius=self.radius,
                                              return_distance=False)

        def win_func(x, xq): return self.window(x=x, xq=xq, bw=self.bw)
        probs = []
        for j, i in enumerate(inds):
            if len(i) == 0:
                probs.append(np.atleast_1d(0.5))
                continue
            neighbors_x = self.X[i]
            neighbors_y = self.labels[i]
            probi = regularized_parzen_estimate(xq=x[j],
                                                x=neighbors_x,
                                                y=neighbors_y,
                                                window=win_func,
                                                epsilon=self.epsilon)
            probs.append(probi)
        probs = np.squeeze(probs)
        return probs


def optimize_parzen_neighbors(classifier, dataset, optimizer):
    """Optimizes a classifier using a dataset and optimizer

    Returns
    -------
    err : float
        The final optimized classification loss
    """

    x = dataset.all_positives + dataset.all_negatives
    y = [True] * dataset.num_positives + [False] * dataset.num_negatives

    def objective(p):
        return compute_classification_loss(classifier=classifier,
                                           x=x, y=y, params=p)

    p, err = optimizer.optimize(x_init=classifier.log_params, func=objective)
    classifier.log_params = p
    return err


class ParzenClassifierLearner(object):
    """Maintains dataset splits and plotting around a ParzenNeighborsClassifier

    Parameters
    ==========
    embed_func : Function taking BinaryDatasetTranslator as input
        Returns embedding of dataset data
    classifier : ParzenNeighborsClassifier
        The classifier to train
    holdout : dict
        Args for the holdout approach
    optimizer : dict
        Args for the hyperparameter optimizer
    """

    def __init__(self, classifier, holdout, optimizer,
                 visualize=True, vis_res=10):

        self.classifier = classifier
        self.optimizer = optim.parse_optimizers(**optimizer)

        self.training_base = adel.BasicDataset()
        self.tuning_base = adel.BasicDataset()
        self.tuning_holdout = adel.HoldoutWrapper(training=self.training_base,
                                                  holdout=self.tuning_base,
                                                  **holdout)
        self.report_binary = adel.BinaryDatasetTranslator(self.tuning_holdout)

        self.training_binary = adel.BinaryDatasetTranslator(self.training_base)
        self.tuning_binary = adel.BinaryDatasetTranslator(self.tuning_base)

        self.validation_base = adel.BasicDataset()
        self.validation_binary = adel.BinaryDatasetTranslator(
            self.validation_base)

        self.update_counter = 0

        self.visualize = visualize
        self.vis_res = vis_res

        if self.visualize:
            # TODO Put scope in name
            self.heat_plotter = rr.ImagePlotter(vmin=0, vmax=1,
                                                title='Classifier')
            self.point_plotter = rr.LineSeriesPlotter(other=self.heat_plotter)
            self.error_plotter = rr.LineSeriesPlotter(title='Logistic losses over time',
                                                      xlabel='Spin iter',
                                                      ylabel='Logistic loss')
            self.roc_plotter = rr.LineSeriesPlotter(title='ROC',
                                                    xlabel='False positive rate',
                                                    ylabel='True positive rate')

    def get_plottables(self):
        if self.visualize:
            return [self.heat_plotter, self.point_plotter, self.error_plotter, self.roc_plotter]
        else:
            return []

    def train(self, training_data, holdout_data=None):
        self.tuning_holdout.clear()
        self.validation_base.clear()
        for s, c in training_data.all_data():
            self.report_binary.report_data(s=s, c=c)

        if holdout_data is not None:
            for s, c in holdout_data.all_data():
                self.validation_binary.report_data(s=s, c=c)

        self.update_counter += 1
        print 'Classifier has %d/%d (pos/neg) training, %d/%d tuning, %d/%d validation' % \
            (self.training_binary.num_positives, self.training_binary.num_negatives,
             self.tuning_binary.num_positives, self.tuning_binary.num_negatives,
             self.validation_binary.num_positives, self.validation_binary.num_negatives)

        self.update_classifier()
        self.optimize_hyperparameters()

    def update_classifier(self):
        inputs, classes = zip(*self.training_binary.all_data)
        self.classifier.update_model(X=inputs, labels=classes)

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

        vis_probs = self.classifier.query(vis_x)
        pimg = np.reshape(vis_probs, (self.vis_res, self.vis_res))

        self.heat_plotter.set_image(img=pimg,
                                    extents=(mins[0], maxs[0], mins[1], maxs[1]))
        if len(pos) > 0:
            self.point_plotter.set_line(name='pos', x=pos[:, 0], y=pos[:, 1],
                                        color='k', marker='o', markerfacecolor='none',
                                        linestyle='none')
        if len(neg) > 0:
            self.point_plotter.set_line(name='neg', x=neg[:, 0], y=neg[:, 1],
                                        color='k', marker='x', linestyle='none')

        # Display ROC
        all_probs = self.classifier.query(all_data)
        all_labels = [True] * data.num_positives + [False] * data.num_negatives
        tpr, fpr = rr.compute_threshold_roc(probs=all_probs, labels=all_labels)
        self.roc_plotter.set_line(name='ROC', x=fpr, y=tpr)
        auc = rr.compute_auc(tpr, fpr)
        print 'Classifier AUC: %f' % auc

    def optimize_hyperparameters(self):
        rr.optimize_parzen_neighbors(classifier=self.classifier,
                                     dataset=self.tuning_binary,
                                     optimizer=self.optimizer)
        print 'Classifier params: %s' % self.classifier.print_params()

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
        print 'Classifier training loss: %f validation loss: %f' % \
            (tr_loss, val_loss)

        self.error_plotter.add_line('training', self.update_counter, tr_loss)
        self.error_plotter.add_line('validation',
                                    self.update_counter,
                                    val_loss)
