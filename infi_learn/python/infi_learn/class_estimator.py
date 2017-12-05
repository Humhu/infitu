"""Classes for using embeddings to predict success
"""
import math
import numpy as np
import scipy.spatial.distance as ssd
import infi_learn as rr
import sklearn.neighbors as skn
import scipy.stats as sps


def rbf_window(x, xq, bandwidth):
    """A vectorized radial basis function kernel
    """
    dists = ssd.cdist(x, xq, metric='euclidean')
    return np.exp(-dists / bandwidth)


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
    return pos_factor / tot_factor


class ParzenNeighborsClassifier(object):
    """A binary classifier using Parzen windows on nearest-neighbors.
    """

    def __init__(self, radius, epsilon, window=None):
        self.nn_model = skn.NearestNeighbors(radius=radius)
        if window is None:
            self.window = lambda x, xq: rbf_window(x, xq, 1.0)
        else:
            self.window = window
        self.epsilon = epsilon

        self.embeddings = None
        self.labels = None

    def update_model(self, embeddings, labels):
        """Regenerates the dataset embedding and retrains the
        nearest neighbor model

        Parameters
        ----------
        embeddings : array of vectors, or numpy 2D
        """
        self.embeddings = embeddings
        self.labels = labels
        self.nn_model.fit(X=embeddings)

    def query(self, x):
        """Predicts the positive class probability and certainty
        at the specified embedding query points

        Parameters
        ----------
        x : 1D or 2D numpy
            Input(s) to query

        Returns
        -------
        pos : 1D array of bool
            Predicted class
        probs : 1D array
            Positive class probabilities
        """
        x = np.atleast_2d(x)
        # NOTE there's a slight speedup here if we use L2 distances for RBF
        # kernel
        inds = self.nn_model.radius_neighbors(x,
                                              return_distance=False)

        probs = []
        for i in inds:
            neighbors_x = self.embeddings[i]
            neighbors_y = self.labels[i]
            probi = regularized_parzen_estimate(xq=x,
                                                x=neighbors_x,
                                                y=neighbors_y,
                                                window=self.window,
                                                epsilon=self.epsilon)
            probs.append(probi)
        probs = np.array(probs)
        classes = probs >= 0.5 # NOTE rounding 0.5 to positive
        return classes, probs
