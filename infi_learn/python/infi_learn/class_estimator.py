"""Classes for using X to predict success
"""
import numpy as np
import scipy.stats as ss
import scipy.spatial.distance as ssd
import sklearn.neighbors as skn


def rbf_window(x, xq, bw):
    """A vectorized radial basis function kernel
    """
    #dists = ssd.cdist(x, xq, metric='euclidean')
    #return np.exp(-dists / (2 * bw * bw))
    
    # return np.atleast_1d(ss.multivariate_normal.pdf(x=x-xq, cov=np.diag(bw)))

    D = np.diag(bw)
    deltas = (x - xq).T
    ips = np.einsum('ji,jk,ki->i',deltas, D, deltas)
    return np.exp( -ips * 0.5 )


def compute_classification_loss(classifier, x, y, params):
    """Computes the classification loss of a classifier with the given
    log hyperparameters on a dataset (x,y)
    """
    classifier.set_log_params(params)
    pp = classifier.query(x)
    return np.mean(np.abs(y - pp))

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
        self.bw = bandwidth
        self.epsilon = epsilon

        if window is None:
            self.window = rbf_window
        else:
            self.window = window
        if radius is None:
            # Rule of thumb
            self.radius = 3 * self.bw
        else:
            self.radius = radius

        self.X = None
        self.labels = None

    def set_log_params(self, p):
        """Set the bandwidth and regularization log-parameters.
        Assumes [eps, rad, bw] ordering for p.
        """
        params = np.exp(p)
        self.epsilon = params[0]
        self.radius = params[1]
        self.bw = params[2:]

    def get_log_params(self):
        p = np.hstack([self.epsilon, self.radius, self.bw])
        return np.log(p)

    def update_model(self, X, labels):
        """Regenerates the dataset embedding and retrains the
        nearest neighbor model

        Parameters
        ----------
        X : array of vectors, or numpy 2D
        """
        self.X = X
        self.labels = labels
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
