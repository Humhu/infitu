import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import infi_learn as rr
import optim

# 1. Generate data
n_dim = 2
n_clusters = 1


def sample_points(n_train, n_test, offset):
    centers = np.random.uniform(low=-1, high=1, size=(n_clusters, n_dim))
    covs = []
    for _ in range(n_clusters):
        covs.append(np.diag(np.random.uniform(size=n_dim)))

    def sample(n):
        return [np.random.multivariate_normal(x, C, size=n)
                for x, C in zip(centers, covs)]
    train_points = np.vstack(sample(n_train)) + offset
    test_points = np.vstack(sample(n_test)) + offset
    return train_points, test_points


n_train_pts_cluster = 1000
n_test_pts_cluster = 100

pos_offset = np.random.uniform(low=-1, high=1, size=n_dim)
pos_train_pts, pos_test_pts = sample_points(n_train_pts_cluster,
                                            n_test_pts_cluster,
                                            pos_offset)
neg_offset = np.random.uniform(low=-1, high=1, size=n_dim)
neg_train_pts, neg_test_pts = sample_points(n_train_pts_cluster,
                                            n_test_pts_cluster,
                                            neg_offset)
pos_train_labels = np.ones(len(pos_train_pts), dtype=bool)
pos_test_labels = np.ones(len(pos_test_pts), dtype=bool)
neg_train_labels = np.zeros(len(neg_train_pts), dtype=bool)
neg_test_labels = np.zeros(len(neg_test_pts), dtype=bool)

train_pts = np.vstack((pos_train_pts, neg_train_pts))
train_labels = np.hstack((pos_train_labels, neg_train_labels))
test_pts = np.vstack((pos_test_pts, neg_test_pts))
test_labels = np.hstack((pos_test_labels, neg_test_labels))

plt.ion()

# 2. Train models
classifier = rr.ParzenNeighborsClassifier(bandwidth=[0.5, 0.5],
                                          epsilon=1e-6,
                                          radius=1)


classifier.update_model(X=train_pts, labels=train_labels)

mins = np.min(np.vstack((pos_train_pts, neg_train_pts)), axis=0)
maxs = np.max(np.vstack((pos_train_pts, neg_train_pts)), axis=0)
n_vis = 30
vis_lims = [np.linspace(start=l, stop=u, num=n_vis)
            for l, u in zip(mins, maxs)]
vis_pts = np.meshgrid(*vis_lims)
vis_x = np.vstack([vi.flatten() for vi in vis_pts]).T


def reshape_vis(x):
    return np.reshape(x, (n_vis, n_vis))


def visualize(title, vis_train=False):
    pos_vis_prob = classifier.query(vis_x)

    plt.figure()
    #plt.contour(vis_pts[0], vis_pts[1], reshape_vis(pos_vis_prob))
    plt.imshow(reshape_vis(pos_vis_prob), origin='lower',
               extent=(mins[0], maxs[0], mins[1], maxs[1]),
               cmap=cm.bwr,
               vmin=0, vmax=1)
    if vis_train:
        plt.plot(pos_train_pts[:, 0], pos_train_pts[:, 1], 'ko', mew=0.01)
        plt.plot(neg_train_pts[:, 0], neg_train_pts[:, 1], 'kx', mew=1.0)
    else:
        plt.plot(pos_test_pts[:, 0], pos_test_pts[:, 1], 'ko', mew=0.01)
        plt.plot(neg_test_pts[:, 0], neg_test_pts[:, 1], 'kx', mew=1.0)
    plt.title(title)
    plt.colorbar()


def obj(p):
    return rr.compute_classification_loss(classifier=classifier,
                                          x=test_pts,
                                          y=test_labels,
                                          params=p)


def print_params():
    print 'bandwidth: %s epsilon: %f radius: %f' % \
        (str(classifier.bw), classifier.epsilon, classifier.radius)


# 3. Perform hyperparameter optimization
p0 = classifier.log_params
visualize('Initial validation loss is %f' % obj(p0))
print_params()

cma_optimizer = optim.CMAOptimizer(mode='min')
cma_optimizer.lower_bounds = [-15, -2, -2, -2]
cma_optimizer.upper_bounds = [0, 2, 2, 2]

pcma = cma_optimizer.optimize(x_init=p0, func=obj)[0]
visualize('CMA final validation loss is %f' % obj(pcma))
print_params()

bfgs_optimizer = optim.BFGSOptimizer(mode='min', num_restarts=3)
bfgs_optimizer.lower_bounds = [-15, -2, -2, -2]
bfgs_optimizer.upper_bounds = [0, 2, 2, 2]

pbfgs = bfgs_optimizer.optimize(x_init=p0, func=obj)[0]
visualize('BFGS final validation loss is %f' % obj(pbfgs))
print_params()