import numpy as np



def minibatch_mean_std(batch_data):
    n = 0
    s = 0.0
    s2 = 0.0
    for minibatch_data in np.split(batch_data, 100):
        n += len(minibatch_data)
        s += minibatch_data.sum(axis=0)
        s2 += (minibatch_data ** 2).sum(axis=0)
    mean = s / n
    std = np.sqrt((s2 - (s ** 2) / n) / n)
    return mean, std


class OnlineStatistics(object):
    def __init__(self, axis=0):
        self.axis = axis
        self.n = None
        self.s = None
        self.s2 = None
        self.reset()

    def reset(self):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0

    def add_data(self, data):
        self.n += data.shape[axis]
        self.s += data.sum(axis=self.axis)
        self.s2 += (data ** 2).sum(axis=self.axis)

    @property
    def mean(self):
        return self.s / self.n

    @property
    def std(self):
        return np.sqrt((self.s2 - (self.s ** 2) / self.n) / self.n)


# def online_mean_std(X):
#     n = 0
#     mean = 0
#     var = 0  # before divided by n
#     for x in X:
#         n += 1
#         delta = x - mean
#         mean += delta / n
#
#         mean = s / n
#
#
#         var += delta * (x - mean)
#     return mean, np.sqrt(var / n)


def online_stats(X):
    """
    Converted from John D. Cook
    http://www.johndcook.com/blog/standard_deviation/
    """
    prev_mean = None
    prev_var = None
    n_seen = 0
    for i in range(len(X)):
        n_seen += 1
        if prev_mean is None:
            prev_mean = X[i]
            prev_var = 0.
        else:
            curr_mean = prev_mean + (X[i] - prev_mean) / n_seen
            curr_var = prev_var + (X[i] - prev_mean) * (X[i] - curr_mean)
            prev_mean = curr_mean
            prev_var = curr_var
    # n - 1 for sample variance, but numpy default is n
    return prev_mean, np.sqrt(prev_var / n_seen)

from numpy.testing import assert_almost_equal
X = np.random.rand(100000, 512)

online_stats = OnlineStatistics()
for x in np.split(X, 1000):
    online_stats.add_data(x)

tm = X.mean(axis=0)
ts = X.std(axis=0)
sm, ss = online_stats.mean, online_stats.std
assert_almost_equal(tm, sm)
assert_almost_equal(ts, ss)
