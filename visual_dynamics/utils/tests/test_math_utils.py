import numpy as np
from nose2 import tools

from visual_dynamics import utils


@tools.params(((1000, 25), 10, 0),
              ((1000, 25), 10, 1),
              ((1000, 25), 77, 0),
              ((1000, 1, 2, 3), 10, (0, 3))
              )
def test_online_statistics(shape, batch_size, axis):
    online_stats = utils.OnlineStatistics(axis=axis)
    X = np.random.random(shape)
    if isinstance(axis, (list, tuple)):
        data_size = np.prod([X.shape[ax] for ax in axis])
    else:
        data_size = X.shape[axis]
    curr_ind = 0
    while curr_ind < data_size:
        slices = []
        for i in range(X.ndim):
            if i == axis:
                slices.append(slice(curr_ind, min(curr_ind + batch_size, data_size)))
            else:
                slices.append(slice(None))
        batch_data = X[slices]
        online_stats.add_data(batch_data)
        curr_ind += batch_size
    mean = X.mean(axis=axis)
    std = X.std(axis=axis)
    assert np.allclose(mean, online_stats.mean)
    assert np.allclose(std, online_stats.std)
