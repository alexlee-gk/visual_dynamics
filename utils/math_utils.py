import numpy as np


def sample_interval(min_limit, max_limit):
    assert min_limit.shape == max_limit.shape
    assert min_limit.dtype == max_limit.dtype
    if min_limit.dtype == np.int:
        return np.array([np.random.random_integers(low, high) for (low, high) in zip(min_limit, max_limit)])
    else:
        return min_limit + np.random.random_sample(min_limit.shape) * (max_limit - min_limit)


def axis2quat(axis, angle):
    axis = np.asarray(axis)
    axis = 1.0*axis/axis.sum();
    return np.append(np.cos(angle/2.0), axis*np.sin(angle/2.0))


def quaternion_multiply(*qs):
    if len(qs) == 2:
        q0, q1 = qs
        return np.array([-q1[1]*q0[1] - q1[2]*q0[2] - q1[3]*q0[3] + q1[0]*q0[0],
                          q1[1]*q0[0] + q1[2]*q0[3] - q1[3]*q0[2] + q1[0]*q0[1],
                         -q1[1]*q0[3] + q1[2]*q0[0] + q1[3]*q0[1] + q1[0]*q0[2],
                          q1[1]*q0[2] - q1[2]*q0[1] + q1[3]*q0[0] + q1[0]*q0[3]])
    else:
        return quaternion_multiply(qs[0], quaternion_multiply(*qs[1:]))
