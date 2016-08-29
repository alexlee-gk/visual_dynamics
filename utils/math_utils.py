from __future__ import division, print_function
import numpy as np


def divide_nonzero(a, b):
    """
    Return a/b for the nonzero elements of b and return 0 for the zero elements of b.
    """
    shape = (a * b).shape
    nonzero = b != 0
    c = np.zeros(shape)
    try:
        if a.shape == shape:
            a = a[nonzero]
    except AttributeError:
        pass
    try:
        if b.shape == shape:
            b = b[nonzero]
    except AttributeError:
        pass
    c[nonzero] = a / b
    return c


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


def clip_pos_aa(pos_aa, min_dof_limits, max_dof_limits):
    assert 3 <= len(pos_aa) <= 6
    assert 3 <= len(min_dof_limits) <= 4
    assert 3 <= len(max_dof_limits) <= 4
    pos, aa = np.split(pos_aa, [3])
    pos = np.clip(pos, min_dof_limits[:3], max_dof_limits[:3])
    min_angle = min_dof_limits[3] if len(min_dof_limits) > 3 else float('-inf')
    max_angle = max_dof_limits[3] if len(max_dof_limits) > 3 else float('inf')
    angle = np.linalg.norm(aa)
    axis = aa / angle if angle else np.array([0, 0, 1])
    angle = np.clip(angle, min_angle, max_angle)
    aa = axis * angle
    return np.concatenate([pos, aa])


def pack_image(image, fixed_point_min=0.01, fixed_point_max=100.0):
    assert image.ndim == 3 and image.shape[2] == 1
    image = image.squeeze()
    fixed_point_image = np.clip(image, fixed_point_min, fixed_point_max)
    fixed_point_image = (2 ** 24) * (fixed_point_image - fixed_point_min) / (fixed_point_max - fixed_point_min)
    fixed_point_image = fixed_point_image.astype(np.uint32)
    fixed_point_image = fixed_point_image.view(dtype=np.uint8).reshape(fixed_point_image.shape + (4,))[..., :-1]
    return fixed_point_image


def unpack_image(fixed_point_image, fixed_point_min=0.01, fixed_point_max=100.0):
    fixed_point_image = np.concatenate([fixed_point_image, np.zeros(fixed_point_image.shape[:-1] + (1,), dtype=np.uint8)], axis=-1)
    fixed_point_image = fixed_point_image.view(np.uint32).astype(int).squeeze()
    fixed_point_image = fixed_point_min + fixed_point_image * (fixed_point_max - fixed_point_min) / (2 ** 24)
    image = fixed_point_image.astype(np.float32)
    image = np.expand_dims(image, axis=-1)
    return image
