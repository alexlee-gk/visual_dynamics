import numpy as np


def retime_with_vel_limits(positions, vel_limits_j):
    move_nj = positions[1:] - positions[:-1]
    dur_n = (np.abs(move_nj) / vel_limits_j[None, :]).max(axis=1)  # time according to velocity limit
    times = np.cumsum(np.r_[0, dur_n])

    return times
