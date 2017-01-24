"""
Resample time serieses to reduce the number of datapoints
"""
import numpy as np
import scipy.interpolate as si


def get_velocities(positions, times, tol):
    positions = np.atleast_2d(positions)
    n = len(positions)
    deg = min(3, n - 1)

    good_inds = np.r_[True, (abs(times[1:] - times[:-1]) >= 1e-6)]
    good_positions = positions[good_inds]
    good_times = times[good_inds]

    if len(good_inds) == 1:
        return np.zeros(positions[0:1].shape)

    (tck, _) = si.splprep(good_positions.T, s=tol ** 2 * (n + 1), u=good_times, k=deg)
    # smooth_positions = np.r_[si.splev(times,tck,der=0)].T
    velocities = np.r_[si.splev(times, tck, der=1)].T
    return velocities
