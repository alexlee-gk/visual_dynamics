"""
Simple functions on numpy arrays
"""
from __future__ import division
import numpy as np


def interp2d(x,xp,yp):
    "Same as np.interp, but yp is 2d"
    yp = np.asarray(yp)
    assert yp.ndim == 2
    return np.array([np.interp(x,xp,col) for col in yp.T]).T
