import numpy as np
from spaces import Space
import utils.transformations as tf


class AxisAngleSpace(Space):
    """
    SO(3) space where the rotation is represented as an axis-angle vector in
    R^3 where its magnitude is constrained within an interval and the axis can
    optionally be constrained. If the axis is not constrained, then the
    absolute value of low and high should be equal to each other.
    """
    def __init__(self, low, high, axis=None):
        """
        TODO: handle angle wrap-around
        """
        self.low = np.squeeze(low)[None]
        self.high = np.squeeze(high)[None]
        if self.low.shape != (1,) or self.high.shape != (1,):
            raise ValueError("low and high should each contain a single number or be a single number")
        self.axis = axis / np.linalg.norm(axis) if axis is not None else None
        if self.axis is None:
            assert -self.low == self.high

    def sample(self):
        if self.axis is None:
            axis_angle = tf.axis_angle_from_matrix(tf.random_rotation_matrix())
            axis, angle = tf.split_axis_angle(axis_angle)
            if not (self.low <= angle <= self.high):
                # rescale angle from [-pi, pi) to [-low_angle, high_angle)
                angle = self.low + (self.high - self.low) * (angle + np.pi) / (2 * np.pi)
            return angle * axis
        else:
            angle = np.random.uniform(low=self.low, high=self.high, size=self.shape)
            return angle

    def contains(self, x):
        angle = np.linalg.norm(x)
        return x.shape == self.shape and (self.low <= angle <= self.high)

    def clip(self, x, out=None):
        assert x.shape == self.shape
        if self.axis is None:
            axis, angle = tf.split_axis_angle(x)
            x_clipped = np.clip(angle, self.low, self.high) * axis
        else:
            x_clipped = np.clip(x, self.low, self.high)
        if out is not None:
            out[:] = x_clipped
            x_clipped = out
        return x_clipped

    @property
    def shape(self):
        """
        shape of data that this space handles
        """
        if self.axis is None:
            return (3,)
        else:
            return (1,)

    def _get_config(self):
        config = super(AxisAngleSpace, self)._get_config()
        config.update({'low': np.asscalar(self.low),
                       'high': np.asscalar(self.high),
                       'axis': self.axis.tolist() if self.axis is not None else None})
        return config
