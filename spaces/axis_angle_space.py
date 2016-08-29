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
        try:
            self.low, self.high = [np.asscalar(np.squeeze(limit)) for limit in [low, high]]
        except ValueError:
            raise ValueError("low and high should each contain a single number or be a single number")
        assert np.isscalar(self.low)
        assert np.isscalar(self.high)
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
        else:
            axis = self.axis
            angle = np.random.uniform(low=self.low, high=self.high)
        return angle * axis

    def contains(self, x):
        assert x.shape == self.shape
        axis, angle = tf.split_axis_angle(x, reference_axis=self.axis)
        if self.axis is None:
            axis_contained = True
        else:
            assert np.dot(axis, self.axis) >= 0.0
            axis_contained = np.dot(axis, self.axis) == 1.0
        angle_contained = (self.low <= angle <= self.high)
        return axis_contained and angle_contained

    def clip(self, x, out=None):
        assert x.shape == self.shape
        if self.axis is None:
            axis_clipped, angle = tf.split_axis_angle(x)
        else:
            x_projected = np.dot(x, self.axis) * self.axis
            axis_clipped, angle = tf.split_axis_angle(x_projected, reference_axis=self.axis)
            assert np.allclose(axis_clipped, self.axis)
        angle_clipped = np.clip(angle, self.low, self.high)
        x_clipped = angle_clipped * axis_clipped
        if out is not None:
            out[:] = x_clipped
            x_clipped = out
        return x_clipped

    @property
    def shape(self):
        """
        shape of data that this space handles
        """
        return (3,)

    def _get_config(self):
        config = super(AxisAngleSpace, self)._get_config()
        config.update({'low': self.low,
                       'high': self.high,
                       'axis': self.axis.tolist() if self.axis is not None else None})
        return config
