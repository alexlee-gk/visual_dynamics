from __future__ import division

import numpy as np

class Controller(object):
    def step(self, obs):
        raise NotImplementedError


class RandomController(Controller):
    def __init__(self, action_min, action_max):
        self.action_min = action_min
        self.action_max = action_max
        assert action_min.dtype == action_max.dtype
        self.action_type = action_min.dtype

    def step(self, obs):
        if self.action_type == np.int:
            action = np.array([np.random.random_integers(low, high) for (low, high) in zip(self.action_min, self.action_max)])
        else:
            action = self.action_min + np.random.random_sample(self.action_min.shape) * (self.action_max - self.action_min)
        return action


class ServoingController(Controller):
    def __init__(self, feature_predictor, alpha=1.0, vel_max=None, lambda_=0.0, w=None):
        self.predictor = feature_predictor
        self.alpha = alpha
        self.vel_max = vel_max
        self.lambda_ = lambda_
        self.w = w
        self._image_target = None
        self._y_target = None

    def step(self, image):
        if self.image_target is not None:
            x = image
            y = self.predictor.feature_from_input(x)

            # use model to optimize for action
            J = self.predictor.jacobian_control(x, None)
            if self.w is not None:
                JW = J * self.w[:, None]
            else:
                JW = J
            try:
                u = self.alpha * np.linalg.solve(JW.T.dot(J) + self.lambda_*np.eye(J.shape[1]), JW.T.dot(self.y_target - y))
            except np.linalg.LinAlgError:
                u = np.zeros(self.predictor.u_shape)
        else:
            u = np.zeros(self.predictor.u_shape)

        vel = u
        if self.vel_max is not None:
            vel = np.clip(vel, -self.vel_max, self.vel_max)
        return vel

    @property
    def image_target(self):
        return self._image_target.copy()

    @image_target.setter
    def image_target(self, image):
        self._image_target = image.copy()
        self._y_target = self.predictor.feature_from_input(self._image_target )

    @property
    def y_target(self):
        return self._y_target.copy()

    @y_target.setter
    def y_target(self, y):
        raise

    def set_target_obs(self, image_target):
        self.image_target = image_target


class SpecializedServoingController(ServoingController):
    def __init__(self, feature_predictor, pos_image_targets, neg_image_targets, alpha=1.0, vel_max=None, lambda_=0.0):
        from sklearn import linear_model
        Y_train = np.r_[feature_predictor.feature_from_input(pos_image_targets),
                        feature_predictor.feature_from_input(neg_image_targets)]
        label_train = np.r_[np.ones(len(pos_image_targets)),
                            -np.ones(len(neg_image_targets))]
        regr = linear_model.LinearRegression()
        regr.fit(Y_train, label_train)
        w = regr.coef_
        super(SpecializedServoingController, self).__init__(feature_predictor, alpha=alpha, vel_max=vel_max, lambda_=lambda_, w=w)
