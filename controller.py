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
            # use model to optimize for action
            J, y = self.predictor.jacobian_control(x, None)
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
    def __init__(self, feature_predictor, pos_target_generator, neg_target_generator, alpha=1.0, vel_max=None, lambda_=0.0, pool_channels=True):
        from sklearn import linear_model
        pos_image_targets = []
        neg_image_targets = []
        for target_generator, image_targets in zip([pos_target_generator, neg_target_generator], [pos_image_targets, neg_image_targets]):
            for _ in range(target_generator.num_images):
                image_targets.append(target_generator.get_target()[0])
        pos_image_targets = np.asarray(pos_image_targets)
        neg_image_targets = np.asarray(neg_image_targets)
        if pool_channels:
            Z_train = np.r_[feature_predictor.mean_feature_from_input(pos_image_targets),
                            feature_predictor.mean_feature_from_input(neg_image_targets)]
            xlevels = feature_predictor.features_from_input(pos_image_targets[0])
            if 'x0' in xlevels and len(xlevels) > 1:
                image_dim = pos_image_targets.shape[1]
                Z_train = Z_train[:, image_dim:]
            label_train = np.r_[np.ones(len(pos_image_targets), dtype=np.int),
                                np.zeros(len(neg_image_targets), dtype=np.int)]
            regr = linear_model.LogisticRegression(penalty='l1', C=10e4)
            regr.fit(Z_train, label_train)
            w = np.squeeze(regr.coef_)
            w = np.maximum(w, 0)
            print("%d out of %d weights are non-zero"%((w != 0).sum(), len(w)))
            w_new = []
            for output_name, xlevel in xlevels.items():
                if output_name == 'x0' and len(xlevels) > 1:
                    w_new.append(np.zeros(np.prod(xlevel.shape)))
                else:
                    w_new.append(np.repeat(w[:xlevel.shape[0]], np.prod(xlevel.shape[1:])))
                    w = w[xlevel.shape[0]:]
            w = np.concatenate(w_new)
        else:
            Y_train = np.r_[feature_predictor.feature_from_input(pos_image_targets),
                            feature_predictor.feature_from_input(neg_image_targets)]
            xlevels = feature_predictor.features_from_input(pos_image_targets[0])
            if 'x0' in xlevels and len(xlevels) > 1:
                image_dim = np.prod(pos_image_targets.shape[1:])
                Y_train = Y_train[:, image_dim:]
            label_train = np.r_[np.ones(len(pos_image_targets), dtype=np.int),
                                np.zeros(len(neg_image_targets), dtype=np.int)]
            regr = linear_model.LogisticRegression(penalty='l1', C=10e4)
            regr.fit(Y_train, label_train)
            w = np.squeeze(regr.coef_)
            inds = w.argsort()
            w_top = np.zeros_like(w)
            w_top[inds[-25:]] = w[inds[-25:]]
            w_top /= w_top.max()
            w = w_top
            print("%d out of %d weights are non-zero"%((w != 0).sum(), len(w)))
            if 'x0' in xlevels and len(xlevels) > 1:
                w = np.r_[np.zeros(image_dim), w]
        self.w = w
        super(SpecializedServoingController, self).__init__(feature_predictor, alpha=alpha, vel_max=vel_max, lambda_=lambda_, w=w)
