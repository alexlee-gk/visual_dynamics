import numpy as np
from policy import Policy


class ServoingPolicy(Policy):
    def __init__(self, predictor, alpha=1.0, lambda_=0.0, w=None):
        self.predictor = predictor
        self.action_transformer = self.predictor.transformers['u']
        # self.action_transformer = self.predictor.transformers.get('action', utils.transformer.Transformer())
        self.alpha = alpha
        self.lambda_ = lambda_
        self.w = w
        self._image_target = None
        self._y_target = None
        self.u_prev = np.zeros(self.predictor.input_shapes[1])  # u_prev is in original (non-preprocessed) units

    def act(self, obs):
        image = obs[0]
        if self.image_target is not None:
            # use model to optimize for action
            jac, next_feature = self.predictor.feature_jacobian(image, self.u_prev)  # Jacobian is in preprocessed units
            J = np.concatenate(jac)
            y_next_pred = np.concatenate([f.flatten() for f in next_feature])
            if self.alpha == 1.0:
                y_target = self.y_target
            else:
                # TODO: get feature in the same pass as when the jacobian is computed
                feature = self.predictor.feature(image)
                y = np.concatenate([f.flatten() for f in feature])
                y_target = self.alpha * self.y_target + (1 - self.alpha) * y
            if self.w is None:
                WJ = J
            elif self.w.ndim == 1 and self.w.shape == (J.shape[0],):
                WJ = J * self.w[:, None]
            elif self.w.ndim == 2 and self.w.shape == (J.shape[0], J.shape[0]):
                WJ = self.w.dot(J)
            elif self.w.ndim == 2 and self.w.shape[0] == J.shape[0]:
                WJ = self.w.dot(self.w.T.dot(J))
            else:
                raise ValueError('invalid weights w, %r' % self.w)
            try:
                u = np.linalg.solve(WJ.T.dot(J) + self.lambda_ * np.eye(J.shape[1]),
                                    WJ.T.dot(y_target - y_next_pred
                                             + J.dot(self.action_transformer.preprocess(self.u_prev))))  # preprocessed units
            except np.linalg.LinAlgError:
                u = None
        else:
            u = None

        if u is None:
            u = np.zeros(self.predictor.input_shapes[1])
        else:
            u = self.action_transformer.deprocess(u)
        self.u_prev = u.copy()  # original units
        return u

    @property
    def image_target(self):
        return self._image_target.copy()

    @image_target.setter
    def image_target(self, image):
        self._image_target = image.copy()
        self._y_target = np.concatenate([f.flatten() for f in self.predictor.feature(self._image_target)])

    @property
    def y_target(self):
        return self._y_target.copy()

    @y_target.setter
    def y_target(self, y):
        raise

    def set_image_target(self, image_target):
        self.image_target = image_target

    def set_target(self, target_obs):
        self.image_target = target_obs[0]

    def _get_config(self):
        config = super(ServoingPolicy, self)._get_config()
        config.update({'predictor': self.predictor,
                       'alpha': self.alpha,
                       'lambda_': self.lambda_,
                       'w': self.w.tolist() if isinstance(self.w, np.ndarray) else self.w})
        return config
