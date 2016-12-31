from __future__ import division, print_function
import numpy as np
import lasagne.layers as L
import theano
import theano.tensor as T
import time
from policy import Policy
import spaces
import utils


class ServoingPolicy(Policy):
    def __init__(self, predictor, alpha=1.0, lambda_=0.0, w=None, use_constrained_opt=False):
        self.predictor = predictor
        self.action_transformer = self.predictor.transformers['u']
        self.action_space = utils.from_config(self.predictor.environment_config['action_space'])
        self.alpha = alpha
        if np.isscalar(lambda_):
            self.lambda_ = lambda_ * np.ones(self.action_space.shape)
        else:
            self.lambda_ = lambda_
        feature_names = self.predictor.feature_name if isinstance(self.predictor.feature_name, list) else [self.predictor.feature_name]
        feature_shapes = L.get_output_shape([self.predictor.pred_layers[name] for name in feature_names])
        self.repeats = []
        for feature_shape in feature_shapes:
            self.repeats.extend([np.prod(feature_shape[2:])] * feature_shape[1])
        self.w = w if w is not None else np.ones(len(self.repeats))
        self.bias = np.array(0.0)  # used for fitted Q iteration
        self._image_target = None
        self._y_target = None
        self.use_constrained_opt = use_constrained_opt

    def phi(self, states, actions, preprocessed=False, with_constant=True):
        """
        Corresponds to the linearized objective

        The following should be true
        phi = self.phi(states, actions)
        theta = np.append(self.w, self.lambda_)
        linearized_objectives = [self.linearized_objective(state, action, with_constant=False) for (state, action) in zip(states, actions)]
        objectives = phi.dot(theta)
        assert np.allclose(objectives, linearized_objectives)
        """
        batch_size = len(states)
        if preprocessed:
            batch_image = np.array([obs[0] for (obs, target_obs) in states])
            batch_target_image = np.array([target_obs[0] for (obs, target_obs) in states])
            batch_u = np.array(actions)
        else:
            batch_image, = self.predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
            batch_target_image, = self.predictor.preprocess([target_obs[0] for (obs, target_obs) in states],
                                                            batch_size=len(states))
            batch_u = np.array([self.action_transformer.preprocess(action) for action in actions])
        batch_target_feature = self.predictor.feature(batch_target_image, preprocessed=True)
        batch_y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_target_feature], axis=1)
        if self.alpha != 1.0:
            batch_feature = self.predictor.feature(batch_image, preprocessed=True)
            batch_y = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_feature], axis=1)
            batch_y_target = self.alpha * batch_y_target + (1 - self.alpha) * batch_y
        action_lin = np.zeros(self.action_space.shape)
        u_lin = self.action_transformer.preprocess(action_lin)
        batch_jac, batch_next_feature = self.predictor.feature_jacobian(batch_image, np.array([u_lin] * batch_size), preprocessed=True)
        batch_J = np.concatenate(batch_jac, axis=1)
        batch_y_next_pred = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_next_feature], axis=1)

        batch_z = batch_y_target - batch_y_next_pred + batch_J.dot(u_lin)
        cs_repeats = np.cumsum(np.r_[0, self.repeats])
        slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
        batch_A_split = np.array([np.einsum('nij,nik->njk', batch_J[:, s], batch_J[:, s]) for s in slices])
        batch_b_split = np.array([np.einsum('nij,ni->nj', batch_J[:, s], batch_z[:, s]) for s in slices])
        if with_constant:
            batch_c_split = np.array([np.einsum('ni,ni->n', batch_z[:, s], batch_z[:, s]) for s in slices])
        else:
            batch_c_split = 0.0

        phi_errors = (np.einsum('lnj,nj->ln', np.einsum('lnjk,nk->lnj', batch_A_split, batch_u), batch_u)
                      - 2 * np.einsum('lnj,nj->ln', batch_b_split, batch_u)
                      + batch_c_split).T
        phi_actions = batch_u ** 2
        phi = np.c_[phi_errors / self.repeats, phi_actions]
        return phi

    def pi(self, states, preprocessed=False):
        """
        Corresponds to the linearized objective

        The following should be true
        actions_pi = self.pi(states)
        actions_act = [self.act(state) for state in states]
        assert np.allclose(actions_pi, actions_act)
        """
        if self.w.shape != (len(self.repeats),):
            raise NotImplementedError
        batch_size = len(states)
        if preprocessed:
            batch_image = np.array([obs[0] for (obs, target_obs) in states])
            batch_target_image = np.array([target_obs[0] for (obs, target_obs) in states])
        else:
            batch_image, = self.predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
            batch_target_image, = self.predictor.preprocess([target_obs[0] for (obs, target_obs) in states], batch_size=len(states))
        batch_target_feature = self.predictor.feature(batch_target_image, preprocessed=True)
        batch_y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_target_feature], axis=1)
        if self.alpha != 1.0:
            batch_feature = self.predictor.feature(batch_image, preprocessed=True)
            batch_y = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_feature], axis=1)
            batch_y_target = self.alpha * batch_y_target + (1 - self.alpha) * batch_y
        action_lin = np.zeros(self.action_space.shape)
        u_lin = self.action_transformer.preprocess(action_lin)
        batch_jac, batch_next_feature = self.predictor.feature_jacobian(batch_image, np.array([u_lin] * batch_size), preprocessed=True)
        batch_J = np.concatenate(batch_jac, axis=1)
        batch_y_next_pred = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_next_feature], axis=1)

        batch_z = batch_y_target - batch_y_next_pred + batch_J.dot(u_lin)
        cs_repeats = np.cumsum(np.r_[0, self.repeats])
        slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
        batch_A_split = np.array([np.einsum('nij,nik->njk', batch_J[:, s], batch_J[:, s]) for s in slices])
        batch_b_split = np.array([np.einsum('nij,ni->nj', batch_J[:, s], batch_z[:, s]) for s in slices])

        batch_A = np.tensordot(batch_A_split, self.w / self.repeats, axes=(0, 0)) + np.diag(self.lambda_)
        batch_b = np.tensordot(batch_b_split, self.w / self.repeats, axes=(0, 0))
        batch_u = np.linalg.solve(batch_A, batch_b)

        actions = np.array([self.action_transformer.deprocess(u) for u in batch_u])
        for action in actions:
            self.action_space.clip(action, out=action)
        return actions

    def objective(self, obs, action, preprocessed=False):
        """
        The following should be true if the predictor of the next feature is linear
        objectives = [self.objective(state, action) for (state, action) in zip(states, actions)]
        linearized_objectives = [self.linearized_objective(state, action) for (state, action) in zip(states, actions)]
        assert np.allclose(objectives, linearized_objectives)
        """
        if self.w.shape != (len(self.repeats),):
            raise NotImplementedError
        assert isinstance(obs, tuple)
        assert all(isinstance(o, tuple) for o in obs)
        obs, target_obs = obs
        image = obs[0]
        target_image = target_obs[0]

        features = self.predictor.feature(np.array([image, target_image]), preprocessed=preprocessed)
        y, y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in features], axis=1)
        if self.alpha != 1.0:
            y_target = self.alpha * y_target + (1 - self.alpha) * y

        # unlike in the other methods, the next feature in here uses the action action
        next_feature = self.predictor.next_feature(image, action, preprocessed=preprocessed)
        y_next_pred = np.concatenate([f.flatten() for f in next_feature])

        if preprocessed:
            u = action
        else:
            u = self.action_transformer.preprocess(action)

        value = np.repeat(self.w / self.repeats, self.repeats).dot((y_target - y_next_pred) ** 2) + self.lambda_.dot(u ** 2)
        return value

    def linearized_objective(self, obs, action, action_lin=None, preprocessed=False, with_constant=True):
        """
        The following should be true if the predictor of the next feature is linear
        objectives = [self.objective(state, action) for (state, action) in zip(states, actions)]
        linearized_objectives = [self.linearized_objective(state, action) for (state, action) in zip(states, actions)]
        assert np.allclose(objectives, linearized_objectives)
        """
        assert isinstance(obs, tuple)
        assert all(isinstance(o, tuple) for o in obs)
        obs, target_obs = obs
        image = obs[0]
        target_image = target_obs[0]

        features = self.predictor.feature(np.array([image, target_image]), preprocessed=preprocessed)
        y, y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in features], axis=1)
        if self.alpha != 1.0:
            y_target = self.alpha * y_target + (1 - self.alpha) * y

        if action_lin is None:
            action_lin = np.zeros(self.predictor.input_shapes[1])  # original units
        if preprocessed:
            u_lin = action_lin
        else:
            u_lin = self.action_transformer.preprocess(action_lin)
        jac, next_feature = self.predictor.feature_jacobian(image, action_lin, preprocessed=preprocessed)  # Jacobian is in preprocessed units
        J = np.concatenate(jac)
        y_next_pred = np.concatenate([f.flatten() for f in next_feature])

        if self.w is None:
            WJ = J
        elif self.w.shape == (len(self.repeats),):
            WJ = J * np.repeat(self.w / self.repeats, self.repeats)[:, None]
        elif self.w.shape == (J.shape[0],):
            WJ = J * self.w[:, None]
        elif self.w.shape == (J.shape[0], J.shape[0]):
            WJ = self.w.dot(J)
        elif self.w.ndim == 2 and self.w.shape[0] == J.shape[0]:
            WJ = self.w.dot(self.w.T.dot(J))
        else:
            raise ValueError('invalid weights w, %r' % self.w)

        z = y_target - y_next_pred + J.dot(self.action_transformer.preprocess(action_lin))
        A = WJ.T.dot(J) + np.diag(self.lambda_)
        b = WJ.T.dot(z)
        if preprocessed:
            u = action
        else:
            u = self.action_transformer.preprocess(action)
        value = u.T.dot(A.dot(u)) - 2 * b.dot(u)
        if with_constant:
            value += z.T.dot(np.repeat(self.w / self.repeats, self.repeats) * z)
        return value

    def act(self, obs, action_lin=None):
        """
        images and actions are in preprocessed units
        u is in processed units
        """
        assert isinstance(obs, tuple)
        assert all(isinstance(o, tuple) for o in obs)
        obs, target_obs = obs
        image = obs[0]
        target_image = target_obs[0]

        features = self.predictor.feature(np.array([image, target_image]))
        y, y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in features], axis=1)
        if self.alpha != 1.0:
            y_target = self.alpha * y_target + (1 - self.alpha) * y

        # use model to optimize for action
        if action_lin is None:
            action_lin = np.zeros(self.predictor.input_shapes[1])  # original units
        jac, next_feature = self.predictor.feature_jacobian(image, action_lin)  # Jacobian is in preprocessed units
        J = np.concatenate(jac)
        y_next_pred = np.concatenate([f.flatten() for f in next_feature])

        if self.w is None:
            WJ = J
        elif self.w.shape == (len(self.repeats),):
            WJ = J * np.repeat(self.w / self.repeats, self.repeats)[:, None]
        elif self.w.shape == (J.shape[0],):
            WJ = J * self.w[:, None]
        elif self.w.shape == (J.shape[0], J.shape[0]):
            WJ = self.w.dot(J)
        elif self.w.ndim == 2 and self.w.shape[0] == J.shape[0]:
            WJ = self.w.dot(self.w.T.dot(J))
        else:
            raise ValueError('invalid weights w, %r' % self.w)

        if self.use_constrained_opt:
            import cvxpy
            A = WJ.T.dot(J) + np.diag(self.lambda_)
            b = WJ.T.dot(y_target - y_next_pred + J.dot(self.action_transformer.preprocess(action_lin)))

            x = cvxpy.Variable(self.action_space.shape[0])
            objective = cvxpy.Minimize((1. / 2.) * cvxpy.quad_form(x, A) - x.T * b)

            action_low = self.action_transformer.preprocess(np.array(self.action_space.low))
            action_high = self.action_transformer.preprocess(np.array(self.action_space.high))
            if isinstance(self.action_space, (spaces.AxisAngleSpace, spaces.TranslationAxisAngleSpace)) and \
                    self.action_space.axis is None:
                assert action_low[-1] ** 2 == action_high[-1] ** 2
                contraints = [cvxpy.sum_squares(x[-3:]) <= action_high ** 2]
                if isinstance(self.action_space, spaces.TranslationAxisAngleSpace):
                    contraints.extend([action_low[:3] <= x[:3], x[:3] <= action_high[:3]])
            else:
                constraints = [action_low <= x, x <= action_high]
            prob = cvxpy.Problem(objective, constraints)

            solved = False
            for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
                try:
                    prob.solve(solver=solver)
                except cvxpy.error.SolverError:
                    continue
                if x.value is None:
                    continue
                solved = True
                break
            if not solved:
                import IPython as ipy; ipy.embed()
            u = np.squeeze(np.array(x.value), axis=1)
        else:
            u = np.linalg.solve(WJ.T.dot(J) + np.diag(self.lambda_),
                                WJ.T.dot(y_target - y_next_pred +
                                         J.dot(self.action_transformer.preprocess(action_lin))))  # preprocessed units

        action = self.action_transformer.deprocess(u)
        if self.use_constrained_opt:
            try:
                assert self.action_space.contains((1 - 1e-6) * action)  # allow for rounding errors
            except AssertionError:
                import IPython as ipy; ipy.embed()
        else:
            self.action_space.clip(action, out=action)
        return action

    # @property
    # def image_target(self):
    #     return self._image_target.copy()
    #
    # @image_target.setter
    # def image_target(self, image):
    #     self._image_target = image.copy()
    #     self._y_target = np.concatenate([f.flatten() for f in self.predictor.feature(self._image_target)])
    #
    # @property
    # def y_target(self):
    #     return self._y_target.copy()
    #
    # @y_target.setter
    # def y_target(self, y):
    #     raise
    #
    # def set_image_target(self, image_target):
    #     self.image_target = image_target
    #
    # def set_target(self, target_obs):
    #     self.image_target = target_obs[0]

    def _get_config(self):
        config = super(ServoingPolicy, self)._get_config()
        config.update({'predictor': self.predictor,
                       'alpha': self.alpha,
                       'lambda_': self.lambda_.tolist() if isinstance(self.lambda_, np.ndarray) else self.lambda_,
                       'w': self.w.tolist() if isinstance(self.w, np.ndarray) else self.w,
                       'use_constrained_opt': self.use_constrained_opt})
        return config


class TheanoServoingPolicy(ServoingPolicy):
    def __init__(self, *args, **kwargs):
        super(TheanoServoingPolicy, self).__init__(*args, **kwargs)
        self.phi_fn = None
        self.pi_fn = None
        X_var, U_var = self.predictor.input_vars
        X_target_var = T.tensor4('x_target')
        U_lin_var = T.matrix('u_lin')
        alpha_var = theano.tensor.scalar(name='alpha')
        self.input_vars = [X_var, U_var, X_target_var, U_lin_var, alpha_var]
        w_var = T.vector('w')
        lambda_var = T.vector('lambda')
        self.param_vars = [w_var, lambda_var]
        if len(self.repeats) <= 256 * 3:
            self.max_batch_size = 100
        else:
            self.max_batch_size = (100 * 256 * 3) // len(self.repeats)
            print("Using maximum batch size of %d" % self.max_batch_size)
        # if len(self.repeats) <= 128 * 3:
        #     self.max_batch_size = 100
        # else:
        #     self.max_batch_size = (100 * 128 * 3) // len(self.repeats)
        #     print("Using maximum batch size of %d" % self.max_batch_size)
        assert self.max_batch_size > 0

    def _get_A_b_c_split_vars(self):
        mode = None
        if mode is None:
            try:
                if self.predictor.bilinear_type in ('group_convolution', 'channelwise_local'):
                    mode = 'linear2'
                elif self.predictor.bilinear_type == 'channelwise' or self.predictor.tf_nl_layer:
                    mode = 'linear'
            except AttributeError:
                pass

        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars

        # feature_vars = L.get_output([self.predictor.pred_layers[feature_name] for feature_name in self.predictor.feature_name], deterministic=True)
        # y_var = T.concatenate([T.flatten(feature_var, outdim=2) for feature_var in feature_vars], axis=1)
        # y_target_var = theano.clone(y_var, replace={X_var: X_target_var})
        # y_target_var = theano.ifelse.ifelse(T.eq(alpha_var, 1.0), y_target_var, alpha_var * y_target_var + (1 - alpha_var) * y_var)
        # jac_vars, next_feature_vars = self.predictor._get_batched_jacobian_var(self.predictor.next_feature_name,
        #                                                                        self.predictor.control_name,
        #                                                                        mode=mode)
        # J_var = T.concatenate(jac_vars, axis=1)
        # y_next_pred_var = T.concatenate([T.flatten(next_feature_var, outdim=2) for next_feature_var in next_feature_vars], axis=1)
        # J_var = theano.clone(J_var, replace={U_var: U_lin_var})
        # y_next_pred_var = theano.clone(y_next_pred_var, replace={U_var: U_lin_var})
        #
        # z_var = y_target_var - y_next_pred_var + T.batched_dot(J_var, U_lin_var)
        # cs_repeats = np.cumsum(np.r_[0, self.repeats])
        # # slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
        # # A_split_vars = [T.batched_dot(J_var[:, s, :].dimshuffle((0, 2, 1)), J_var[:, s, :]) for s in slices]
        # # b_split_vars = [T.batched_dot(J_var[:, s, :].dimshuffle((0, 2, 1)), z_var[:, s]) for s in slices]
        # A_split_var, _ = theano.scan(fn=lambda start, stop, J_var: T.batched_dot(J_var[:, start:stop, :].dimshuffle((0, 2, 1)), J_var[:, start:stop, :]),
        #                              sequences=[T.as_tensor(cs_repeats[:-1]), T.as_tensor(cs_repeats[1:])],
        #                              non_sequences=[J_var])
        # b_split_var, _ = theano.scan(fn=lambda start, stop, J_var, z_var: T.batched_dot(J_var[:, start:stop, :].dimshuffle((0, 2, 1)), z_var[:, start:stop]),
        #                              sequences=[T.as_tensor(cs_repeats[:-1]), T.as_tensor(cs_repeats[1:])],
        #                              non_sequences=[J_var, z_var])

        feature_vars = L.get_output([self.predictor.pred_layers[feature_name] for feature_name in self.predictor.feature_name], deterministic=True)
        y_vars = [T.flatten(feature_var, outdim=2) for feature_var in feature_vars]
        y_target_vars = [theano.clone(y_var, replace={X_var: X_target_var}) for y_var in y_vars]
        y_target_vars = [theano.ifelse.ifelse(T.eq(alpha_var, 1.0), y_target_var, alpha_var * y_target_var + (1 - alpha_var) * y_var)
                         for (y_var, y_target_var) in zip(y_vars, y_target_vars)]
        jac_vars, next_feature_vars = self.predictor._get_batched_jacobian_var(self.predictor.next_feature_name,
                                                                               self.predictor.control_name,
                                                                               mode=mode)
        y_next_pred_vars = [T.flatten(next_feature_var, outdim=2) for next_feature_var in next_feature_vars]
        jac_vars = [theano.clone(jac_var, replace={U_var: U_lin_var}) for jac_var in jac_vars]
        y_next_pred_vars = [theano.clone(y_next_pred_var, replace={U_var: U_lin_var}) for y_next_pred_var in y_next_pred_vars]

        z_vars = [y_target_var - y_next_pred_var + T.batched_dot(jac_var, U_lin_var)
                  for (y_target_var, y_next_pred_var, jac_var) in zip(y_target_vars, y_next_pred_vars, jac_vars)]

        feature_names = self.predictor.feature_name if isinstance(self.predictor.feature_name, list) else [self.predictor.feature_name]
        feature_shapes = L.get_output_shape([self.predictor.pred_layers[name] for name in feature_names])

        A_split_vars = []
        b_split_vars = []
        c_split_vars = []
        for jac_var, z_var, feature_shape in zip(jac_vars, z_vars, feature_shapes):
            jac_split_vars = T.split(jac_var, [np.prod(feature_shape[2:])] * feature_shape[1], feature_shape[1], axis=1)
            z_split_vars = T.split(z_var, [np.prod(feature_shape[2:])] * feature_shape[1], feature_shape[1], axis=1)
            A_split_var, _ = theano.scan(fn=lambda i, jac_split_vars: T.batched_dot(jac_split_vars[i].dimshuffle((0, 2, 1)), jac_split_vars[i]),
                                         sequences=[T.arange(len(jac_split_vars))],
                                         non_sequences=[T.as_tensor(jac_split_vars)])
            b_split_var, _ = theano.scan(fn=lambda i, jac_split_vars, z_split_vars: T.batched_dot(jac_split_vars[i].dimshuffle((0, 2, 1)), z_split_vars[i]),
                                         sequences=[T.arange(len(jac_split_vars))],
                                         non_sequences=[T.as_tensor(jac_split_vars), T.as_tensor(z_split_vars)])
            c_split_var, _ = theano.scan(fn=lambda i, z_split_vars: T.batched_dot(z_split_vars[i], z_split_vars[i]),
                                         sequences=[T.arange(len(z_split_vars))],
                                         non_sequences=[T.as_tensor(z_split_vars)])
            A_split_vars.append(A_split_var)
            b_split_vars.append(b_split_var)
            c_split_vars.append(c_split_var)
        A_split_var = T.concatenate(A_split_vars)
        b_split_var = T.concatenate(b_split_vars)
        c_split_var = T.concatenate(c_split_vars)

        # X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars
        # X = np.random.random((100, 3, 32, 32))
        # U = np.random.random((100, 4))
        # X_target = np.random.random((100, 3, 32, 32))
        # U_lin = np.zeros((100, 4))
        # alpha = 1.0
        # A_b_split_vars_fn2 = theano.function([X_var, U_var, X_target_var, U_lin_var, alpha_var], [A_split_var, b_split_var, c_split_var], on_unused_input='warn', allow_input_downcast=True)
        # A_split2, b_split2 = A_b_split_vars_fn2(X, U, X_target, U_lin, alpha)
        # A_b_c_split_vars_fn3 = theano.function([X_var, U_var, X_target_var, U_lin_var, alpha_var], [A_split_var, b_split_var, c_split_var], on_unused_input='warn', allow_input_downcast=True)
        # A_split3, b_split3, c_split3 = A_b_c_split_vars_fn3(X, U, X_target, U_lin, alpha)

        # # the following is equivalent as above except that it raises recursion limit exception because the graph is very big
        # A_split_vars = []
        # b_split_vars = []
        # c_split_vars = []
        # for jac_var, z_var, feature_shape in zip(jac_vars, z_vars, feature_shapes):
        #     jac_split_vars = T.split(jac_var, [np.prod(feature_shape[2:])] * feature_shape[1], feature_shape[1], axis=1)
        #     z_split_vars = T.split(z_var, [np.prod(feature_shape[2:])] * feature_shape[1], feature_shape[1], axis=1)
        #     for jac_split_var, z_split_var in zip(jac_split_vars, z_split_vars):
        #         jac_transpose_split_var = jac_split_var.dimshuffle((0, 2, 1))
        #         A_split_var = T.batched_dot(jac_transpose_split_var, jac_split_var)[None, ...]
        #         b_split_var = T.batched_dot(jac_transpose_split_var, z_split_var)[None, ...]
        #         c_split_var = T.batched_dot(z_split_var, z_split_var)[None, ...]
        #         A_split_vars.append(A_split_var)
        #         b_split_vars.append(b_split_var)
        #         c_split_vars.append(c_split_var)
        # A_split_var = T.concatenate(A_split_vars)
        # b_split_var = T.concatenate(b_split_vars)
        # c_split_var = T.concatenate(c_split_vars)

        # A_b_c_split_vars_fn = theano.function([X_var, U_var, X_target_var, U_lin_var, alpha_var], [A_split_var, b_split_var, c_split_var], on_unused_input='warn', allow_input_downcast=True)
        # A_split, b_split, c_split = A_b_c_split_vars_fn(X, U, X_target, U_lin, alpha)

        return A_split_var, b_split_var, c_split_var

    def _get_A_b_split_vars(self):
        pass

    def _get_phi_var(self):
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars
        A_split_var, b_split_var, c_split_var = self._get_A_b_c_split_vars()

        phi_errors_var = (T.batched_dot(T.batched_tensordot(A_split_var.dimshuffle((1, 0, 2, 3)), U_var, axes=(3, 1)), U_var)
                          - 2 * T.batched_dot(b_split_var.dimshuffle((1, 0, 2)), U_var)
                          + c_split_var.T)
        phi_actions_var = U_var ** 2
        phi_var = T.concatenate([phi_errors_var / self.repeats, phi_actions_var], axis=1)
        return phi_var

    def _compile_phi_fn(self):
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars
        phi_var = self._get_phi_var()
        start_time = time.time()
        print("Compiling phi function...")
        phi_fn = theano.function([X_var, U_var, X_target_var, U_lin_var, alpha_var], phi_var, on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))
        return phi_fn

    def _get_pi_var(self):
        w_var, lambda_var = self.param_vars
        A_split_var, b_split_var, _ = self._get_A_b_c_split_vars()

        A_var = T.tensordot(A_split_var, w_var / self.repeats, axes=(0, 0)) + T.diag(lambda_var)
        B_var = T.tensordot(b_split_var, w_var / self.repeats, axes=(0, 0))
        pi_var = T.batched_dot(T.nlinalg.matrix_inverse(A_var), B_var)  # preprocessed units
        return pi_var

    def _compile_pi_fn(self):
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars
        w_var, lambda_var = self.param_vars
        pi_var = self._get_pi_var()
        start_time = time.time()
        print("Compiling pi function...")
        pi_fn = theano.function([X_var, X_target_var, U_lin_var, alpha_var, w_var, lambda_var], pi_var, on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))
        return pi_fn

    def phi(self, states, actions, preprocessed=False):
        """
        Corresponds to the linearized objective

        The following should be true
        phi = self.phi(states, actions)
        theta = np.append(self.w, self.lambda_)
        linearized_objectives = [self.linearized_objective(state, action, with_constant=False) for (state, action) in zip(states, actions)]
        objectives = phi.dot(theta)
        assert np.allclose(objectives, linearized_objectives)
        """
        batch_size = len(states)
        if batch_size <= self.max_batch_size:
            if preprocessed:
                batch_image = [obs[0] for (obs, target_obs) in states]
                batch_target_image = [target_obs[0] for (obs, target_obs) in states]
                batch_u = np.array(actions)
            else:
                batch_image, = self.predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
                batch_target_image, = self.predictor.preprocess([target_obs[0] for (obs, target_obs) in states], batch_size=len(states))
                batch_u = np.array([self.action_transformer.preprocess(action) for action in actions])
            action_lin = np.zeros(self.action_space.shape)
            u_lin = self.action_transformer.preprocess(action_lin)
            batch_u_lin = np.array([u_lin] * batch_size)

            if self.phi_fn is None:
                self.phi_fn = self._compile_phi_fn()
            phi = self.phi_fn(batch_image, batch_u, batch_target_image, batch_u_lin, self.alpha)
            return phi
        else:
            phi = None
            for i in range(0, batch_size, self.max_batch_size):
                s = slice(i, min(i + self.max_batch_size, batch_size))
                minibatch_phi = self.phi(states[s], actions[s], preprocessed=preprocessed)
                if phi is None:
                    phi = np.empty((batch_size,) + minibatch_phi.shape[1:])
                phi[s] = minibatch_phi
            return phi

    def pi(self, states, preprocessed=False):
        """
        Corresponds to the linearized objective

        The following should be true
        actions_pi = self.pi(states)
        actions_act = [self.act(state) for state in states]
        assert np.allclose(actions_pi, actions_act)
        """
        if self.w.shape != (len(self.repeats),):
            raise NotImplementedError
        batch_size = len(states)
        if batch_size <= self.max_batch_size:
            if preprocessed:
                batch_image = [obs[0] for (obs, target_obs) in states]
                batch_target_image = [target_obs[0] for (obs, target_obs) in states]
            else:
                batch_image, = self.predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
                batch_target_image, = self.predictor.preprocess([target_obs[0] for (obs, target_obs) in states], batch_size=len(states))
            action_lin = np.zeros(self.action_space.shape)
            u_lin = self.action_transformer.preprocess(action_lin)
            batch_u_lin = np.array([u_lin] * batch_size)

            if self.pi_fn is None:
                self.pi_fn = self._compile_pi_fn()
            batch_u = self.pi_fn(batch_image, batch_target_image, batch_u_lin, self.alpha, self.w, self.lambda_)

            actions = np.array([self.action_transformer.deprocess(u) for u in batch_u])
            for action in actions:
                self.action_space.clip(action, out=action)
            if preprocessed:
                return np.array([self.action_transformer.preprocess(action) for action in actions])
            else:
                return actions
        else:
            actions = None
            for i in range(0, batch_size, self.max_batch_size):
                s = slice(i, min(i + self.max_batch_size, batch_size))
                minibatch_actions = self.pi(states[s], preprocessed=preprocessed)
                if actions is None:
                    actions = np.empty((batch_size,) + minibatch_actions.shape[1:])
                actions[s] = minibatch_actions
            return actions
