from __future__ import division, print_function

import time

import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T
import yaml

from visual_dynamics.policies import Policy
from visual_dynamics.spaces import AxisAngleSpace
from visual_dynamics.spaces import TranslationAxisAngleSpace
from visual_dynamics.utils import iter_util
from visual_dynamics.utils.config import Python2to3Loader
from visual_dynamics.utils.config import from_config, from_yaml


class ServoingPolicy(Policy):
    def __init__(self, predictor, alpha=1.0, lambda_=0.0, w=1.0, use_constrained_opt=False, unweighted_features=False, algorithm_or_fname=None):
        if isinstance(predictor, str):
            with open(predictor) as predictor_file:
                predictor = from_yaml(predictor_file)
        self.predictor = predictor
        self.action_transformer = self.predictor.transformers['u']
        self.action_space = from_config(self.predictor.environment_config['action_space'])
        self.alpha = alpha
        lambda_ = np.asarray(lambda_)
        if np.isscalar(lambda_) or lambda_.ndim == 0:
            lambda_ = lambda_ * np.ones(self.action_space.shape)  # numpy fails with augmented assigment
        assert lambda_.shape == self.action_space.shape
        self._lambda_ = lambda_
        feature_names = iter_util.flatten_tree(self.predictor.feature_name)
        feature_shapes = L.get_output_shape([self.predictor.pred_layers[name] for name in feature_names])
        self.repeats = []
        for feature_shape in feature_shapes:
            self.repeats.extend([np.prod(feature_shape[2:])] * feature_shape[1])
        w = np.asarray(w)
        if np.isscalar(w) or w.ndim == 0 or len(w) == 1:
            w = w * np.ones(len(self.repeats))  # numpy fails with augmented assigment
        elif w.shape == (len(feature_names),):
            w = np.repeat(w, [feature_shape[1] for feature_shape in feature_shapes])
        assert w.shape == (len(self.repeats),)
        self._w = w
        self._theta = np.append(self._w, self._lambda_)
        self._w, self._lambda_ = np.split(self._theta, [len(self._w)])  # alias the parameters
        self.use_constrained_opt = use_constrained_opt
        self.unweighted_features = unweighted_features
        self.image_name = 'image'
        self.target_image_name = 'target_image'

        if algorithm_or_fname is not None:
            from visual_dynamics.algorithms import ServoingFittedQIterationAlgorithm
            if isinstance(algorithm_or_fname, str):
                with open(algorithm_or_fname) as algorithm_file:
                    algorithm_config = yaml.load(algorithm_file, Loader=Python2to3Loader)
                assert issubclass(algorithm_config['class'], ServoingFittedQIterationAlgorithm)
                mean_returns = algorithm_config['mean_returns']
                thetas = algorithm_config['thetas']
            else:
                algorithm = algorithm_or_fname
                assert isinstance(algorithm, ServoingFittedQIterationAlgorithm)
                mean_returns = algorithm.mean_returns
                thetas = algorithm.thetas
            print("using parameters based on best returns")
            best_return, best_theta = max(zip(mean_returns, thetas))
            print(best_return)
            print(best_theta)
            self.theta = best_theta

    @property
    def theta(self):
        assert all(self._theta == np.append(self._w, self._lambda_))
        if self.unweighted_features:
            assert all(self._w == self._w[0])
            theta = np.append(self.w[0], self.lambda_)
        else:
            theta = self._theta
        return theta

    @theta.setter
    def theta(self, theta):
        assert all(self._theta == np.append(self._w, self._lambda_))
        if self.unweighted_features:
            self.w = theta[0]
            self.lambda_ = theta[1:]
            assert all(self._w == self._w[0])
        else:
            self._theta[...] = theta

    @property
    def w(self):
        assert all(self._theta == np.append(self._w, self._lambda_))
        if self.unweighted_features:
            assert all(self._w == self._w[0])
        return self._w

    @w.setter
    def w(self, w):
        assert all(self._theta == np.append(self._w, self._lambda_))
        self._w[...] = w
        if self.unweighted_features:
            assert all(self._w == self._w[0])

    @property
    def lambda_(self):
        assert all(self._theta == np.append(self._w, self._lambda_))
        return self._lambda_

    @lambda_.setter
    def lambda_(self, lambda_):
        assert all(self._theta == np.append(self._w, self._lambda_))
        self._lambda_[...] = lambda_

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
        assert isinstance(obs, dict)
        image = obs[self.image_name]
        target_image = obs[self.target_image_name]

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
        assert isinstance(obs, dict)
        image = obs[self.image_name]
        target_image = obs[self.target_image_name]

        features = self.predictor.feature(np.array([image, target_image]), preprocessed=preprocessed)
        y, y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in features], axis=1)
        if self.alpha != 1.0:
            y_target = self.alpha * y_target + (1 - self.alpha) * y

        if action_lin is None:
            action_lin = np.zeros(self.predictor.input_shapes[1])  # original units
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
        assert isinstance(obs, dict)
        image = obs[self.image_name]
        target_image = obs[self.target_image_name]

        features = self.predictor.feature([np.array([image, target_image])])
        y, y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in features], axis=1)
        if self.alpha != 1.0:
            y_target = self.alpha * y_target + (1 - self.alpha) * y

        if action_lin is None:
            action_lin = np.zeros(self.predictor.input_shapes[1])  # original units
        jac, next_feature = self.predictor.feature_jacobian([image, action_lin])  # Jacobian is in preprocessed units
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
            if isinstance(self.action_space, (AxisAngleSpace, TranslationAxisAngleSpace)) and \
                    self.action_space.axis is None:
                assert action_low[-1] ** 2 == action_high[-1] ** 2
                contraints = [cvxpy.sum_squares(x[-3:]) <= action_high ** 2]
                if isinstance(self.action_space, TranslationAxisAngleSpace):
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
            try:
                u = np.linalg.solve(WJ.T.dot(J) + np.diag(self.lambda_),
                                    WJ.T.dot(y_target - y_next_pred +
                                             J.dot(self.action_transformer.preprocess(action_lin))))  # preprocessed units
            except np.linalg.LinAlgError as e:
                print("Got linear algebra error. Returning zero action")
                u = np.zeros(self.action_space.shape)

        action = self.action_transformer.deprocess(u)
        if self.use_constrained_opt:
            try:
                assert self.action_space.contains((1 - 1e-6) * action)  # allow for rounding errors
            except AssertionError:
                import IPython as ipy; ipy.embed()
        else:
            self.action_space.clip(action, out=action)
        return action

    def reset(self):
        return None

    def _get_config(self):
        config = super(ServoingPolicy, self)._get_config()
        config.update({'predictor': self.predictor,
                       'alpha': self.alpha,
                       'lambda_': self.lambda_.tolist(),
                       'w': self.w.tolist() if self.w is not None else self.w,
                       'use_constrained_opt': self.use_constrained_opt,
                       'unweighted_features': self.unweighted_features})
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
        if not self.predictor.feature_jacobian_name:
            raise NotImplementedError

        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars

        names = [self.predictor.feature_name, self.predictor.feature_jacobian_name, self.predictor.next_feature_name]
        vars_ = L.get_output([self.predictor.pred_layers[name] for name in iter_util.flatten_tree(names)], deterministic=True)
        feature_vars, jac_vars, next_feature_vars = iter_util.unflatten_tree(names, vars_)

        y_vars = [T.flatten(feature_var, outdim=2) for feature_var in feature_vars]
        y_target_vars = [theano.clone(y_var, replace={X_var: X_target_var}) for y_var in y_vars]
        y_target_vars = [theano.ifelse.ifelse(T.eq(alpha_var, 1.0),
                                              y_target_var,
                                              alpha_var * y_target_var + (1 - alpha_var) * y_var)
                         for (y_var, y_target_var) in zip(y_vars, y_target_vars)]

        jac_vars = [theano.clone(jac_var, replace={U_var: U_lin_var}) for jac_var in jac_vars]
        y_next_pred_vars = [T.flatten(next_feature_var, outdim=2) for next_feature_var in next_feature_vars]
        y_next_pred_vars = [theano.clone(y_next_pred_var, replace={U_var: U_lin_var}) for y_next_pred_var in y_next_pred_vars]

        z_vars = [y_target_var - y_next_pred_var + T.batched_tensordot(jac_var, U_lin_var, axes=(2, 1))
                  for (y_target_var, y_next_pred_var, jac_var) in zip(y_target_vars, y_next_pred_vars, jac_vars)]

        feature_shapes = L.get_output_shape([self.predictor.pred_layers[name] for name in iter_util.flatten_tree(self.predictor.feature_name)])

        A_split_vars = []
        b_split_vars = []
        c_split_vars = []
        for jac_var, z_var, feature_shape in zip(jac_vars, z_vars, feature_shapes):
            jac_split_vars = T.split(jac_var, [np.prod(feature_shape[2:])] * feature_shape[1], feature_shape[1], axis=1)
            z_split_vars = T.split(z_var, [np.prod(feature_shape[2:])] * feature_shape[1], feature_shape[1], axis=1)
            A_split_var, _ = theano.scan(fn=lambda i, jac_split_vars: T.batched_dot(jac_split_vars[i].dimshuffle((0, 2, 1)), jac_split_vars[i]),
                                         sequences=[T.arange(len(jac_split_vars))],
                                         non_sequences=[T.as_tensor(jac_split_vars)])
            b_split_var, _ = theano.scan(fn=lambda i, jac_split_vars, z_split_vars: T.batched_tensordot(jac_split_vars[i].dimshuffle((0, 2, 1)), z_split_vars[i], axes=(2, 1)),
                                         sequences=[T.arange(len(jac_split_vars))],
                                         non_sequences=[T.as_tensor(jac_split_vars), T.as_tensor(z_split_vars)])
            c_split_var, _ = theano.scan(fn=lambda i, z_split_vars: T.batched_tensordot(z_split_vars[i], z_split_vars[i], axes=(1, 1)),
                                         sequences=[T.arange(len(z_split_vars))],
                                         non_sequences=[T.as_tensor(z_split_vars)])
            A_split_vars.append(A_split_var)
            b_split_vars.append(b_split_var)
            c_split_vars.append(c_split_var)
        A_split_var = T.concatenate(A_split_vars)
        b_split_var = T.concatenate(b_split_vars)
        c_split_var = T.concatenate(c_split_vars)

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

        return A_split_var, b_split_var, c_split_var

    def _get_A_b_split_vars(self):
        pass

    def _get_phi_var(self):
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.input_vars
        A_split_var, b_split_var, c_split_var = self._get_A_b_c_split_vars()

        phi_errors_var = (T.batched_tensordot(T.batched_tensordot(A_split_var.dimshuffle((1, 0, 2, 3)), U_var, axes=(3, 1)), U_var, axes=(2, 1))
                          - 2 * T.batched_tensordot(b_split_var.dimshuffle((1, 0, 2)), U_var, axes=(2, 1))
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
        pi_var = T.batched_tensordot(T.nlinalg.matrix_inverse(A_var), B_var, axes=(2, 1))  # preprocessed units
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

    def phi(self, observations, actions, preprocessed=False):
        """
        Corresponds to the linearized objective

        The following should be true
        phi = self.phi(states, actions)
        theta = np.append(self.w, self.lambda_)
        linearized_objectives = [self.linearized_objective(state, action, with_constant=False) for (state, action) in zip(states, actions)]
        objectives = phi.dot(theta)
        assert np.allclose(objectives, linearized_objectives)
        """
        batch_size = len(observations)
        if batch_size <= self.max_batch_size:
            if preprocessed:
                batch_image = np.array([obs['image'] for obs in observations])
                batch_target_image = np.array([obs['target_image'] for obs in observations])
                batch_u = np.array(actions)
            else:
                batch_image, = self.predictor.preprocess([[obs['image'] for obs in observations]], batch_size)
                batch_target_image, = self.predictor.preprocess([[obs['target_image'] for obs in observations]], batch_size)
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
                minibatch_phi = self.phi(observations[s], actions[s], preprocessed=preprocessed)
                if phi is None:
                    phi = np.empty((batch_size,) + minibatch_phi.shape[1:])
                phi[s] = minibatch_phi
            return phi

    def pi(self, observations, preprocessed=False):
        """
        Corresponds to the linearized objective

        The following should be true
        actions_pi = self.pi(states)
        actions_act = [self.act(state) for state in states]
        assert np.allclose(actions_pi, actions_act)
        """
        if self.w.shape != (len(self.repeats),):
            raise NotImplementedError
        batch_size = len(observations)
        if batch_size <= self.max_batch_size:
            if preprocessed:
                batch_image = np.array([obs['image'] for obs in observations])
                batch_target_image = np.array([obs['target_image'] for obs in observations])
            else:
                batch_image, = self.predictor.preprocess([[obs['image'] for obs in observations]], batch_size)
                batch_target_image, = self.predictor.preprocess([[obs['target_image'] for obs in observations]], batch_size)
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
                minibatch_actions = self.pi(observations[s], preprocessed=preprocessed)
                if actions is None:
                    actions = np.empty((batch_size,) + minibatch_actions.shape[1:])
                actions[s] = minibatch_actions
            return actions
