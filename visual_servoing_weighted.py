from __future__ import division, print_function
import argparse
import yaml
import numpy as np
import scipy.stats
import lasagne
import theano
import theano.tensor as T
import time
import cv2
import envs
import policy
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
from gui.loss_plotter import LossPlotter
import _tkinter


# modified from https://github.com/openai/gym/blob/master/examples/agents/cem.py
def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0, th_low=None, th_high=None):
    """
    Generic implementation of the cross-entropy method for minimizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size * elite_frac))
    if isinstance(initial_std, np.ndarray) and initial_std.shape == th_mean.shape:
        th_std = initial_std
    else:
        th_std = np.ones_like(th_mean) * initial_std
    use_truncnorm = th_low is not None or th_high is not None
    if use_truncnorm:
        th_low = -np.inf if th_low is None else th_low
        th_high = np.inf if th_high is None else th_high

    for _ in range(n_iter):
        if use_truncnorm:
            tn = scipy.stats.truncnorm((th_low - th_mean) / th_std, (th_high - th_mean) / th_std)
            ths = np.array([th_mean + dth for dth in th_std[None, :] * tn.rvs((batch_size, th_mean.size))])
            assert np.all(th_low <= ths) and np.all(ths <= th_high)
        else:
            ths = np.array([th_mean + dth for dth in th_std[None, :] * np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys': ys, 'theta_mean': th_mean, 'y_mean': ys.mean()}


A_b_split_fn = None
imitation_train_fn = None
w_prerelu_var = None
lambda_prerelu_var = None


def imitation_learning(states, actions, target_actions, servoing_pol, n_iter, learning_rate=0.001):
    # TODO: check that target actions are in bounds
    # TODO: check that theano function outputs actions taken
    w = servoing_pol.w
    lambda_ = servoing_pol.lambda_
    predictor = servoing_pol.predictor
    action_transformer = servoing_pol.action_transformer
    action_space = servoing_pol.action_space
    alpha = servoing_pol.alpha

    # batch_size = len(states)
    # batch_image = np.asarray([obs[0] for (obs, target_obs) in states])
    # batch_target_image = np.asarray([target_obs[0] for (obs, target_obs) in states])
    # batch_target_feature = predictor.feature(batch_target_image)
    # batch_y_target = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_target_feature], axis=1)
    # if alpha != 1.0:
    #     batch_feature = predictor.feature(batch_image)
    #     batch_y = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_feature], axis=1)
    #     batch_y_target = alpha * batch_y_target + (1 - alpha) * batch_y
    #
    # batch_u_lin = np.zeros((batch_size,) + action_space.shape)  # original units
    # batch_jac, batch_next_feature = predictor.feature_jacobian(batch_image, batch_u_lin)
    # batch_J = np.concatenate(batch_jac, axis=1)
    # batch_y_next_pred = np.concatenate([f.reshape((f.shape[0], -1)) for f in batch_next_feature], axis=1)



    # jac_vars, next_feature_vars = predictor._get_batched_jacobian_var(predictor.next_feature_name, predictor.control_name, mode=mode)
    # J_var = T.concatenate(jac_vars, axis=1)
    # y_next_pred_var = T.concatenate([T.flatten(next_feature_var, outdim=2) for next_feature_var in next_feature_vars], axis=1)
    #
    # feature_vars = L.get_output([predictor.pred_layers[feature_name] for feature_name in predictor.feature_name], deterministic=True)
    # y_var = T.concatenate([T.flatten(feature_var, outdim=2) for feature_var in feature_vars], axis=1)
    #
    # X_var, = [input_var for input_var in predictor.input_vars if input_var in theano.gof.graph.inputs([y_var])]
    # X_target_var = T.tensor4('x_target')
    # y_target_var = theano.clone(y_var, replace={X_var: X_target_var})
    #
    # z_var = y_target_var - y_next_pred_var  # TODO: J.dot(u_lin)
    #
    # cs_repeats = np.cumsum(np.r_[0, servoing_pol.repeats])
    # slices = [slice(start, stop) for (start, stop) in zip(cs_repeats[:-1], cs_repeats[1:])]
    # A_split_vars = [T.batched_dot(J_var[:, s, :].dimshuffle((0, 2, 1)), J_var[:, s, :]) for s in slices]
    # b_split_vars = [T.batched_dot(J_var[:, s, :].dimshuffle((0, 2, 1)), z_var[:, s]) for s in slices]

    batch_size = len(states)
    # batch_image, = predictor.preprocess([obs[0] for (obs, target_obs) in states], batch_size=len(states))
    batch_image = [obs[0] for (obs, target_obs) in states]
    # batch_target_image, = predictor.preprocess([target_obs[0] for (obs, target_obs) in states], batch_size=len(states))
    batch_target_image = [target_obs[0] for (obs, target_obs) in states]
    action_lin = np.zeros(action_space.shape)
    u_lin = action_transformer.preprocess(action_lin)
    batch_u_lin = np.array([u_lin] * batch_size)
    batch_target_u = np.array([action_transformer.preprocess(target_action) for target_action in target_actions])

    assert isinstance(servoing_pol, policy.TheanoServoingPolicy)
    X_var, U_var, X_target_var, U_lin_var, alpha_var = servoing_pol.input_vars
    A_split_var, b_split_var, _ = servoing_pol._get_A_b_c_split_vars()
    U_target_var = T.matrix('u_target')

    global A_b_split_fn
    if A_b_split_fn is None:
        start_time = time.time()
        print("Compiling A b split function...")
        A_b_split_fn = theano.function([X_var, X_target_var, U_lin_var, alpha_var],
                                       [A_split_var, b_split_var],
                                       on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))

    # batch_A_split, batch_b_split = A_b_split_fn(batch_image, batch_target_image, batch_u_lin, alpha)
    max_batch_size = 100
    batch_A_split = None
    batch_b_split = None
    for i in range(0, batch_size, max_batch_size):
        s = slice(i, min(i + max_batch_size, batch_size))
        minibatch_A_split, minibatch_b_split = A_b_split_fn(batch_image[s], batch_target_image[s], batch_u_lin[s], alpha)
        if batch_A_split is None:
            batch_A_split = np.empty((minibatch_A_split.shape[0], batch_size) + minibatch_A_split.shape[2:])
        if batch_b_split is None:
            batch_b_split = np.empty((minibatch_b_split.shape[0], batch_size) + minibatch_b_split.shape[2:])
        batch_A_split[:, s] = minibatch_A_split
        batch_b_split[:, s] = minibatch_b_split

    global imitation_train_fn, w_prerelu_var, lambda_prerelu_var
    if imitation_train_fn is None:
        w_prerelu_var = theano.shared(w.astype(theano.config.floatX))
        lambda_prerelu_var = theano.shared(lambda_.astype(theano.config.floatX))
        w_var = T.nnet.relu(w_prerelu_var)
        lambda_var = T.nnet.relu(lambda_prerelu_var)
        A_var = T.tensordot(A_split_var, w_var / servoing_pol.repeats, axes=(0, 0)) + T.diag(lambda_var)
        B_var = T.tensordot(b_split_var, w_var / servoing_pol.repeats, axes=(0, 0))
        pi_var = T.batched_dot(T.nlinalg.matrix_inverse(A_var), B_var)  # preprocessed units

        # training loss
        loss_var = ((pi_var - U_target_var) ** 2).mean(axis=0).sum() / 2.

        # training updates
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        updates = lasagne.updates.adam(loss_var, [w_prerelu_var, lambda_prerelu_var], learning_rate=learning_rate_var)

        start_time = time.time()
        print("Compiling imitation training function...")
        imitation_train_fn = theano.function([A_split_var, b_split_var, U_target_var, learning_rate_var], loss_var, updates=updates,
                                             on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))

    for iter_ in range(n_iter):
        train_loss = float(imitation_train_fn(batch_A_split, batch_b_split,
                                              batch_target_u, learning_rate))
        print("Iteration {} of {}".format(iter_, n_iter))
        print("    training loss = {:.6f}".format(train_loss))
    servoing_pol.w = np.maximum(w_prerelu_var.get_value(), 0)
    servoing_pol.lambda_ = np.maximum(lambda_prerelu_var.get_value(), 0)
    return train_loss


from utils import tic, toc


class ImitationLearningAlgorithm(object):
    def __init__(self, servoing_pol, n_iter, l2_reg=0.0, learning_rate=0.1, param_representation='softplus'):
        self.servoing_pol = servoing_pol
        self.n_iter = n_iter
        if l2_reg != 0.0:
            raise NotImplementedError
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.param_representation = param_representation
        self.w_unmapped_var = theano.shared(self.unmap_param_value(self.servoing_pol.w))
        self.lambda_unmapped_var = theano.shared(self.unmap_param_value(self.servoing_pol.lambda_))
        self.w_var = self.map_param_var(self.w_unmapped_var)
        self.lambda_var = self.map_param_var(self.lambda_unmapped_var)
        # if self.weight_representation == 'softplus':
        #     self.w_presoftplus_var = theano.shared(np.where(self.servoing_pol.w < 100,
        #                                                     np.log(np.exp(self.servoing_pol.w) - 1),
        #                                                     self.servoing_pol.w).astype(theano.config.floatX))
        #     self.lambda_presoftplus_var = theano.shared(np.where(self.servoing_pol.lambda_ < 100,
        #                                                          np.log(np.exp(self.servoing_pol.lambda_) - 1),
        #                                                          self.servoing_pol.lambda_).astype(theano.config.floatX))
        #     self.params = [self.w_presoftplus_var, self.lambda_presoftplus_var]
        #     self.w_var = T.switch(T.lt(self.w_presoftplus_var, 100),
        #                                       T.log(1 + T.exp(self.w_presoftplus_var)),
        #                                       self.w_presoftplus_var)
        #     self.lambda_var = T.switch(T.lt(self.lambda_presoftplus_var, 100),
        #                                            T.log(1 + T.exp(self.lambda_presoftplus_var)),
        #                                            self.lambda_presoftplus_var)
        # elif self.weight_representation == 'relu':
        #     self.w_prerelu_var = theano.shared(self.servoing_pol.w.astype(theano.config.floatX))
        #     self.lambda_prerelu_var = theano.shared(self.servoing_pol.lambda_.astype(theano.config.floatX))
        #     self.params = [self.w_prerelu_var, self.lambda_prerelu_var]
        #     self.w_var = T.nnet.relu(self.w_prerelu_var)
        #     self.lambda_var = T.nnet.relu(self.lambda_prerelu_var)
        # elif self.weight_representation is None:
        #     self.w_var = theano.shared(self.servoing_pol.w.astype(theano.config.floatX))
        #     self.lambda_var = theano.shared(self.servoing_pol.lambda_.astype(theano.config.floatX))
        #     self.params = [self.w_var, self.lambda_var]
        # else:
        #    ValueError("Unknown weight representation %r" % self.weight_representation)
        self.A_b_split_fn = None
        self.imitation_train_fn = None

    def unmap_param_value(self, mapped_param):
        if self.param_representation == 'softplus':
            param = np.where(mapped_param < 100, np.log(np.exp(np.maximum(mapped_param, 0.0)) - 1), mapped_param)
        elif self.param_representation == 'relu':
            param = np.maximum(mapped_param, 0.0)
        elif self.param_representation == 'identity':
            param = mapped_param
        else:
            raise ValueError("Unknown param representation %r" % self.param_representation)
        return param

    def map_param_value(self, param):
        if self.param_representation == 'softplus':
            mapped_param = np.where(param < 100, np.log(1 + np.exp(param)), param)
        elif self.param_representation == 'relu':
            mapped_param = np.maximum(param, 0.0)
        elif self.param_representation == 'identity':
            mapped_param = param
        else:
            raise ValueError("Unknown param representation %r" % self.param_representation)
        return mapped_param

    def map_param_var(self, param_var):
        if self.param_representation == 'softplus':
            mapped_param_var = T.switch(T.lt(param_var, 100), T.log(1 + T.exp(param_var)), param_var)
        elif self.param_representation == 'relu':
            mapped_param_var = T.nnet.relu(param_var)
        elif self.param_representation == 'identity':
            mapped_param_var = param_var
        else:
            raise ValueError("Unknown param representation %r" % self.param_representation)
        return mapped_param_var

    # def set_param_values(self, w, lambda_):
    #     if self.weight_representation == 'softplus':
    #         self.w_presoftplus_var.set_value(np.where(w < 100, np.log(np.exp(w) - 1), w).astype(theano.config.floatX))
    #         self.lambda_presoftplus_var.set_value(np.where(lambda_ < 100, np.log(np.exp(lambda_) - 1), lambda_).astype(theano.config.floatX))
    #     elif self.weight_representation == 'relu':
    #         self.w_prerelu_var.set_value(w.astype(theano.config.floatX))
    #         self.lambda_prerelu_var.set_value(lambda_.astype(theano.config.floatX))
    #     elif self.weight_representation is None:
    #         self.w_var.set_value(w.astype(theano.config.floatX))
    #         self.lambda_var.set_value(lambda_.astype(theano.config.floatX))
    #     else:
    #        ValueError("Unknown weight representation %r" % self.weight_representation)
    #
    # def get_param_values(self):
    #     if self.weight_representation == 'softplus':
    #         w = np.where(self.w_presoftplus_var.get_value() < 100,
    #                      np.log(1 + np.exp(self.w_presoftplus_var.get_value())),
    #                      self.w_presoftplus_var.get_value())
    #         lambda_ = np.where(self.lambda_presoftplus_var.get_value() < 100,
    #                            np.log(1 + np.exp(self.lambda_presoftplus_var.get_value())),
    #                            self.lambda_presoftplus_var.get_value())
    #     elif self.weight_representation == 'relu':
    #         w = np.maximum(self.w_prerelu_var.get_value(), 0)
    #         lambda_ = np.maximum(self.lambda_prerelu_var.get_value(), 0)
    #     elif self.weight_representation is None:
    #         w = self.w_var.get_value()
    #         lambda_ = self.lambda_var.get_value()
    #     else:
    #         ValueError("Unknown weight representation %r" % self.weight_representation)
    #     return w, lambda_

    def _compile_A_b_split_fn(self):
        assert isinstance(self.servoing_pol, policy.TheanoServoingPolicy)
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.servoing_pol.input_vars
        A_split_var, b_split_var, _ = self.servoing_pol._get_A_b_c_split_vars()

        start_time = time.time()
        print("Compiling A b split function...")
        A_b_split_fn = theano.function([X_var, X_target_var, U_lin_var, alpha_var],
                                       [A_split_var, b_split_var],
                                       on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))
        return A_b_split_fn

    def _compile_train_fn(self):
        assert isinstance(self.servoing_pol, policy.TheanoServoingPolicy)
        A_split_var, b_split_var, _ = self.servoing_pol._get_A_b_c_split_vars()
        U_target_var = T.matrix('u_target')

        A_var = T.tensordot(A_split_var, self.w_var / self.servoing_pol.repeats, axes=(0, 0)) + T.diag(self.lambda_var)
        B_var = T.tensordot(b_split_var, self.w_var / self.servoing_pol.repeats, axes=(0, 0))
        pi_var = T.batched_dot(T.nlinalg.matrix_inverse(A_var), B_var)  # preprocessed units

        # training loss
        loss_var = ((pi_var - U_target_var) ** 2).mean(axis=0).sum() / 2.

        # training updates
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        updates = lasagne.updates.adam(loss_var, [self.w_unmapped_var, self.lambda_unmapped_var], learning_rate=learning_rate_var)

        start_time = time.time()
        print("Compiling imitation training function...")
        imitation_train_fn = theano.function([A_split_var, b_split_var, U_target_var, learning_rate_var], loss_var, updates=updates,
                                             on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))
        return imitation_train_fn

    def update(self, states, target_actions):
        states = [state for traj_states in states for state in traj_states]
        target_actions = [target_action for traj_target_actions in target_actions for target_action in traj_target_actions]
        self.w_unmapped_var.set_value(self.unmap_param_value(self.servoing_pol.w))
        self.lambda_unmapped_var.set_value(self.unmap_param_value(self.servoing_pol.lambda_))

        if self.A_b_split_fn is None:
            self.A_b_split_fn = self._compile_A_b_split_fn()
        if self.imitation_train_fn is None:
            self.imitation_train_fn = self._compile_train_fn()

        batch_size = len(states)
        batch_image = [obs[0] for (obs, target_obs) in states]
        batch_target_image = [target_obs[0] for (obs, target_obs) in states]
        action_lin = np.zeros(self.servoing_pol.action_space.shape)
        u_lin = self.servoing_pol.action_transformer.preprocess(action_lin)
        batch_u_lin = np.array([u_lin] * batch_size)
        # batch_target_u = np.array([action_transformer.preprocess(target_action) for target_action in target_actions])
        batch_target_u = np.array(target_actions)

        max_batch_size = 100
        batch_A_split = None
        batch_b_split = None
        for i in range(0, batch_size, max_batch_size):
            s = slice(i, min(i + max_batch_size, batch_size))
            minibatch_A_split, minibatch_b_split = \
                self.A_b_split_fn(batch_image[s], batch_target_image[s], batch_u_lin[s], self.servoing_pol.alpha)
            if batch_A_split is None:
                batch_A_split = np.empty((minibatch_A_split.shape[0], batch_size) + minibatch_A_split.shape[2:])
            if batch_b_split is None:
                batch_b_split = np.empty((minibatch_b_split.shape[0], batch_size) + minibatch_b_split.shape[2:])
            batch_A_split[:, s] = minibatch_A_split
            batch_b_split[:, s] = minibatch_b_split

        for iter_ in range(self.n_iter):
            train_loss = float(self.imitation_train_fn(batch_A_split, batch_b_split, batch_target_u, self.learning_rate))
            print("Iteration {} of {}".format(iter_, self.n_iter))
            print("    training loss = {:.6f}".format(train_loss))
            if iter_ % 100 == 0:
                theta = np.concatenate([self.w_var.eval(), self.lambda_var.eval()])
                print("theta\n\t%r" % theta)
        self.servoing_pol.w = self.w_var.eval()
        self.servoing_pol.lambda_ = self.lambda_var.eval()
        return train_loss


class MonteCarloAlgorithm(object):
    def __init__(self, servoing_pol, gamma, l2_reg=0.0):
        self.servoing_pol = servoing_pol
        self.gamma = gamma
        if l2_reg != 0.0:
            raise NotImplementedError
        self.l2_reg = l2_reg

    def update(self, states, actions, rewards):
        S = [traj_states[0] for traj_states in states]
        A = np.array([traj_actions[0] for traj_actions in actions])
        batch_size = len(S)
        phi = np.c_[self.servoing_pol.phi(S, A, preprocessed=True), np.ones(batch_size)]

        Q_sample = np.array([np.dot(traj_rewards, self.gamma ** np.arange(len(traj_rewards))) for traj_rewards in rewards])

        # theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, 0.0]
        # self.servoing_pol.bias = (Q_sample - phi.dot(theta)).mean(axis=0)
        theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]

        lsq_A = phi
        lsq_b = Q_sample

        print("theta\n\t%r" % theta)
        objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
        print("    bellman error = {:.6f}".format(objective_value))

        import cvxpy
        theta_var = cvxpy.Variable(theta.shape[0])
        objective = cvxpy.Minimize(
            (1 / 2.) * cvxpy.sum_squares((lsq_A / np.sqrt(len(lsq_A))) * theta_var - (lsq_b / np.sqrt(len(lsq_A)))) +
            (self.l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
        constraints = [0 <= theta_var[:-1]]  # no constraint on bias

        prob = cvxpy.Problem(objective, constraints)
        solved = False
        for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
            try:
                prob.solve(solver=solver)
            except cvxpy.error.SolverError:
                continue
            if theta_var.value is None:
                continue
            solved = True
            break
        if not solved:
            import IPython as ipy;
            ipy.embed()
        theta = np.squeeze(np.array(theta_var.value), axis=1)

        print("theta\n\t%r" % theta)
        objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
        print("    bellman error = {:.6f}".format(objective_value))

        w, lambda_, bias = np.split(theta, np.cumsum([len(self.servoing_pol.w), len(self.servoing_pol.lambda_)]))
        self.servoing_pol.w = w
        self.servoing_pol.lambda_ = lambda_
        self.servoing_pol.bias, = bias
        return objective_value


class DqnAlgorithm(object):
    def __init__(self, servoing_pol, n_iter, gamma, l2_reg=0.0, learning_rate=0.01, max_batch_size=1000, max_memory_size=0):
        self.servoing_pol = servoing_pol
        self.n_iter = n_iter
        self.gamma = gamma
        if l2_reg != 0.0:
            raise NotImplementedError
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.max_batch_size = max_batch_size
        self.max_memory_size = max_memory_size
        self.memory_sars = [[] for _ in range(4)]
        # self.w_var = theano.shared(self.servoing_pol.w.astype(theano.config.floatX))
        # self.lambda_var = theano.shared(self.servoing_pol.lambda_.astype(theano.config.floatX))
        self.w_presoftplus_var = theano.shared(np.log(np.exp(self.servoing_pol.w) - 1).astype(theano.config.floatX))
        self.lambda_presoftplus_var = theano.shared(np.log(np.exp(self.servoing_pol.lambda_) - 1).astype(theano.config.floatX))
        self.bias_var = theano.shared(np.array(self.servoing_pol.bias).astype(theano.config.floatX))
        self.dqn_train_fn = None

    def _compile_train_fn(self):
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.servoing_pol.input_vars
        X_next_var = T.tensor4('x_next')
        X_next_target_var = T.tensor4('x_next_target')
        R_var = T.vector('R')

        w_var = T.log(1 + T.exp(self.w_presoftplus_var))
        lambda_var = T.log(1 + T.exp(self.lambda_presoftplus_var))
        bias_var = self.bias_var
        theta_var = T.concatenate([w_var, lambda_var, [bias_var]])

        # depends on X_var, X_target_var and U_var
        phi_var = self.servoing_pol._get_phi_var()
        phi_var = T.concatenate([phi_var, T.ones((phi_var.shape[0], 1))], axis=1)

        # depends on X_next_var and X_next_target_var
        pi_var = self.servoing_pol._get_pi_var().astype(theano.config.floatX)
        pi_p_var = theano.clone(pi_var, replace=dict(zip([X_var, X_target_var] + self.servoing_pol.param_vars,
                                                         [X_next_var, X_next_target_var, w_var, lambda_var])))
        # pi_p_fn = theano.function([X_var, X_target_var, U_lin_var, alpha_var], pi_p_var,
        #                           on_unused_input='warn', allow_input_downcast=True)
        # pi_p2 = pi_p_fn(batch_next_image, batch_next_target_image, batch_u_lin, self.servoing_pol.alpha)
        phi_p_var = theano.clone(phi_var, replace={X_var: X_next_var, X_target_var: X_next_target_var, U_var: pi_p_var})
        V_p_var = T.dot(phi_p_var, theta_var)
        Q_sample_var = R_var + self.gamma * V_p_var
        # Q_sample_fn = theano.function([X_var, X_target_var, U_lin_var, alpha_var], Q_sample_var,
        #                               on_unused_input='warn', allow_input_downcast=True)
        # Q_sample2 = Q_sample_fn(batch_next_image, batch_next_target_image, batch_u_lin, self.servoing_pol.alpha)

        # training loss
        loss_var = ((T.dot(phi_var, theta_var) - Q_sample_var) ** 2).mean(axis=0).sum() / 2.

        # training updates
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        updates = lasagne.updates.adam(loss_var, [self.w_presoftplus_var, self.lambda_presoftplus_var, self.bias_var],
                                       learning_rate=learning_rate_var)

        start_time = time.time()
        print("Compiling DQN training function...")
        dqn_train_fn = theano.function([X_var, X_target_var,
                                        U_var, R_var,
                                        X_next_var, X_next_target_var,
                                        U_lin_var, alpha_var, learning_rate_var], loss_var,
                                       updates=updates,
                                       on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))
        return dqn_train_fn

    def update(self, *sars):
        sars = [[step_data for traj_data in data for step_data in traj_data] for data in sars]
        assert len(sars) == 4
        if self.max_memory_size:
            for memory_data, data in zip(self.memory_sars, sars):
                memory_data.extend(data)
                del memory_data[:(len(memory_data) - self.max_memory_size)]
            sars = self.memory_sars

        batch_size = min(len(sars[0]), self.max_batch_size)
        self.w_presoftplus_var.set_value(np.log(np.exp(self.servoing_pol.w) - 1).astype(theano.config.floatX))
        self.lambda_presoftplus_var.set_value(np.log(np.exp(self.servoing_pol.lambda_) - 1).astype(theano.config.floatX))
        self.bias_var.set_value(np.array(self.servoing_pol.bias).astype(theano.config.floatX))

        if self.dqn_train_fn is None:
            self.dqn_train_fn = self._compile_train_fn()

        for iter_ in range(self.n_iter):
            if len(sars[0]) > self.max_batch_size:
                choice = np.random.choice(len(sars[0]), self.max_batch_size)
                S, A, R, S_p = [[data[i] for i in choice] for data in sars]
            elif iter_ == 0:  # the data is fixed across iterations so only set at the first iteration
                S, A, R, S_p = sars

            # compute phi only if (S, A) has changed
            if len(sars[0]) > self.max_batch_size or iter_ == 0:
                assert len(S) == batch_size
                batch_image = [obs[0] for (obs, target_obs) in S]
                batch_target_image = [target_obs[0] for (obs, target_obs) in S]
                batch_actions = np.asarray(A)
                batch_rewards = np.asarray(R)
                batch_next_image = [obs[0] for (obs, target_obs) in S_p]
                batch_next_target_image = [target_obs[0] for (obs, target_obs) in S_p]
                action_lin = np.zeros(self.servoing_pol.action_space.shape)
                u_lin = self.servoing_pol.action_transformer.preprocess(action_lin)
                batch_u_lin = np.array([u_lin] * batch_size)

            train_loss = float(self.dqn_train_fn(batch_image, batch_target_image,
                                                 batch_actions, batch_rewards,
                                                 batch_next_image, batch_next_target_image,
                                                 batch_u_lin, self.servoing_pol.alpha, self.learning_rate))
            print("Iteration {} of {}".format(iter_, self.n_iter))
            print("    training loss = {:.6f}".format(train_loss))
            if iter_ % 20 == 0:
                theta = np.r_[np.log(1 + np.exp(self.w_presoftplus_var.get_value())),
                              np.log(1 + np.exp(self.lambda_presoftplus_var.get_value())),
                              [self.bias_var.get_value()]]
                print("theta\n\t%r" % theta)
        self.servoing_pol.w = np.log(1 + np.exp(self.w_presoftplus_var.get_value()))
        self.servoing_pol.lambda_ = np.log(1 + np.exp(self.lambda_presoftplus_var.get_value()))
        self.servoing_pol.bias = self.bias_var.get_value()
        return train_loss


class FittedQIterationAlgorithm(object):
    def __init__(self, servoing_pol, n_iter, gamma, l2_reg=0.0, eps=1.0,
                 use_variable_bias=True, fit_free_params=False, constrain_theta=False,
                 max_batch_size=1000, max_memory_size=0):
        self.servoing_pol = servoing_pol
        self.n_iter = n_iter
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.eps = eps
        self.use_variable_bias = use_variable_bias
        self.fit_free_params = fit_free_params
        self.constrain_theta = constrain_theta
        self.max_batch_size = max_batch_size
        self.max_memory_size = max_memory_size
        self.memory_sars = [[] for _ in range(4)]

    def update(self, *sars):
        # TODO: check that target actions are in bounds
        # TODO: check that theano function outputs actions taken
        sars = [[step_data for traj_data in data for step_data in traj_data] for data in sars]
        assert len(sars) == 4
        if self.max_memory_size:
            for memory_data, data in zip(self.memory_sars, sars):
                memory_data.extend(data)
                del memory_data[:(len(memory_data) - self.max_memory_size)]
            sars = self.memory_sars

        batch_size = min(len(sars[0]), self.max_batch_size)
        theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        thetas = [theta.copy()]
        bellman_errors = []
        proxy_bellman_errors = []

        # import IPython as ipy; ipy.embed()
        #
        # servoing_pol2 = policy.ServoingPolicy(self.servoing_pol.predictor,
        #                                       alpha=self.servoing_pol.alpha,
        #                                       lambda_=self.servoing_pol.lambda_,
        #                                       w=self.servoing_pol.w)
        # S, A, R, S_p = sars
        # A = np.asarray(A)
        # R = np.asarray(R)
        #
        # phi = self.servoing_pol.phi(S, A, preprocessed=True)
        # phi2 = servoing_pol2.phi(S, A, preprocessed=True)
        # assert np.allclose(phi, phi2)
        #
        # states, actions = S, A
        # phi = servoing_pol2.phi(states, actions, preprocessed=True, with_constant=True)
        # theta = np.append(servoing_pol2.w, servoing_pol2.lambda_)
        # linearized_objectives = [servoing_pol2.linearized_objective(state, action, preprocessed=True, with_constant=True) for (state, action) in zip(states, actions)]
        # nonlinearized_objectives = [servoing_pol2.objective(state, action, preprocessed=True) for (state, action) in zip(states, actions)]
        # objectives = phi.dot(theta)
        # assert np.allclose(objectives, linearized_objectives)
        # assert np.allclose(objectives, nonlinearized_objectives)

        # import IPython as ipy; ipy.embed()
        # theta_init = theta
        #
        # servoing_pol.w = np.array([5.921114686349468, 2.3245162817195593, 2.3458439929810195, 5.566302291968323, 2.265380216663454,
        #      2.28574557354553, 4.591682192249294, 2.1187573493370406, 2.137875263507064])
        # servoing_pol.lambda_ = np.array([0.9999999999696686, 0.7374846591657251, 2.114078372188159, 2.0670524314948473])
        # servoing_pol.bias = np.array(19.740879972099144)
        #
        # servoing_pol.w = np.ones_like(servoing_pol.w)
        # servoing_pol.lambda_ = np.ones_like(servoing_pol.lambda_)
        # servoing_pol.bias = np.array(0.0)
        # theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        #
        # servoing_pol.w = np.array([22.446248, 16.38214429, 20.08670612, 12.22685585, 17.8761917, 15.69352661, 17.6225469, 8.0052753, 21.75278794])
        # servoing_pol.lambda_ = np.array([13.04235679, 4.80352296, 15.26906728, 11.25787556])
        # servoing_pol.w /= servoing_pol.lambda_[0]
        # servoing_pol.lambda_ /= servoing_pol.lambda_[0]
        # servoing_pol.bias = np.array(0.0)
        # theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        # tic()
        # if self.gamma == 0:
        #     Q_sample = R
        # else:
        #     A_p = self.servoing_pol.pi(S_p, preprocessed=True)
        #     phi_p = np.c_[self.servoing_pol.phi(S_p, A_p, preprocessed=True), np.ones(batch_size)]
        #     V_p = phi_p.dot(theta)
        #     Q_sample = R + self.gamma * V_p
        # toc("Q_sample")
        # servoing_pol.bias = (Q_sample - phi.dot(theta)).mean(axis=0) / (1 - self.gamma)
        # theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        #
        # servoing_pol.w = np.array([5.921114686349468, 2.3245162817195593, 2.3458439929810195, 5.566302291968323, 2.265380216663454,
        #      2.28574557354553, 4.591682192249294, 2.1187573493370406, 2.137875263507064])
        # servoing_pol.lambda_ = np.array([0.9999999999696686, 0.7374846591657251, 2.114078372188159, 2.0670524314948473])
        # servoing_pol.bias = np.array(0.0)
        # theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        # tic()
        # if self.gamma == 0:
        #     Q_sample = R
        # else:
        #     A_p = self.servoing_pol.pi(S_p, preprocessed=True)
        #     phi_p = np.c_[self.servoing_pol.phi(S_p, A_p, preprocessed=True), np.ones(batch_size)]
        #     V_p = phi_p.dot(theta)
        #     Q_sample = R + self.gamma * V_p
        # toc("Q_sample")
        # servoing_pol.bias = (Q_sample - phi.dot(theta)).mean(axis=0) / (1 - self.gamma)
        # theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        #
        #
        # tic()
        # if self.gamma == 0:
        #     Q_sample = R
        # else:
        #     A_p = self.servoing_pol.pi(S_p, preprocessed=True)
        #     phi_p = np.c_[self.servoing_pol.phi(S_p, A_p, preprocessed=True), np.ones(batch_size)]
        #     V_p = phi_p.dot(theta)
        #     Q_sample = R + self.gamma * V_p
        # toc("Q_sample")
        # fig = plt.figure()
        # pt0 = min(phi.dot(theta).min(), Q_sample.min())
        # pt1 = max(phi.dot(theta).max(), Q_sample.max())
        # plt.plot([pt0, pt1], [pt0, pt1])
        # plt.axis('equal')
        # plt.scatter(phi.dot(theta), Q_sample)
        # plt.show(block=False)

        for iter_ in range(self.n_iter):
            if len(sars[0]) > self.max_batch_size and iter_ == 0:  # TODO: only first iteration?
                choice = np.random.choice(len(sars[0]), self.max_batch_size)
                S, A, R, S_p = [[data[i] for i in choice] for data in sars]
            elif iter_ == 0:  # the data is fixed across iterations so only set at the first iteration
                S, A, R, S_p = sars

            # compute phi only if (S, A) has changed
            if (len(sars[0]) > self.max_batch_size and iter_ == 0) or iter_ == 0:  # TODO: only first iteration?
                assert len(S) == batch_size
                A = np.asarray(A)
                R = np.asarray(R)
                tic()
                phi = np.c_[self.servoing_pol.phi(S, A, preprocessed=True), np.ones(batch_size)]
                toc("phi")

            # compute Q_sample
            tic()
            if self.gamma == 0:
                Q_sample = R
            else:
                A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                phi_p = np.c_[self.servoing_pol.phi(S_p, A_p, preprocessed=True)
                , np.ones(batch_size)]
                V_p = phi_p.dot(theta)
                Q_sample = R + self.gamma * V_p
            toc("Q_sample")

            # if iter_ == 0 and self.fit_free_params:
            if self.fit_free_params:  # TODO: every iteration?
                old_objective_value = (1 / 2.) * ((phi.dot(theta) - Q_sample) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
                A = np.c_[phi[:, :-1].dot(theta[:-1]) - self.gamma * phi_p[:, :-1].dot(theta[:-1]), (1 - self.gamma) * np.ones(batch_size)]
                L = np.diag([batch_size * self.l2_reg * theta[:-1].dot(theta[:-1]), 0])

                lsq_A_fit = A.T.dot(A) + L
                lsq_b_fit = A.T.dot(R)
                alpha, bias = np.linalg.solve(lsq_A_fit, lsq_b_fit)
                if alpha < 0:
                    print("Unconstrained alpha is negative. Solving constrained optimization.")
                    import cvxpy
                    x_var = cvxpy.Variable(2)
                    objective = cvxpy.Minimize((1 / 2.) * cvxpy.sum_squares(lsq_A_fit * x_var - lsq_b_fit))
                    constraints = [0.0 < x_var[0]]
                    prob = cvxpy.Problem(objective, constraints)
                    solved = False
                    for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
                        try:
                            prob.solve(solver=solver)
                        except cvxpy.error.SolverError:
                            continue
                        if x_var.value is None:
                            continue
                        solved = True
                        break
                    if solved:
                        alpha, bias = np.squeeze(np.array(x_var.value))
                    else:
                        print("Unable to solve constrained optimization. Setting alpha = 0 solving for bias.")
                        alpha = 0.0
                        # bias = (R + self.gamma * phi_p[:, :-1].dot(theta[:-1]) - phi[:, :-1].dot(theta[:-1])).mean() / (1 - self.gamma)
                        bias = R.mean() / (1 - self.gamma)
                theta[:-1] *= alpha
                theta[-1] = bias
                V_p = phi_p.dot(theta)
                Q_sample = R + self.gamma * V_p
                new_objective_value = (1 / 2.) * ((phi.dot(theta) - Q_sample) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
                if new_objective_value > old_objective_value:
                    print("Objective increased from %.6f to %.6f after fitting alpha and bias." % (old_objective_value, new_objective_value))

            # if iter_ == 0 and self.fit_first_bias:
            #     old_objective_value = (1 / 2.) * ((phi.dot(theta) - Q_sample) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
            #     bias_inc = (Q_sample - phi.dot(theta)).mean(axis=0) / (1 - self.gamma)
            #     theta[-1] += bias_inc
            #     Q_sample += self.gamma * bias_inc
            #     new_objective_value2 = (1 / 2.) * ((phi.dot(theta) - Q_sample) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
            #     assert new_objective_value <= old_objective_value

            tic()
            lsq_A = phi
            lsq_b = Q_sample

            objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
            print("Iteration {} of {}".format(iter_, self.n_iter))
            print("    bellman error = {:.6f}".format(objective_value))
            bellman_errors.append(objective_value)

            import cvxpy
            theta_var = cvxpy.Variable(theta.shape[0])
            if self.use_variable_bias:
                objective = cvxpy.Minimize(
                    (1 / 2.) * cvxpy.sum_squares((lsq_A / np.sqrt(len(lsq_A))) * theta_var
                                                 - ((lsq_b + (theta_var[-1] - theta[-1]) * self.gamma) / np.sqrt(len(lsq_A)))) +
                    (self.l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
            else:
                objective = cvxpy.Minimize(
                    (1 / 2.) * cvxpy.sum_squares((lsq_A / np.sqrt(len(lsq_A))) * theta_var - (lsq_b / np.sqrt(len(lsq_A)))) +
                    (self.l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
            constraints = [0 <= theta_var[:-1]]  # no constraint on bias
            if self.constrain_theta:
                constraints.append(theta_var[len(self.servoing_pol.w)] == theta[len(self.servoing_pol.w)])
            # constraints.append(theta_var[-1] == theta[-1])

            if self.eps is not None:
                constraints.append(cvxpy.sum_squares(theta_var[:-1] - theta[:-1]) <= ((len(theta) - 1.) * self.eps))
                # constraints.append(cvxpy.sum_squares(theta_var - theta) <= self.eps)
                # constraints.append(cvxpy.sum_squares(theta_var[:-1] - theta[:-1]) <= self.eps)

            prob = cvxpy.Problem(objective, constraints)
            solved = False
            for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
                try:
                    prob.solve(solver=solver)
                except cvxpy.error.SolverError:
                    continue
                if theta_var.value is None:
                    continue
                solved = True
                break
            if not solved:
                import IPython as ipy;
                ipy.embed()
            theta = np.squeeze(np.array(theta_var.value), axis=1)
            thetas.append(theta)
            # print("    bellman error = {:.6f}".format(objective.value))
            objective_increased = objective.value > objective_value
            print(u"                  {} {:.6f}".format(u"\u2191" if objective_increased else u"\u2193", objective.value))
            proxy_bellman_errors.append(objective.value)
            toc("cvxpy")

            print("theta\n\t%r" % theta)
            w, lambda_, bias = np.split(theta, np.cumsum([len(self.servoing_pol.w), len(self.servoing_pol.lambda_)]))
            self.servoing_pol.w = w
            self.servoing_pol.lambda_ = lambda_
            self.servoing_pol.bias, = bias

        # compute objective value using all the data used in this update (which might include data from the memory)
        if len(sars[0]) > self.max_batch_size and iter_ == 0:  # TODO: only first iteration?
        # if len(sars[0]) > self.max_batch_size:
            S, A, R, S_p = sars
            batch_size = len(S)
            A = np.asarray(A)
            R = np.asarray(R)
            tic()
            phi = np.c_[self.servoing_pol.phi(S, A, preprocessed=True), np.ones(batch_size)]
            toc("phi")
        select_best_theta = False
        if select_best_theta:
            tic()
            if self.gamma == 0:
                Q_sample = R
                Q_samples = [Q_sample] * len(thetas)
            else:
                Q_samples = []
                for theta in thetas:
                    w, lambda_, bias = np.split(theta, np.cumsum([len(self.servoing_pol.w), len(self.servoing_pol.lambda_)]))
                    self.servoing_pol.w = w
                    self.servoing_pol.lambda_ = lambda_
                    self.servoing_pol.bias = bias
                    A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                    phi_p = np.c_[self.servoing_pol.phi(S_p, A_p, preprocessed=True), np.ones(batch_size)]
                    V_p = phi_p.dot(theta)
                    Q_sample = R + self.gamma * V_p
                    Q_samples.append(Q_sample)
            toc("Q_samples")
            objective_values = []
            for theta, Q_sample in zip(thetas, Q_samples):
                lsq_A = phi
                lsq_b = Q_sample
                objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
                objective_values.append(objective_value)
            objective_value = objective_values[np.argmin(objective_values)]
            theta = thetas[np.argmin(objective_values)]
            print("theta\n\t%r" % theta)
            w, lambda_, bias = np.split(theta, np.cumsum([len(self.servoing_pol.w), len(self.servoing_pol.lambda_)]))
            self.servoing_pol.w = w
            self.servoing_pol.lambda_ = lambda_
            self.servoing_pol.bias = bias
        else:
            tic()
            if self.gamma == 0:
                Q_sample = R
            else:
                A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                phi_p = np.c_[self.servoing_pol.phi(S_p, A_p, preprocessed=True), np.ones(batch_size)]
                V_p = phi_p.dot(theta)
                Q_sample = R + self.gamma * V_p
            toc("Q_sample")
            lsq_A = phi
            lsq_b = Q_sample
            objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (self.l2_reg / 2.) * (theta[:-1] ** 2).sum()
        return objective_value, thetas, bellman_errors + [objective_value]


def fitted_q_iteration(states, actions, rewards, next_states, servoing_pol, n_iter, gamma, l2_reg=0.0, eps=1.0, first_unconstrained=False):
    # TODO: check that target actions are in bounds
    # TODO: check that theano function outputs actions taken
    batch_size = len(states)
    S = states
    A = np.asarray(actions)
    R = np.asarray(rewards)
    S_p = next_states

    from utils import tic, toc
    tic()
    phi = np.c_[servoing_pol.phi(S, A), np.ones(batch_size)]
    toc("phi")

    # import IPython as ipy; ipy.embed()
    # eps = 1000.0
    # servoing_pol.w[:] = 1.0
    # servoing_pol.lambda_[:] = 1.0
    # servoing_pol.bias[:] = 0.0

    # theta = np.r_[servoing_pol.w, servoing_pol.lambda_, servoing_pol.bias]
    # thetas = [theta.copy()]
    # bellman_errors = []
    # proxy_bellman_errors = []
    # for iter_ in range(n_iter):
    #     tic()
    #     if gamma == 0:
    #         Q_sample = R
    #     else:
    #         A_p = servoing_pol.pi(S_p)
    #         phi_p = np.c_[servoing_pol.phi(S_p, A_p), np.ones(batch_size)]
    #         V_p = phi_p.dot(theta)
    #         Q_sample = rewards + gamma * V_p
    #     toc("Q_sample")
    #
    #     tic()
    #     lsq_A = phi
    #     lsq_b = Q_sample
    #
    #     objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (l2_reg / 2.) * (theta[:-1] ** 2).sum()
    #     print("Iteration {} of {}".format(iter_, n_iter))
    #     print("    bellman error = {:.6f}".format(objective_value))
    #     bellman_errors.append(objective_value)
    #
    #     if len(bellman_errors) >= 2 and bellman_errors[-1] > bellman_errors[-2]:
    #         thetas.pop()
    #         bellman_errors.pop()
    #         proxy_bellman_errors.pop()
    #         theta = thetas[-1]
    #         objective_value = bellman_errors[-1]
    #         eps *= 0.5
    #         w, lambda_, bias = np.split(theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))
    #         servoing_pol.w = w
    #         servoing_pol.lambda_ = lambda_
    #         servoing_pol.bias = bias
    #         if eps < 1e-3:
    #             break
    #
    #     import cvxpy
    #     theta_var = cvxpy.Variable(theta.shape[0])
    #     # objective = cvxpy.Minimize((1 / (2. * len(lsq_A))) * cvxpy.sum_squares(lsq_A * theta_var - lsq_b) +
    #     #                            (l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
    #     objective = cvxpy.Minimize((1 / 2.) * cvxpy.sum_squares((lsq_A / np.sqrt(len(lsq_A))) * theta_var - (lsq_b / np.sqrt(len(lsq_A)))) +
    #                                (l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
    #     constraints = [0 <= theta_var[:-1]]  # no constraint on bias
    #     if eps is not None and (not first_unconstrained or (first_unconstrained and iter_ > 0)):
    #         constraints.append(cvxpy.sum_squares(theta_var - theta) <= eps)
    #
    #     prob = cvxpy.Problem(objective, constraints)
    #
    #     solved = False
    #     for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
    #         try:
    #             prob.solve(solver=solver)
    #         except cvxpy.error.SolverError:
    #             continue
    #         if theta_var.value is None:
    #             continue
    #         solved = True
    #         break
    #     if not solved:
    #         import IPython as ipy;
    #         ipy.embed()
    #     theta = np.squeeze(np.array(theta_var.value), axis=1)
    #     thetas.append(theta)
    #     # print("    bellman error = {:.6f}".format(objective.value))
    #     objective_increased = objective.value > objective_value
    #     print(u"                  {} {:.6f}".format(u"\u2191" if objective_increased else u"\u2193", objective.value))
    #     proxy_bellman_errors.append(objective.value)
    #     toc("cvxpy")
    #
    #     # converged = np.allclose(new_theta, theta, atol=1e-3)
    #     # theta = new_theta
    #
    #     print("theta\n\t%r" % theta)
    #     w, lambda_, bias = np.split(theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))
    #     servoing_pol.w = w
    #     servoing_pol.lambda_ = lambda_
    #     servoing_pol.bias = bias
    #
    # tic()
    # if gamma == 0:
    #     Q_sample = R
    # else:
    #     A_p = servoing_pol.pi(S_p)
    #     phi_p = np.c_[servoing_pol.phi(S_p, A_p), np.ones(batch_size)]
    #     V_p = phi_p.dot(theta)
    #     Q_sample = rewards + gamma * V_p
    # toc("Q_sample")
    #
    # tic()
    # lsq_A = phi
    # lsq_b = Q_sample
    #
    # objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (l2_reg / 2.) * (theta[:-1] ** 2).sum()
    # return objective_value


    theta = np.r_[servoing_pol.w, servoing_pol.lambda_, [servoing_pol.bias]]
    thetas = [theta.copy()]
    bellman_errors = []
    proxy_bellman_errors = []
    for iter_ in range(n_iter):
        tic()
        if gamma == 0:
            Q_sample = R
        else:
            A_p = servoing_pol.pi(S_p)
            phi_p = np.c_[servoing_pol.phi(S_p, A_p), np.ones(batch_size)]
            V_p = phi_p.dot(theta)
            Q_sample = rewards + gamma * V_p
        toc("Q_sample")

        tic()
        lsq_A = phi
        lsq_b = Q_sample

        objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (l2_reg / 2.) * (theta[:-1] ** 2).sum()
        print("Iteration {} of {}".format(iter_, n_iter))
        print("    bellman error = {:.6f}".format(objective_value))
        bellman_errors.append(objective_value)

        import cvxpy
        theta_var = cvxpy.Variable(theta.shape[0])
        # objective = cvxpy.Minimize((1 / (2. * len(lsq_A))) * cvxpy.sum_squares(lsq_A * theta_var - lsq_b) +
        #                            (l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
        objective = cvxpy.Minimize((1 / 2.) * cvxpy.sum_squares((lsq_A / np.sqrt(len(lsq_A))) * theta_var - (lsq_b / np.sqrt(len(lsq_A)))) +
                                   (l2_reg / 2.) * cvxpy.sum_squares(theta_var[:-1]))  # no regularization on bias
        constraints = [0 <= theta_var[:-1]]  # no constraint on bias
        if eps is not None:
            constraints.append(cvxpy.sum_squares(theta_var - theta) <= eps)

        prob = cvxpy.Problem(objective, constraints)

        solved = False
        for solver in [None, cvxpy.GUROBI, cvxpy.CVXOPT]:
            try:
                prob.solve(solver=solver)
            except cvxpy.error.SolverError:
                continue
            if theta_var.value is None:
                continue
            solved = True
            break
        if not solved:
            import IPython as ipy;
            ipy.embed()
        theta = np.squeeze(np.array(theta_var.value), axis=1)
        thetas.append(theta)
        # print("    bellman error = {:.6f}".format(objective.value))
        objective_increased = objective.value > objective_value
        print(u"                  {} {:.6f}".format(u"\u2191" if objective_increased else u"\u2193", objective.value))
        proxy_bellman_errors.append(objective.value)
        toc("cvxpy")

        # converged = np.allclose(new_theta, theta, atol=1e-3)
        # theta = new_theta

        print("theta\n\t%r" % theta)
        w, lambda_, bias = np.split(theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))
        servoing_pol.w = w
        servoing_pol.lambda_ = lambda_
        servoing_pol.bias = bias

        # if converged:
        #     break
    # if converged:
    #     print('fitted Q-iteration converged in %d iterations' % (iter_ + 1))
    # else:
    #     print('fitted Q-iteration reached maximum number of iterations %d' % n_iter)

    tic()
    if gamma == 0:
        Q_sample = R
    else:
        A_p = servoing_pol.pi(S_p)
        phi_p = np.c_[servoing_pol.phi(S_p, A_p), np.ones(batch_size)]
        V_p = phi_p.dot(theta)
        Q_sample = rewards + gamma * V_p
    toc("Q_sample")

    tic()
    lsq_A = phi
    lsq_b = Q_sample

    objective_value = (1 / 2.) * ((lsq_A.dot(theta) - lsq_b) ** 2).mean() + (l2_reg / 2.) * (theta[:-1] ** 2).sum()
    return objective_value

    # print(servoing_pol.objective(states[0], actions[0]))
    # print(servoing_pol.linearized_objective(states[0], actions[0]))
    #
    # print(servoing_pol.pi(states)[0])
    # print(servoing_pol.act(states[0]))
    #
    # phi = servoing_pol.phi(states, actions)
    # theta = np.append(servoing_pol.w, servoing_pol.lambda_)
    # print(servoing_pol.linearized_objective(states[0], actions[0], with_constant=False))
    # print(phi.dot(theta)[0])


def rollout(predictor, env, pol, target_pol, num_trajs, num_steps, action_normal_scale=None, container=None,
            visualize=None, image_visualizer=None, record_file=None, writer=None,
            reward_type='image_formation', target_distance=0):
    # assert len(pol.policy) == 2
    # if isinstance(pol.policy[0], policy.MixedPolicy):
    #     target_pol, servoing_pol = pol.policy[0].policy
    # else:
    #     target_pol, servoing_pol = pol.policy
    # assert isinstance(target_pol, policy.TargetPolicy)
    # assert isinstance(servoing_pol, policy.ServoingPolicy)

    if target_distance:
        random_pol = policy.RandomPolicy(env.action_space, env.state_space)

    error_names = env.get_error_names()
    all_errors = []
    errors_header_format = "{:>30}" + "{:>15}" * (len(error_names) + 1)
    errors_row_format = "{:>30}" + "{:>15.4f}" * (len(error_names) + 1)
    print('=' * (30 + 15 * (len(error_names) + 1)))
    print(errors_header_format.format("(traj_iter, step_iter)", *(error_names + [reward_type or 'cost'])))
    done = False
    states = []
    actions = []
    rewards = []
    next_states = []
    if reward_type == 'target_action':
        target_actions = []
        action_transformer = predictor.transformers['u']
    for traj_iter in range(num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = pol.reset()
            env.reset(state)

            target_obs = env.observe()
            target_image = target_obs[0]
            # servoing_pol.set_image_target(target_image)

            if target_distance:
                reset_action = random_pol.act(obs=None)
                for _ in range(target_distance):
                    env.step(reset_action)

            if isinstance(env, envs.Pr2Env):
                import rospy
                rospy.sleep(1)

            if container:
                container.add_datum(traj_iter,
                                    **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], target_obs)))
            for step_iter in range(num_steps):
                state, obs = env.get_state_and_observe()
                image = obs[0]

                if reward_type == 'target_action':
                    target_action = target_pol.act(obs, tightness=1.0)
                    target_actions.append(target_action)

                action = pol.act((obs, target_obs))  # TODO: better way to pass target_obs?
                # if action_normal_scale is not None:
                #     action_noise = np.random.normal(loc=(env.action_space.low + env.action_space.high) / 2,
                #                                     scale=action_normal_scale * (env.action_space.high - env.action_space.low) / 2)
                #     if action.shape != action_noise.shape:
                #         import utils.transformations as tf
                #         import spaces
                #         assert isinstance(env.action_space, spaces.TranslationAxisAngleSpace)
                #         axis, _ = tf.split_axis_angle(action[3:])
                #         translation_noise, angle_noise = np.split(action_noise, [3])
                #         action_noise = np.r_[translation_noise, axis * angle_noise]
                #     action += action_noise
                env.step(action)  # action is updated in-place if needed

                # errors
                errors = env.get_errors(target_pol.get_target_state())
                all_errors.append(errors.values())
                if step_iter == (num_steps - 1):
                    next_state, next_obs = env.get_state_and_observe()
                    next_errors = env.get_errors(target_pol.get_target_state())
                    all_errors.append(next_errors.values())

                # states, actions, next_states, rewards
                states.append((obs, target_obs))
                actions.append(action)
                if step_iter > 0:
                    next_states.append((obs, target_obs))
                    if reward_type == 'errors':
                        reward = np.array(errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        reward = ((predictor.preprocess(image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        reward = ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        reward = np.array(errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        reward = ((action_transformer.preprocess(target_actions[-2]) - action_transformer.preprocess(actions[-2])) ** 2).sum()
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                    rewards.append(reward)
                    print(errors_row_format.format(str((traj_iter, step_iter - 1)), *(all_errors[-2] + [reward])))
                if step_iter == (num_steps - 1):
                    next_image = next_obs[0]
                    next_states.append((next_obs, target_obs))
                    if reward_type == 'errors':
                        next_reward = np.array(next_errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        next_reward = ((predictor.preprocess(next_image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        next_reward = ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        next_reward = np.array(next_errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        next_reward = ((action_transformer.preprocess(target_actions[-1]) - action_transformer.preprocess(actions[-1])) ** 2).sum()
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                    rewards.append(next_reward)
                    print(errors_row_format.format(str((traj_iter, step_iter)), *(all_errors[-1] + [next_reward])))

                # container
                if container:
                    container.add_datum(traj_iter, step_iter, state=state, action=action,
                                        reward=((action_transformer.preprocess(target_actions[-1]) - \
                                                 action_transformer.preprocess(actions[-1])) ** 2).sum(),
                                        **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                    if step_iter == (num_steps-1):
                        container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                            **dict(list(next_errors.items()) + list(zip(env.sensor_names, next_obs))))

                # visualization
                if visualize:
                    env.render()
                    next_obs = env.observe()
                    next_image = next_obs[0]
                    vis_images = [image, next_image, target_image]
                    for i, vis_image in enumerate(vis_images):
                        vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                    if visualize == 1:
                        vis_features = vis_images
                    else:
                        feature = predictor.feature(image)
                        feature_next_pred = predictor.next_feature(image, action)
                        feature_next = predictor.feature(next_image)
                        feature_target = predictor.feature(target_image)
                        # put all features into a flattened list
                        vis_features = [feature, feature_next_pred, feature_next, feature_target]
                        if not isinstance(predictor.feature_name, str):
                            vis_features = [vis_features[icol][irow] for irow in range(image_visualizer.rows - 1) for icol in range(image_visualizer.cols)]
                        vis_images.insert(2, None)
                        vis_features = vis_images + vis_features
                    # deprocess features if they have 3 channels (useful for RGB images)
                    for i, vis_feature in enumerate(vis_features):
                        if vis_feature is None:
                            continue
                        if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                            vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                    try:
                        image_visualizer.update(vis_features)
                        if record_file:
                            writer.grab_frame()
                    except _tkinter.TclError:  # TODO: is this the right exception?
                        done = True
                if done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    print('-' * (30 + 15 * (len(error_names) + 1)))
    rms_errors = np.sqrt(np.mean(np.square(all_errors), axis=0))
    rms_reward = np.sqrt(np.mean(np.square(rewards), axis=0))
    print(errors_row_format.format("RMS", *np.r_[rms_errors, rms_reward]))
    return all_errors, states, actions, next_states, rewards, target_actions


def do_rollout(predictor, env, reset_pol, act_pol, target_pol, num_steps,
               visualize=None, image_visualizer=None, record_file=None, writer=None,
               reward_type='image_formation', target_distance=0,
               quiet=False, ret_target_actions=True, reset_after_target=False, ret_tpv=False):
    if ret_tpv:
        import ogre
        tpv_camera_sensor = ogre.PyCameraSensor(env.app.camera, 640, 480)
        tpv_images = []
    if target_distance:
        random_pol = policy.RandomPolicy(env.action_space, env.state_space)

    all_errors = []
    if not quiet:
        error_names = env.get_error_names()
        errors_header_format = "{:>30}" + "{:>15}" * (len(error_names) + 1)
        errors_row_format = "{:>30}" + "{:>15.4f}" * (len(error_names) + 1)
        print('=' * (30 + 15 * (len(error_names) + 1)))
        print(errors_header_format.format("step_iter", *(error_names + [reward_type or 'cost'])))
    done = False
    states = []
    actions = []
    rewards = []
    next_states = []
    if reward_type == 'target_action' or ret_target_actions:
        target_actions = []
        action_transformer = predictor.transformers['u']
    if reward_type == 'image_formation':
        target_directions = []
        image_formation_errors = []
    try:
        state = reset_pol.reset()
    except AttributeError:
        state = reset_pol
    env.reset(state)

    target_obs = env.observe()
    target_image = target_obs[0]
    if reset_after_target:
        try:
            state = reset_pol.reset()
        except AttributeError:
            state = reset_pol
        env.reset(state)

    if target_distance:
        reset_action = random_pol.act(obs=None)
        target_distance_integer, target_distance_decimal = divmod(target_distance, 1)
        print(env.get_state())
        for _ in range(int(target_distance_integer)):
            env.step(reset_action)
            print(env.get_state())
        if target_distance_decimal:
            env.step(target_distance_decimal * reset_action)
        print(env.get_state())

    if isinstance(env, envs.Pr2Env):
        import rospy
        rospy.sleep(1)

    for step_iter in range(num_steps):
        try:
            state, obs = env.get_state_and_observe()
            image = obs[0]

            if reward_type == 'target_action' or ret_target_actions:
                target_action = target_pol.act((obs, target_obs))
                target_action = env.action_space.clip(target_action)
                pre_target_action = action_transformer.preprocess(target_action)
                target_actions.append(pre_target_action)

            action = act_pol.act((obs, target_obs))  # TODO: better way to pass target_obs?
            env.step(action)  # action is updated in-place if needed

            # image formation error
            if reward_type == 'image_formation':
                import utils.transformations as tf
                target_T = target_pol.target_node.getTransform()
                # agent transform in world coordinates
                agent_T = target_pol.agent_node.getTransform()
                # camera transform relative to the agent
                agent_to_camera_T = target_pol.camera_node.getTransform()
                # camera transform in world coordinates
                camera_T = agent_T.dot(agent_to_camera_T)
                # target relative to camera
                camera_to_target_T = tf.inverse_matrix(camera_T).dot(target_T)
                # target direction relative to camera
                target_direction = -camera_to_target_T[:3, 3]
                target_directions.append(target_direction)
                # x_error = np.arctan2(target_direction[0], target_direction[2])
                # y_error = np.arctan2(target_direction[1], target_direction[2])
                x_error = (target_direction[0] / target_direction[2])
                y_error = (target_direction[1] / target_direction[2])
                z_error = (1.0 / np.linalg.norm(target_direction) - 1.0 / np.linalg.norm(target_pol.offset))
                # image_formation_error = (np.array([x_error, y_error, z_error]) ** 2).sum()
                fov_y = np.pi / 4.
                height = 480
                f = height / (2. * np.tan(fov_y / 2.))
                image_formation_error = np.linalg.norm(f * np.array([x_error, y_error, z_error]))
                image_formation_errors.append(image_formation_error)

            # errors
            errors = env.get_errors(target_pol.get_target_state())
            all_errors.append(errors.values())
            if step_iter == (num_steps - 1):
                next_state, next_obs = env.get_state_and_observe()
                next_errors = env.get_errors(target_pol.get_target_state())
                all_errors.append(next_errors.values())

            # states, actions, next_states, rewards
            pre_obs, pre_action = zip(*[predictor.preprocess(o, action) for o in obs])
            pre_action, _ = pre_action
            pre_target_obs, = zip(*[predictor.preprocess(o) for o in target_obs])
            states.append((pre_obs, pre_target_obs))
            actions.append(pre_action)
            if step_iter > 0:
                next_states.append((pre_obs, pre_target_obs))
                # car is not in the preprocessed image or camera is too close to the car
                if np.all(predictor.preprocess(obs[1])[0] == predictor.preprocess(np.zeros_like(obs[1]))[0]) or \
                        np.linalg.norm(target_directions[-2]) < 4.0:
                    done = True
                    reward = (num_steps - step_iter + 1) * rewards[-1]  # high cost when the car gets out of the view
                else:
                    if reward_type == 'errors':
                        reward = np.array(errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        reward = ((predictor.preprocess(image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        reward = ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        reward = np.array(errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        # reward = ((action_transformer.preprocess(target_actions[-2]) - action_transformer.preprocess(actions[-2])) ** 2).sum()
                        reward = ((target_actions[-2] - actions[-2]) ** 2).sum()
                    elif reward_type == 'image_formation':
                        reward = image_formation_errors[-2]
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                rewards.append(reward)
                if not quiet:
                    print(errors_row_format.format(str(step_iter - 1), *(all_errors[-2] + [reward])))
            if step_iter == (num_steps - 1) and not done:
                next_image = next_obs[0]
                if np.all(predictor.preprocess(next_obs[1])[0] == predictor.preprocess(np.zeros_like(next_obs[1]))[0]) or \
                        np.linalg.norm(target_directions[1]) < 4.0:
                    done = True
                    next_reward = rewards[-1]
                else:
                    pre_next_obs = [predictor.preprocess(o)[0] for o in next_obs]
                    next_states.append((pre_next_obs, pre_target_obs))
                    if reward_type == 'errors':
                        next_reward = np.array(next_errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        next_reward = ((predictor.preprocess(next_image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        next_reward = ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        next_reward = np.array(next_errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        # next_reward = ((action_transformer.preprocess(target_actions[-1]) - action_transformer.preprocess(actions[-1])) ** 2).sum()
                        next_reward = ((target_actions[-1] - actions[-1]) ** 2).sum()
                    elif reward_type == 'image_formation':
                        next_reward = image_formation_errors[-1]
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                rewards.append(next_reward)
                if not quiet:
                    print(errors_row_format.format(str(step_iter), *(all_errors[-1] + [next_reward])))

            # visualization
            if visualize or ret_tpv:
                env.render()
            if ret_tpv:
                tpv_images.append(tpv_camera_sensor.observe().copy())
            if visualize:
                next_obs = env.observe()
                next_image = next_obs[0]
                vis_images = [image, next_image, target_image]
                for i, vis_image in enumerate(vis_images):
                    vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                if visualize == 1:
                    vis_features = vis_images
                else:
                    feature = predictor.feature(image)
                    feature_next_pred = predictor.next_feature(image, action)
                    feature_next = predictor.feature(next_image)
                    feature_target = predictor.feature(target_image)
                    # put all features into a flattened list
                    vis_features = [feature, feature_next, feature_next_pred, feature_target]
                    if not isinstance(predictor.feature_name, str):
                        vis_features = [vis_features[icol][irow] for irow in range(image_visualizer.rows - 1) for icol in range(image_visualizer.cols)]

                    # A = np.linalg.inv(WJ.T.dot(J) + np.diag(self.lambda_)).dot(WJ.T)
                    # # B = np.array([A[:, start:stop].sum(axis=1) for (start, stop) in zip(np.r_[0, np.cumsum(self.repeats)[:-1]], np.cumsum(self.repeats))]).T
                    # b = y_target - y_next_pred + J.dot(self.action_transformer.preprocess(action_lin))
                    # B = np.array([A[:, start:stop].dot(b[start:stop]) for (start, stop) in zip(np.r_[0, np.cumsum(self.repeats)[:-1]], np.cumsum(self.repeats))]).T
                    # scores = (B ** 2).sum(axis=0)
                    # # w = np.split(act_pol.w, np.cumsum([512] * 3)[:-1])
                    # inds  = [inds for s in np.split(scores, np.cumsum([512] * 3)[:-1]) for inds in [np.argsort(s)[::-1]] * 4]
                    # vis_features = [vis_feature[argsort_inds] for vis_feature, argsort_inds in zip(vis_features, inds)]

                    vis_images.insert(2, None)
                    vis_features = vis_images + vis_features
                # deprocess features if they have 3 channels (useful for RGB images)
                for i, vis_feature in enumerate(vis_features):
                    if vis_feature is None:
                        continue
                    if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                        vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                try:
                    image_visualizer.update(vis_features)
                    if record_file:
                        writer.grab_frame()
                except _tkinter.TclError:  # TODO: is this the right exception?
                    done = True
                # import matplotlib as mpl
                # import matplotlib.cm as cm
                # # images
                # cv2.imwrite('feature_images/image/image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cv2.imwrite('feature_images/image/image_next.jpg', cv2.cvtColor(next_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cv2.imwrite('feature_images/image/image_target.jpg', cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # # preprocessed images
                # cv2.imwrite('feature_images/x/image.jpg', cv2.cvtColor(predictor.transformers['x'].transformers[0].preprocess(image), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cv2.imwrite('feature_images/x/image_next.jpg', cv2.cvtColor(predictor.transformers['x'].transformers[0].preprocess(next_image), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # cv2.imwrite('feature_images/x/image_target.jpg', cv2.cvtColor(predictor.transformers['x'].transformers[0].preprocess(target_image), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # # features
                # for feature_name, f, f_next_pred, f_next, f_target in \
                #         zip(predictor.feature_name, feature, feature_next_pred, feature_next, feature_target):
                #     f_min = min(f.min(), f_next_pred.min(), f_next.min(), f_target.min())
                #     f_max = max(f.max(), f_next_pred.max(), f_next.max(), f_target.max())
                #     for cmap in ['jet', 'viridis']:
                #         m = cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=f_min, vmax=f_max), cmap=cmap)
                #         for c in range(len(f)):
                #             for f_, postfix_name in zip([f, f_next_pred, f_next, f_target], ['', '_next_pred', '_next', '_target']):
                #                 fname = 'feature_images/%s/%s_%03d.jpg' % (cmap, feature_name + postfix_name, c)
                #                 img = cv2.cvtColor(m.to_rgba(f_[c], bytes=True)[..., :3], cv2.COLOR_RGB2BGR)
                #                 cv2.imwrite(fname, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if done:
                break
        except KeyboardInterrupt:
            break
    if ret_target_actions:
        outputs = states, actions, rewards, next_states, target_actions
    else:
        outputs = states, actions, rewards, next_states
    min_len = min([len(output) for output in outputs])
    outputs = [output[:min_len] for output in outputs]

    if not quiet:
        print('-' * (30 + 15 * (len(error_names) + 1)))
        rms_errors = np.sqrt(np.mean(np.square(all_errors), axis=0))
        rms_reward = np.sqrt(np.mean(np.square(rewards)))
        print(errors_row_format.format("RMS", *np.r_[rms_errors, rms_reward]))
    if ret_tpv:
        tpv_images = tpv_images[:min_len]
        return outputs, tpv_images
    else:
        return outputs


def rollout_bak(predictor, env, pol, num_trajs, num_steps, action_noise=None, container=None,
            visualize=None, image_visualizer=None, record_file=None, writer=None,
            reward_type='target_action', target_distance=0):
    assert len(pol.policies) == 2
    target_pol, servoing_pol = pol.policies
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(servoing_pol, policy.ServoingPolicy)

    if target_distance:
        random_pol = policy.RandomPolicy(env.action_space, env.state_space)

    error_names = env.get_error_names()
    all_errors = []
    errors_header_format = "{:>30}" + "{:>15}" * (len(error_names) + 1)
    errors_row_format = "{:>30}" + "{:>15.4f}" * (len(error_names) + 1)
    print('=' * (30 + 15 * (len(error_names) + 1)))
    print(errors_header_format.format("(traj_iter, step_iter)", *(error_names + [reward_type or 'cost'])))
    done = False
    states = []
    actions = []
    rewards = []
    next_states = []
    if reward_type == 'target_action':
        target_actions = []
        action_transformer = predictor.transformers['u']
        # TODO: no op
        action_transformer = utils.Transformer()
    for traj_iter in range(num_trajs):
        print('traj_iter', traj_iter)
        try:
            state = pol.reset()
            env.reset(state)

            target_obs = env.observe()
            target_image = target_obs[0]
            servoing_pol.set_image_target(target_image)

            if target_distance:
                reset_action = random_pol.act(obs=None)
                for _ in range(target_distance):
                    env.step(reset_action)

            if isinstance(env, envs.Pr2Env):
                import rospy
                rospy.sleep(1)

            if container:
                container.add_datum(traj_iter,
                                    **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], target_obs)))
            for step_iter in range(num_steps):
                state, obs = env.get_state_and_observe()
                image = obs[0]

                if reward_type == 'target_action':
                    target_action = target_pol.act(obs, tightness=1.0)
                    target_actions.append(target_action)

                action = pol.act(obs)
                # print(action_transformer.preprocess(action), action_transformer.preprocess(target_action))
                # TODO
                if action_noise is not None:
                    # TODO
                    action = target_pol.act(obs)
                    import utils.transformations as tf
                    scale = (env.action_space.high - env.action_space.low)
                    action_noise = np.random.normal(scale=scale / 5.0)
                    axis_angle = tf.axis_angle_from_matrix(tf.random_rotation_matrix())
                    axis, angle = tf.split_axis_angle(axis_angle)
                    action_noise = np.r_[action_noise[:3], axis * action_noise[3]]
                    action += action_noise
                env.step(action)  # action is updated in-place if needed

                # errors
                errors = env.get_errors(target_pol.get_target_state())
                all_errors.append(errors.values())
                if step_iter == (num_steps - 1):
                    next_state, next_obs = env.get_state_and_observe()
                    next_errors = env.get_errors(target_pol.get_target_state())
                    all_errors.append(next_errors.values())

                # states, actions, next_states, rewards
                states.append((obs, target_obs))
                actions.append(action)
                if step_iter > 0:
                    next_states.append((obs, target_obs))
                    if reward_type == 'errors':
                        reward = np.array(errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        reward = ((predictor.preprocess(image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        reward = ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        reward = np.array(errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        reward = ((action_transformer.preprocess(target_actions[-2]) - action_transformer.preprocess(actions[-2])) ** 2).sum()
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                    rewards.append(reward)
                    print(errors_row_format.format(str((traj_iter, step_iter - 1)), *(all_errors[-2] + [reward])))
                if step_iter == (num_steps - 1):
                    next_image = next_obs[0]
                    next_states.append((next_obs, target_obs))
                    if reward_type == 'errors':
                        next_reward = np.array(next_errors.values()).dot([1., 5.])
                    elif reward_type == 'image':
                        next_reward = ((predictor.preprocess(next_image)[0] - predictor.preprocess(target_image)[0]) ** 2).mean()
                    elif reward_type == 'mask':
                        next_reward = ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'errors_and_mask':
                        next_reward = np.array(next_errors.values()).dot([0.1, 1.0]) + ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(target_obs[1])[0]) ** 2).mean()
                    elif reward_type == 'target_action':
                        next_reward = ((action_transformer.preprocess(target_actions[-1]) - action_transformer.preprocess(actions[-1])) ** 2).sum()
                    else:
                        raise ValueError('Invalid reward type %s' % reward_type)
                    rewards.append(next_reward)
                    print(errors_row_format.format(str((traj_iter, step_iter)), *(all_errors[-1] + [next_reward])))

                # container
                if container:
                    container.add_datum(traj_iter, step_iter, state=state, action=action,
                                        reward=((action_transformer.preprocess(target_actions[-1]) - \
                                                 action_transformer.preprocess(actions[-1])) ** 2).sum(),
                                        **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                    if step_iter == (num_steps-1):
                        container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                            **dict(list(next_errors.items()) + list(zip(env.sensor_names, next_obs))))

                # visualization
                if visualize:
                    env.render()
                    next_obs = env.observe()
                    next_image = next_obs[0]
                    vis_images = [image, next_image, target_image]
                    for i, vis_image in enumerate(vis_images):
                        vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                    if visualize == 1:
                        vis_features = vis_images
                    else:
                        feature = predictor.feature(image)
                        feature_next_pred = predictor.next_feature(image, action)
                        feature_next = predictor.feature(next_image)
                        feature_target = predictor.feature(target_image)
                        # put all features into a flattened list
                        vis_features = [feature, feature_next_pred, feature_next, feature_target]
                        if not isinstance(predictor.feature_name, str):
                            vis_features = [vis_features[icol][irow] for irow in range(image_visualizer.rows - 1) for icol in range(image_visualizer.cols)]
                        vis_images.insert(2, None)
                        vis_features = vis_images + vis_features
                    # deprocess features if they have 3 channels (useful for RGB images)
                    for i, vis_feature in enumerate(vis_features):
                        if vis_feature is None:
                            continue
                        if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                            vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                    try:
                        image_visualizer.update(vis_features)
                        if record_file:
                            writer.grab_frame()
                    except _tkinter.TclError:  # TODO: is this the right exception?
                        done = True
                if done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    print('-' * (30 + 15 * (len(error_names) + 1)))
    rms_errors = np.sqrt(np.mean(np.square(all_errors), axis=0))
    rms_reward = np.sqrt(np.mean(np.square(rewards), axis=0))
    print(errors_row_format.format("RMS", *np.r_[rms_errors, rms_reward]))
    return all_errors, states, actions, next_states, rewards


def entropy_gaussian(variance):
    return .5 * (1 + np.log(2 * np.pi)) + .5 * np.log(variance)


def compute_information_gain(predictor, env, num_trajs, num_steps, scale=0, use_features=True, learn_masks=False, visualize=None):
    policy_config = predictor.policy_config
    replace_config = {'env': env}
    try:
        replace_config['target_env'] = env.car_env
    except AttributeError:
        pass
    pol = utils.from_config(policy_config, replace_config=replace_config)
    if isinstance(pol, policy.RandomPolicy):
        target_pol = pol
        random_pol = pol
    else:
        assert len(pol.policies) == 2
        target_pol, random_pol = pol.policies
        assert pol.reset_probs == [1, 0]
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(random_pol, policy.RandomPolicy)

    if visualize:
        fig = plt.figure(figsize=(4, 4), frameon=False, tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], 1)
        plt.show(block=False)

    images = []
    labels = []
    done = False
    for car_visible in [False, True]:
        env.car_env.car_node.setVisible(car_visible)
        for traj_iter in range(num_trajs):
            state = pol.reset()
            env.reset(state)
            for step_iter in range(num_steps):
                obs = env.observe()
                image = obs[0]
                action = pol.act(obs)
                env.step(action)  # action is updated in-place if needed

                images.append(image)
                labels.append(car_visible)

                if visualize:
                    vis_images = [predictor.preprocess(image)[0]]
                    vis_features = vis_images
                    for i, vis_feature in enumerate(vis_features):
                        if vis_feature is None:
                            continue
                        if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                            vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                    try:
                        image_visualizer.update(vis_features)
                    except _tkinter.TclError:  # TODO: is this the right exception?
                        done = True
                if done:
                    break
            if done:
                break
        if done:
            break

    images = np.array(images)
    labels = np.array(labels)
    if use_features:
        Xs = []
        ind = 0
        while ind < len(images):
            X = predictor.feature(np.array(images[ind:ind+100]))[scale]
            Xs.append(X)
            ind += 100
        X = np.concatenate(Xs)
    else:
        X = predictor.preprocess(images)[0]
    y = labels

    if learn_masks:
        std_axes = 0
    else:
        std_axes = (0, 2, 3)
    p_y1 = (y == 1).mean()
    p_y0 = 1 - p_y1
    x_std = X.std(axis=std_axes).flatten()
    x_y0_std = X[y == 0].std(axis=std_axes).flatten()
    x_y1_std = X[y == 1].std(axis=std_axes).flatten()
    information_gain = entropy_gaussian(x_std ** 2) - \
                       (p_y0 * entropy_gaussian(x_y0_std ** 2) + \
                        p_y1 * entropy_gaussian(x_y1_std ** 2))
    # assert np.allclose(information_gain, 0.5 * (np.log(x_std ** 2) - p_y0 * np.log(x_y0_std ** 2) - p_y1 * np.log(x_y1_std ** 2)))
    if learn_masks:
        information_gain = information_gain.reshape((X.shape[1:]))
        information_gain[np.isnan(information_gain)] = 0
        information_gain[np.isinf(information_gain)] = 0
    return information_gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('learning_type', type=str, choices=['cem', 'mc', 'imitation_learning', 'fitted_q_iteration', 'dqn'])
    parser.add_argument('--pol_fname', '--pol', type=str, help='config file with policy arguments for overriding')
    parser.add_argument('--car_model_name', type=str, nargs='+', help='car model name(s) for overriding')
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_exploration_trajs', '-e', type=int, default=None)
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--plot', '-p', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    parser.add_argument('--target_distance', '-d', type=float, default=0.0)
    parser.add_argument('--feature_inds', '-i', type=int, nargs='+', help='inds of subset of features to use')
    parser.add_argument('--use_alex_weights', '-a', action='store_true')
    parser.add_argument('--use_information_gain_weights', '-g', action='store_true')
    parser.add_argument('--learn_lambda', '-l', action='store_true')
    parser.add_argument('--learning_iters', type=int, default=10)
    parser.add_argument('--action_sigma', type=float, default=0.2)
    parser.add_argument('--reset_after_target', type=int, default=0)
    parser.add_argument('--alg_iters', type=int, default=None)
    parser.add_argument('--alg_l2_reg', type=float, default=0.0)
    parser.add_argument('--alg_max_batch_size', type=int, default=1000)
    parser.add_argument('--alg_max_memory_size', type=int, default=0)
    parser.add_argument('--alg_learning_rate', '--alg_lr', type=float, default=None)
    parser.add_argument('--fqi_eps', type=float, default=None)
    parser.add_argument('--fqi_use_variable_bias', type=int, default=0)
    parser.add_argument('--fqi_fit_free_params', type=int, default=1)
    parser.add_argument('--fqi_constrain_theta', type=int, default=0)
    parser.add_argument('--w_init', type=float, default=None)
    parser.add_argument('--lambda_init', type=float, default=None)

    args = parser.parse_args()

    predictor = utils.from_yaml(open(args.predictor_fname))
    # predictor.environment_config['car_color'] = 'green'
    if args.feature_inds:
        predictor.feature_name = [predictor.feature_name[ind] for ind in args.feature_inds]
        predictor.next_feature_name = [predictor.next_feature_name[ind] for ind in args.feature_inds]
    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("visual_servoing_weighted")
    environment_config = predictor.environment_config
    if args.car_model_name is not None:
        environment_config['car_model_name'] = args.car_model_name
    env = utils.from_config(environment_config)

    if args.pol_fname:
        with open(args.pol_fname) as yaml_string:
            policy_config = yaml.load(yaml_string)
    else:
        policy_config = predictor.policy_config
    replace_config = {'env': env,
                      'action_space': env.action_space,
                      'state_space': env.state_space}
    try:
        replace_config['target_env'] = env.car_env
    except AttributeError:
        pass
    pol = utils.from_config(policy_config, replace_config=replace_config)
    if isinstance(pol, policy.RandomPolicy):
        target_pol = pol
        random_pol = pol
    else:
        assert len(pol.policies) == 2
        target_pol, random_pol = pol.policies
        assert pol.reset_probs == [1, 0]
    del pol
    assert isinstance(target_pol, policy.TargetPolicy)
    assert isinstance(random_pol, policy.RandomPolicy)
    # servoing_pol = policy.ServoingPolicy(predictor, alpha=1.0, lambda_=1.0)
    # servoing_pol.w = 10.0 * np.ones_like(servoing_pol.w)
    # servoing_pol = policy.TheanoServoingPolicy(predictor, alpha=1.0, lambda_=1.0)
    servoing_pol = policy.TheanoServoingPolicy(predictor, alpha=1.0, lambda_=1.0)
    servoing_pol.w = 10.0 * np.ones_like(servoing_pol.w)
    if args.w_init is not None:
        servoing_pol.w = args.w_init * np.ones_like(servoing_pol.w)
    if args.lambda_init is not None:
        servoing_pol.lambda_ = args.lambda_init * np.ones_like(servoing_pol.lambda_)

    ### BEST
    # servoing_pol = policy.TheanoServoingPolicy(predictor, alpha=1.0, lambda_=10.0)
    # if len(servoing_pol.w) == 9:
    #     servoing_pol.w = np.array([22.446248, 16.38214429, 20.08670612, 12.22685585, 17.8761917, 15.69352661, 17.6225469, 8.0052753, 21.75278794])
    # else:
    #     servoing_pol.w = 15.0 * np.ones_like(servoing_pol.w)
    # servoing_pol.lambda_ = np.array([13.04235679, 4.80352296, 15.26906728, 11.25787556])

    # servoing_pol.w /= servoing_pol.lambda_[0]
    # servoing_pol.lambda_ /= servoing_pol.lambda_[0]

    # pol = policy.MixedPolicy([target_pol, servoing_pol], act_probs=[0, 1], reset_probs=[1, 0])
    reset_pol = target_pol

    error_names = env.get_error_names()
    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['target_state'] + ['target_' + sensor_name for sensor_name in env.sensor_names], args.num_trajs)
        container.reserve(env.sensor_names + ['state'] + error_names, (args.num_trajs, args.num_steps + 1))
        container.reserve(['action', 'reward'], (args.num_trajs, args.num_steps))
        container.add_info(environment_config=env.get_config())
        container.add_info(policy_config=pol.get_config())
    else:
        container = None

    if args.alg_iters is None:
        if args.learning_type == 'imitation_learning':
            args.alg_iters = 1000
        elif args.learning_type == 'fitted_q_iteration':
            args.alg_iters = 10
        elif args.learning_type == 'dqn':
            args.alg_iters = 1000

    if args.alg_learning_rate is None:
        if args.learning_type == 'imitation_learning':
            args.alg_learning_rate = 0.01
        elif args.learning_type == 'dqn':
            args.alg_learning_rate = 0.01

    import os
    conditions = args.predictor_fname.split('/')[-3:-1]
    if args.pol_fname:
        conditions.append(os.path.splitext(args.pol_fname.split('/')[-1])[0])
    if args.car_model_name is not None:
        if args.car_model_name == ['camaro2']:
            conditions.append('red')
        else:
            NotImplementedError
    conditions.append(str(args.action_sigma))
    # TODO: hack
    if args.learning_type == 'imitation_learning':
        conditions.append(str(args.alg_iters))
        conditions.append(str(args.alg_l2_reg))
        conditions.append(str(args.alg_learning_rate))
    elif args.learning_type == 'fitted_q_iteration':
        conditions.append(str(args.alg_iters))
        conditions.append(str(args.alg_l2_reg))
        conditions.append(str(args.alg_max_batch_size))
        conditions.append(str(args.alg_max_memory_size))
        conditions.append(str(args.fqi_eps))
        conditions.append(str(args.fqi_use_variable_bias))
        conditions.append(str(args.fqi_fit_free_params))
        conditions.append(str(args.fqi_constrain_theta))
    elif args.learning_type == 'dqn':
        conditions.append(str(args.alg_iters))
        conditions.append(str(args.alg_l2_reg))
        conditions.append(str(args.alg_max_batch_size))
        conditions.append(str(args.alg_max_memory_size))
        conditions.append(str(args.alg_learning_rate))
    conditions = '_'.join(conditions)

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        rows, cols = 1, 3
        labels = [predictor.input_names[0], predictor.input_names[0] + ' next', predictor.input_names[0] + ' target']
        if args.visualize > 1:
            single_feature = isinstance(predictor.feature_name, str)
            rows += 1 if single_feature else len(predictor.feature_name)
            cols += 1
            if single_feature:
                assert isinstance(predictor.next_feature_name, str)
                feature_names = [predictor.feature_name]
                next_feature_names = [predictor.next_feature_names]
            else:
                assert len(predictor.feature_name) == len(predictor.next_feature_name)
                feature_names = predictor.feature_name
                next_feature_names = predictor.next_feature_name
            labels.insert(2, '')
            for feature_name, next_feature_name in zip(feature_names, next_feature_names):
                labels += [feature_name, feature_name + ' next', next_feature_name, feature_name + ' target']
        fig = plt.figure(figsize=(4 * cols, 4 * rows), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(conditions)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols, labels=labels)
        plt.show(block=False)
        if args.record_file:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.record_file, fig.dpi)
    else:
        image_visualizer = None

    np.set_printoptions(suppress=True)
    # # q_learning = QLearning(servoing_pol, gamma=0.0, l1_reg=0.0, learn_lambda=False, experience_replay=False, max_iters=10)
    # # q_learning = QLearning(servoing_pol, gamma=0.0, l2_reg=1e-6, learn_lambda=False, experience_replay=False, max_iters=1)
    # # q_learning = QLearning(servoing_pol, gamma=0.0, l2_reg=0.0, learn_lambda=False, experience_replay=False, max_iters=1)
    # q_learning = QLearning(servoing_pol, gamma=0.9, tr_reg=1., learn_lambda=args.learn_lambda, experience_replay=False, max_iters=1)
    # if args.use_alex_weights:
    #     assert not args.use_information_gain_weights
    #     if q_learning.theta.shape[0] == 3 * 3 + 2:
    #         theta_init = np.zeros(3)
    #         theta_init[0] = 0.1
    #         theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
    #     elif q_learning.theta.shape[0] == 3 + 2:
    #         theta_init = np.zeros(3)
    #         theta_init[0] = 0.1
    #         theta_init = np.r_[theta_init, [0.0, q_learning.lambda_]]
    #     elif q_learning.theta.shape[0] == 512 * 3 + 2:
    #         theta_init = np.zeros(512)
    #         theta_init[[36, 178, 307, 490]] = 0.1
    #         # theta_init[[0, 1, 2, 3]] = 0.1
    #         # TODO: zeros for lower resolutions
    #         # theta_init = np.r_[theta_init, np.zeros_like(theta_init), np.zeros_like(theta_init), [0]]
    #         theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
    #     elif q_learning.theta.shape[0] == 8 * 3 + 2:
    #         theta_init = np.zeros(8)
    #         theta_init[4:8] = 3.0
    #         theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
    #     else:
    #         raise ValueError('alex weights are not specified')
    #     q_learning.theta = theta_init
    #
    # if args.use_information_gain_weights:
    #     assert not args.use_alex_weights
    #     information_gain = np.concatenate([compute_information_gain(predictor, env, args.num_trajs, args.num_steps, scale=scale) for scale in range(len(predictor.feature_name))])
    #     q_learning.theta = np.r_[information_gain, q_learning.theta[len(information_gain):]]

    # # TODO: fix how w doesn't get updated when theta is gotten by reference
    # if 'vgg' in args.predictor_fname:
    #     q_learning.theta = np.r_[q_learning.theta[:-2] * 1e4, q_learning.theta[-2:]]

    ###
    # theta_init = np.zeros(8)
    # theta_init[4:8] = 3.0
    # theta_init = np.r_[theta_init, theta_init, theta_init, [0.0, q_learning.lambda_]]
    # q_learning.theta = theta_init
    # np.random.set_state(exploitation_random_state)
    # all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
    #                                                             visualize=args.visualize,
    #                                                             image_visualizer=image_visualizer)
    ###

    # import IPython as ipy; ipy.embed()
    # np.random.seed(seed=7)
    # exploitation_random_state = np.random.get_state()
    #
    # information_gain = np.concatenate(
    #     [compute_information_gain(predictor, env, args.num_trajs, args.num_steps, scale=scale) for scale in
    #      range(len(predictor.feature_name))])
    # q_learning.theta = np.r_[10.0 * information_gain, q_learning.theta[len(information_gain):]]
    # np.random.set_state(exploitation_random_state)
    # all_errors, states, actions, next_states, rewards = rollout(predictor, env, pol, args.num_trajs, args.num_steps,
    #                                                            visualize=args.visualize,
    #                                                            image_visualizer=image_visualizer)

    if args.plot:
        # plotting
        fig = plt.figure(figsize=(12, 6), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(conditions)
        gs = gridspec.GridSpec(1, 2)
        plt.show(block=False)

        # esd_cost_plotter = LossPlotter(fig, gs[0], labels=['esd_cost'], ylabel='cost')
        esd_cost_plotter = LossPlotter(fig, gs[0], format_dicts=[dict(linewidth=2)] * 2, ylabel='mean discounted sum of costs')
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        # esd_cost_plotter._ax.set_ylim((0.0, 150))
        esd_cost_major_locator = MultipleLocator(1)
        esd_cost_major_formatter = FormatStrFormatter('%d')
        esd_cost_minor_locator = MultipleLocator(1)
        esd_cost_plotter._ax.xaxis.set_major_locator(esd_cost_major_locator)
        esd_cost_plotter._ax.xaxis.set_major_formatter(esd_cost_major_formatter)
        esd_cost_plotter._ax.xaxis.set_minor_locator(esd_cost_minor_locator)

        if args.learning_type == 'cem':
            label = 'Episode mean cost'
        elif args.learning_type == 'mc':
            label = 'Monte Carlo Bellman error'
        elif args.learning_type == 'imitation_learning':
            label = 'Imitation learning training loss'
        elif args.learning_type == 'fitted_q_iteration':
            label = 'Fitted Q iteration Bellman error'
        elif args.learning_type == 'dqn':
            label = 'DQN Bellman error'
        else:
            raise ValueError
        if args.learning_type == 'fitted_q_iteration':
            # learning_cost_plotter = LossPlotter(fig, gs[1], format_strings=['', '--'], labels=[label], ylabel='cost')
            learning_cost_plotter = LossPlotter(fig, gs[1],
                                                format_strings=['', 'r--'],
                                                format_dicts=[dict(linewidth=2)] * 2,
                                                ylabel='Bellman error', yscale='log')
            # learning_cost_plotter._ax.set_ylim((10.0, 110000))
            learning_cost_major_locator = MultipleLocator(1)
            learning_cost_major_formatter = FormatStrFormatter('%d')
            learning_cost_minor_locator = MultipleLocator(0.2)
            learning_cost_plotter._ax.xaxis.set_major_locator(learning_cost_major_locator)
            learning_cost_plotter._ax.xaxis.set_major_formatter(learning_cost_major_formatter)
            learning_cost_plotter._ax.xaxis.set_minor_locator(learning_cost_minor_locator)
        else:
            learning_cost_plotter = LossPlotter(fig, gs[1], labels=[label], ylabel='cost')

    if False:
        import IPython as ipy; ipy.embed()

        scale = 0.1
        num_trajs = 10
        num_steps = 25
        esd_rewards = []
        seed = 9
        ###### START
        if 'vgg' in predictor.name:
            slices = [[36, 178, 307, 490]]
        else:
            slices = [slice(None)]
        # for i in :
        for i in slices:
            print(i)
            np.random.seed(seed=seed)
            theta = np.zeros_like(q_learning.theta)
            theta[i] = scale
            q_learning.theta = theta

            print("USING THETA", q_learning.theta)
            print(servoing_pol.w)
            all_errors = []
            errors_header_format = "{:>30}" + "{:>15}" * len(error_names)
            errors_row_format = "{:>30}" + "{:>15.2f}" * len(error_names)
            print('=' * (30 + 15 * len(error_names)))
            print(errors_header_format.format("(traj_iter, step_iter)", *error_names))
            done = False
            states = []
            actions = []
            rewards = []
            next_states = []
            for traj_iter in range(num_trajs):
                print('traj_iter', traj_iter)
                try:
                    if traj_iter == -1:
                        state = np.array([  58.44558759,  273.34368945,   12.57534264,   -0.10993641,0.11193145,   -1.58590685,    2.5       ,    2.        ,0.        ,   -1.        ,   20.        ,   22.        ,  129])
                    else:
                        state = pol.reset()
                    env.reset(state)

                    obs_target = env.observe()
                    image_target = obs_target[0]
                    servoing_pol.set_image_target(image_target)

                    if container:
                        container.add_datum(traj_iter,
                                            **dict(zip(['target_' + sensor_name for sensor_name in env.sensor_names], obs_target)))
                    for step_iter in range(num_steps):
                        if step_iter > 0:
                            prev_obs = obs
                        state, obs = env.get_state_and_observe()
                        image = obs[0]
                        # TODO
                        prev_errors = env.get_errors(target_pol.get_target_state())

                        action = pol.act(obs)
                        # NO NOISE
                        # action += np.random.normal(scale=0.1, size=action.shape)
                        # action[:3] += 10 * (np.random.random(3) - 0.5)
                        # action = env.action_space.sample()
                        env.step(action)  # action is updated in-place if needed

                        # errors
                        errors = env.get_errors(target_pol.get_target_state())
                        print(errors_row_format.format(str((traj_iter, step_iter)), *errors.values()))
                        all_errors.append(errors.values())

                        states.append((obs, obs_target))
                        actions.append(action)
                        # rewards.append(list(errors.values()))
                        # rewards.append((np.array(errors.values()) - np.array(prev_errors.values())).dot(np.array([1.0, 50.0])))
                        rewards.append(np.array(errors.values()).dot(np.array([1.0, 0.0])))  # TODO
                        if step_iter > 0:
                            next_states.append((obs, obs_target))
                            # rewards.append(((predictor.preprocess(obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(prev_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                        if step_iter == (args.num_steps-1):
                            next_state, next_obs = env.get_state_and_observe()
                            next_states.append((next_obs, obs_target))
                            # rewards.append(((predictor.preprocess(next_obs[0])[0][0] - predictor.preprocess(image_target)[0][0]) ** 2).mean())
                            # rewards.append(((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())
                            # rewards.append(
                            #     ((predictor.preprocess(next_obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean() -
                            #     ((predictor.preprocess(obs[1])[0] - predictor.preprocess(obs_target[1])[0]) ** 2).mean())

                        # container
                        if container:
                            if step_iter > 0:
                                container.add_datum(traj_iter, step_iter - 1, state_diff=state - prev_state)
                            container.add_datum(traj_iter, step_iter, state=state, action=action,
                                                **dict(list(errors.items()) + list(zip(env.sensor_names, obs))))
                            prev_state = state
                            if step_iter == (args.num_steps-1):
                                container.add_datum(traj_iter, step_iter, state_diff=next_state - state)
                                container.add_datum(traj_iter, step_iter + 1, state=next_state,
                                                    **dict(zip(env.sensor_names, next_obs)))

                        # visualization
                        if args.visualize:
                            env.render()
                            obs_next = env.observe()
                            image_next = obs_next[0]
                            vis_images = [image, image_next, image_target]
                            for i, vis_image in enumerate(vis_images):
                                vis_images[i] = predictor.transformers['x'].preprocess(vis_image)
                            if args.visualize == 1:
                                vis_features = vis_images
                            else:
                                feature = predictor.feature(image)
                                feature_next_pred = predictor.next_feature(image, action)
                                feature_next = predictor.feature(image_next)
                                feature_target = predictor.feature(image_target)
                                # put all features into a flattened list
                                vis_features = [feature, feature_next_pred, feature_next, feature_target]
                                if not isinstance(predictor.feature_name, str):
                                    vis_features = [vis_features[icol][irow] for irow in range(rows - 1) for icol in range(cols)]
                                vis_images.insert(2, None)
                                vis_features = vis_images + vis_features
                            # deprocess features if they have 3 channels (useful for RGB images)
                            for i, vis_feature in enumerate(vis_features):
                                if vis_feature is None:
                                    continue
                                if vis_feature.ndim == 3 and vis_feature.shape[0] == 3:
                                    vis_features[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_feature)
                            try:
                                image_visualizer.update(vis_features)
                                if args.record_file:
                                    writer.grab_frame()
                            except _tkinter.TclError:  # TODO: is this the right exception?
                                done = True
                        if done:
                            break
                    if done:
                        break
                except KeyboardInterrupt:
                    done = True
                    break
            if done:
                break
            esd_reward = (np.array(rewards).reshape((num_trajs, num_steps))
                          * (q_learning.gamma ** np.arange(num_steps))[None, :]).sum(axis=1).mean()
            print("Expected sum of discounted rewards", esd_reward)
            esd_rewards.append(esd_reward)
    ###### END

    rms_positions = []
    rms_rotations = []
    rms_rewards = []
    bellman_errors = []
    thetas = []
    esd_rewards = []
    np.random.seed(seed=7)

    num_exploration_trajs = args.num_exploration_trajs if args.num_exploration_trajs is not None else args.num_trajs

    gamma = 0.9
    act_std = args.action_sigma

    def noisy_evaluation(theta, num_trajs, num_steps, act_std=0.1, visualize=0, deterministic=True, reset_after_target=False):
        if deterministic:
            random_state = np.random.get_state()
        # save servoing parameters
        w, lambda_ = servoing_pol.w.copy(), servoing_pol.lambda_.copy()
        servoing_pol.w, servoing_pol.lambda_ = np.split(theta, [len(servoing_pol.w)])
        noisy_pol = policy.AdditiveNormalPolicy(servoing_pol, env.action_space, env.state_space, act_std=act_std)
        for traj_iter in range(num_trajs):
            if deterministic:
                np.random.seed(traj_iter)
            print('noisy_evaluation. traj_iter', traj_iter)
            states, actions, rewards, next_states, target_actions = do_rollout(predictor, env, reset_pol, noisy_pol, target_pol,
                                                                               num_steps,
                                                                               visualize=visualize,
                                                                               image_visualizer=image_visualizer,
                                                                               target_distance=args.target_distance,
                                                                               quiet=True,
                                                                               reset_after_target=reset_after_target)
            esd_reward = np.array(rewards).dot(gamma ** np.arange(len(rewards)))
            esd_rewards.append(esd_reward)
        # restore servoing parameters
        servoing_pol.w, servoing_pol.lambda_ = w, lambda_
        if deterministic:
            np.random.set_state(random_state)
        return np.mean(esd_rewards)

    def train_rollouts(num_trajs, num_steps, act_std=0.1, visualize=0, deterministic=True, reset_after_target=False):
        if deterministic:
            random_state = np.random.get_state()
        if act_std != 0:
            noisy_pol = policy.AdditiveNormalPolicy(servoing_pol, env.action_space, env.state_space, act_std=act_std)
        else:
            noisy_pol = servoing_pol
        states, actions, rewards, next_states, target_actions = [], [], [], [], []
        for traj_iter in range(num_trajs):
            if deterministic:
                np.random.seed(traj_iter)
            print('train_rollouts. traj_iter', traj_iter)
            states_, actions_, rewards_, next_states_, target_actions_ = \
                do_rollout(predictor, env, reset_pol, noisy_pol, target_pol,
                           num_steps, visualize=visualize, image_visualizer=image_visualizer,
                           target_distance=args.target_distance, quiet=True, reset_after_target=reset_after_target)
            states.append(states_)
            actions.append(actions_)
            rewards.append(rewards_)
            next_states.append(next_states_)
            target_actions.append(target_actions_)
        if deterministic:
            np.random.set_state(random_state)
        return states, actions, rewards, next_states, target_actions

    def best_train_rollouts(num_trajs, num_steps, act_std=0.1, visualize=0, deterministic=True, reset_after_target=False):
        if deterministic:
            random_state = np.random.get_state()
        if act_std != 0:
            noisy_pol = policy.AdditiveNormalPolicy(target_pol, env.action_space, env.state_space, act_std=act_std)
        else:
            noisy_pol = target_pol
        states, actions, rewards, next_states, target_actions = [], [], [], [], []
        for traj_iter in range(num_trajs):
            if deterministic:
                np.random.seed(traj_iter)
            print('train_rollouts. traj_iter', traj_iter)
            states_, actions_, rewards_, next_states_, target_actions_ = \
                do_rollout(predictor, env, reset_pol, noisy_pol, target_pol,
                           num_steps, visualize=visualize, image_visualizer=image_visualizer,
                           target_distance=args.target_distance, quiet=True, reset_after_target=reset_after_target)
            states.append(states_)
            actions.append(actions_)
            rewards.append(rewards_)
            next_states.append(next_states_)
            target_actions.append(target_actions_)
        if deterministic:
            np.random.set_state(random_state)
        return states, actions, rewards, next_states, target_actions

    def test_rollouts(num_trajs, num_steps, visualize=0, deterministic=True, quiet=None, reset_after_target=False):
        if quiet is None:
            quiet = not bool(visualize)
        if deterministic:
            random_state = np.random.get_state()
        esd_rewards = []
        for traj_iter in range(num_trajs):
            if deterministic:
                np.random.seed(traj_iter)
            print('test_rollouts. traj_iter', traj_iter)
            states, actions, rewards, next_states, target_actions = do_rollout(predictor, env, reset_pol, servoing_pol, target_pol,
                                                                               num_steps,
                                                                               visualize=visualize,
                                                                               image_visualizer=image_visualizer,
                                                                               target_distance=args.target_distance,
                                                                               quiet=quiet,
                                                                               reset_after_target=reset_after_target)
            esd_reward = np.array(rewards).dot(gamma ** np.arange(len(rewards)))
            esd_rewards.append(esd_reward)
        if deterministic:
            np.random.set_state(random_state)
        return esd_rewards

    print(servoing_pol.w, servoing_pol.lambda_)
    # esd_rewards = test_rollouts(args.num_trajs, args.num_steps, visualize=args.visualize)
    # print(esd_rewards)
    # print(np.mean(esd_rewards))
    # servoing_pol.bias = np.mean(esd_rewards)

    # S = [traj_states[0] for traj_states in states]
    # A = np.array([traj_actions[0] for traj_actions in actions])
    # batch_size = len(S)
    # phi = np.c_[servoing_pol.phi(S, A, preprocessed=True), np.ones(batch_size)]
    # Q_sample = np.array([np.dot(traj_rewards, gamma ** np.arange(len(traj_rewards))) for traj_rewards in rewards])
    # fig = plt.figure()
    # plt.scatter(phi.dot(theta), Q_sample)

    # states, actions, rewards, next_states, target_actions = train_rollouts(num_exploration_trajs,
    #                                                                        args.num_steps,
    #                                                                        act_std=act_std)
    # S, A, R, S_p = (states, actions, rewards, next_states)
    # theta = np.r_[servoing_pol.w, servoing_pol.lambda_]
    # phi = servoing_pol.phi(S, A)
    # A_p = servoing_pol.pi(S_p)
    # phi_p = servoing_pol.phi(S_p, A_p)
    # V_p = phi_p.dot(theta)
    # Q_sample = R + gamma * V_p
    # servoing_pol.bias = (Q_sample - phi.dot(theta)).mean(axis=0) / (1 - gamma)
    # print("gamma", gamma)
    # servoing_pol.bias = 374.90766783633501
    # return

    reset_after_target = args.reset_after_target
    if args.learning_type == 'cem':
        esd_costs = []
        learning_costs = []
        theta = np.r_[servoing_pol.w, servoing_pol.lambda_]
        thetas = []
        for i, iterdata in enumerate(cem(lambda theta: noisy_evaluation(theta,
                                                                        num_exploration_trajs,
                                                                        args.num_steps,
                                                                        act_std=act_std,
                                                                        reset_after_target=reset_after_target),
                                         theta, batch_size=min(3 * len(theta), 100),
                                         n_iter=args.learning_iters, elite_frac=0.2,
                                         initial_std=np.r_[10.0 * np.ones_like(servoing_pol.w), np.ones_like(servoing_pol.lambda_)],
                                         th_low=0.0)):
            theta = iterdata['theta_mean']
            print(theta)
            thetas.append(theta.copy())
            servoing_pol.w, servoing_pol.lambda_ = np.split(theta, [len(servoing_pol.w)])
            print('Iteration %2i. Episode mean cost: %7.3f' % (i, iterdata['y_mean']))
            learning_costs.append(iterdata['y_mean'])
            if args.plot:
                learning_cost_plotter.update([learning_costs])
            # test
            esd_rewards = test_rollouts(args.num_trajs, args.num_steps, visualize=args.visualize)
            esd_cost = np.mean(esd_rewards)
            print(esd_rewards)
            print(esd_cost)
            esd_costs.append(esd_cost)
            if args.plot:
                esd_cost_plotter.update([esd_costs])
    elif args.learning_type == 'mc':
        states, actions, rewards, next_states, target_actions = \
            train_rollouts(args.num_exploration_trajs, args.num_steps, act_std=0.0, visualize=args.visualize, reset_after_target=reset_after_target)
        import IPython as ipy; ipy.embed()
        alg = MonteCarloAlgorithm(servoing_pol, gamma, l2_reg=args.alg_l2_reg)
        bellman_error = alg.update(states, actions, rewards)
        print('Monte Carlo Bellman error: %7.3f' % bellman_error)
        theta = np.r_[servoing_pol.w, servoing_pol.lambda_, [servoing_pol.bias]]
        print(theta)
    elif args.learning_type in ('imitation_learning', 'fitted_q_iteration', 'dqn'):
        esd_costs = []
        learning_costs = []
        thetas = []
        if args.learning_type == 'imitation_learning':
            alg = ImitationLearningAlgorithm(servoing_pol, args.alg_iters,
                                             l2_reg=args.alg_l2_reg,
                                             learning_rate=args.alg_learning_rate)
        elif args.learning_type == 'fitted_q_iteration':
            all_fqi_thetas = []
            all_fqi_bellman_errors = []
            alg = FittedQIterationAlgorithm(servoing_pol, args.alg_iters, gamma,
                                            l2_reg=args.alg_l2_reg,
                                            eps=args.fqi_eps,
                                            use_variable_bias=args.fqi_use_variable_bias,
                                            fit_free_params=args.fqi_fit_free_params,
                                            constrain_theta=args.fqi_constrain_theta,
                                            max_batch_size=args.alg_max_batch_size,
                                            max_memory_size=args.alg_max_memory_size)
        elif args.learning_type == 'dqn':
            alg = DqnAlgorithm(servoing_pol, args.alg_iters, gamma,
                               l2_reg=args.alg_l2_reg,
                               learning_rate=args.learning_rate,
                               max_batch_size=args.alg_max_batch_size,
                               max_memory_size=args.alg_max_memory_size)

        esd_rewards = test_rollouts(args.num_trajs, args.num_steps, visualize=args.visualize, reset_after_target=reset_after_target)
        init_esd_cost = np.mean(esd_rewards)

        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_target_actions = []
        for i in range(args.learning_iters):
        # for i in range(args.learning_iters, 20):
            # if i == 0:
            #     states, actions, rewards, next_states, target_actions = best_train_rollouts(num_exploration_trajs,
            #                                                                                 args.num_steps,
            #                                                                                 act_std=act_std,
#                                                                                             reset_after_target=reset_after_target)
            # else:
            states, actions, rewards, next_states, target_actions = train_rollouts(num_exploration_trajs,
                                                                                   args.num_steps,
                                                                                   visualize=args.visualize,
                                                                                   act_std=act_std,
                                                                                   reset_after_target=reset_after_target)
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)
            all_target_actions.append(target_actions)
            if args.learning_type == 'imitation_learning':
                train_loss = alg.update(states, target_actions)
                print('Iteration %2i. Imitation learning training loss: %7.3f' % (i, train_loss))
                learning_costs.append(train_loss)
                theta = np.r_[servoing_pol.w, servoing_pol.lambda_]
            elif args.learning_type == 'fitted_q_iteration':
                bellman_error, fqi_thetas, fqi_bellman_errors = alg.update(states, actions, rewards, next_states)
                all_fqi_thetas.append(fqi_thetas)
                all_fqi_bellman_errors.append(fqi_bellman_errors)
                print('Iteration %2i. Fitted Q iteration Bellman error: %7.3f' % (i, bellman_error))
                learning_costs.append(bellman_error)
                theta = np.r_[servoing_pol.w, servoing_pol.lambda_, [servoing_pol.bias]]
            elif args.learning_type == 'dqn':
                bellman_error = alg.update(states, actions, rewards, next_states)
                print('Iteration %2i. DQN Bellman error: %7.3f' % (i, bellman_error))
                learning_costs.append(bellman_error)
                theta = np.r_[servoing_pol.w, servoing_pol.lambda_, [servoing_pol.bias]]
            if args.plot:
                if args.learning_type == 'fitted_q_iteration':
                    all_fqi_bellman_errors_flat = [fqi_bellman_error for fqi_bellman_errors in all_fqi_bellman_errors
                                                   for fqi_bellman_error in (fqi_bellman_errors + [None])]
                    all_fqi_iters = np.linspace(-1, i, (alg.n_iter + 1) * (i + 1) + 1)[1:]
                    all_fqi_iters_flat = [fqi_iter for fqi_iters in all_fqi_iters.reshape((-1, alg.n_iter + 1))
                                          for fqi_iter in (fqi_iters.tolist() + [None])]
                    learning_cost_plotter.update([learning_costs, all_fqi_bellman_errors_flat],
                                                 [np.arange(i+1, dtype=int) + 1,
                                                  [(iter_ + 1 if iter_ is not None else None) for iter_ in all_fqi_iters_flat]])
                else:
                    learning_cost_plotter.update([learning_costs])
            thetas.append(theta.copy())
            print(theta)
            # test
            esd_rewards = test_rollouts(args.num_trajs, args.num_steps, visualize=args.visualize, reset_after_target=reset_after_target)
            esd_cost = np.mean(esd_rewards)
            print(esd_rewards)
            print(esd_cost)
            esd_costs.append(esd_cost)
            if args.plot:
                esd_cost_plotter.update([[init_esd_cost] + esd_costs])
    else:
        ValueError('Invalid learning type %r' % args.learning_type)

    esd_rewards = test_rollouts(args.num_trajs, args.num_steps, visualize=args.visualize, reset_after_target=reset_after_target)
    print("%r\t%r" % (np.mean(esd_rewards), esd_rewards))

    # assert alg.servoing_pol == servoing_pol
    # alg.servoing_pol = None
    # info_dict = dict(
    #     esd_costs=esd_costs,
    #     learning_costs=learning_costs,
    #     thetas=thetas,
    #     gamma=gamma,
    #     all_states=all_states,
    #     all_actions=all_actions,
    #     all_rewards=all_rewards,
    #     all_next_states=all_next_states,
    #     all_target_actions=all_target_actions,
    #     all_fqi_thetas=all_fqi_thetas,
    #     all_fqi_bellman_errors=all_fqi_bellman_errors,
    #     args=args,
    # )
    # import pickle
    # # exp_fname = os.path.join('experiments/models_different_target', "%s_%s.pkl" % (conditions, args.learning_type))
    # # exp_fname = os.path.join('experiments/models', "%s_%s.pkl" % (conditions, args.learning_type))
    # # exp_fname = os.path.join('experiments/models_camaro2', "%s_%s.pkl" % (conditions, args.learning_type))
    # # exp_fname = os.path.join('experiments/models_kia_rio_blue', "%s_%s.pkl" % (conditions, args.learning_type))
    # exp_fname = os.path.join('experiments/models_other', "%s_%s.pkl" % (conditions, args.learning_type))
    # with open(exp_fname, 'wb') as exp_file:
    #     pickle.dump(info_dict, exp_file)
    # alg.servoing_pol = servoing_pol

    # rows = 1
    # cols = 2
    # fig = plt.figure(figsize=(4 * cols, 4 * rows), frameon=False, tight_layout=True)
    # fig.canvas.set_window_title(conditions)
    # gs = gridspec.GridSpec(1, 1)
    # image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols)
    # plt.show(block=False)
    #
    # for traj_iter in range(len(states)):
    #     for step_iter in range(len(states[traj_iter])):
    #         obs, target_obs = states[traj_iter][step_iter]
    #         image, _ = obs
    #         target_image, _ = target_obs
    #         vis_images = [image, target_image]
    #         for i, vis_image in enumerate(vis_images):
    #             vis_images[i] = predictor.transformers['x'].transformers[-1].deprocess(vis_image)
    #         image_visualizer.update(vis_images)

    # old cem, d = 1
    # [3.1884112429356124, 15.345436892392124, 2.9517672522530036, 14.292757756251225, 6.6149355542920825, 9.6664948941623354, 15.342569932727091, 7.3750687359838878, 9.4942180050399685, 13.668898487807599]
    # 9.79405587538

    # mc, d = 1
    # [3.9495937645203512, 16.172870385065931, 3.6010191143905481, 16.541879982890123, 7.7627364291582124, 15.106827761339384, 14.096047151895402, 8.415729133747563, 10.226576669261174, 17.381866415363678]
    # 11.3255146808

    # cem
    # [2.2758781518389801, 1.8902279015572023, 1.5774717825802365, 4.073245301785029, 1.2429286139803442, 2.3092418286331515, 4.2388242041147155, 1.5227050405230818, 3.8260965052651108, 4.2345128267917787]
    # 2.71911321571

    # old cem
    # [2.113636454001715, 2.122695112646519, 1.750093554200536, 3.9841207181751872, 1.1398574782355972, 2.3839641793792805, 4.0462166645569493, 1.2596249371214252, 3.4830499304695599, 3.9443362804908491]
    # 2.62275953093

    # mc
    # [1.8880224798775089, 2.6735645563951929, 1.6650540286936248, 4.3955511362699271, 0.86010585612114943, 2.7281847862072914, 4.4433823097307306, 0.83345966473398592, 3.8544424402436901, 4.3716750669633599]
    # 2.77134423252

    import IPython as ipy; ipy.embed()
    print('w=%r, lambda=%r, bias=%r' % (servoing_pol.w.tolist(), servoing_pol.lambda_.tolist(), servoing_pol.bias.tolist()))
    print("%r\t%r\t%r" % (np.min(esd_costs), np.mean(esd_rewards), esd_rewards))

    import IPython as ipy; ipy.embed()
    best_theta = thetas[np.argmin(esd_costs)]
    best_w, best_lambda, best_bias = np.split(best_theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))
    best_bias, = best_bias
    print("w=%r, lambda=%r, bias=%r \tw=%r, lambda=%r, bias=%r \t%r \t%r \t %s" %
          (best_w.tolist(), best_lambda.tolist(), best_bias.tolist(),
           servoing_pol.w.tolist(), servoing_pol.lambda_.tolist(), servoing_pol.bias.tolist(),
           np.min(esd_costs), np.argmin(esd_costs), '\t'.join([str(cost) for cost in esd_costs])))

    best_theta = thetas[np.argmin(esd_costs)]
    best_w, best_lambda, best_bias = np.split(best_theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))
    print("w=%r, lambda=%r\tw=%r, lambda=%r\t%r\t%r\t%r" %
          (best_w.tolist(), best_lambda.tolist(),
           servoing_pol.w.tolist(), servoing_pol.lambda_.tolist(),
           np.min(esd_costs), np.mean(esd_rewards), esd_rewards))

    # learning_cost_plotter.update([learning_costs])
    # esd_cost_plotter.update([esd_costs])
    import IPython as ipy; ipy.embed()
    return

    learning_cost_plotter = LossPlotter(fig, gs[1], format_strings=['', '--'], labels=[label], ylabel='cost')
    all_fqi_bellman_errors_flat = [fqi_bellman_error for fqi_bellman_errors in all_fqi_bellman_errors
                                   for fqi_bellman_error in (fqi_bellman_errors + [None])]
    all_fqi_iters = np.linspace(-1, i, (alg.n_iter + 1) * (i + 1) + 1)[1:]
    all_fqi_iters_flat = [fqi_iter for fqi_iters in all_fqi_iters.reshape((-1, alg.n_iter + 1))
                          for fqi_iter in (fqi_iters.tolist() + [None])]
    learning_cost_plotter.update([learning_costs, all_fqi_bellman_errors_flat],
                                 [np.arange(len(learning_costs)), all_fqi_iters_flat])


    theta = thetas[0]
    theta = best_theta
    servoing_pol.w, servoing_pol.lambda_, _ = np.split(theta, np.cumsum([len(servoing_pol.w), len(servoing_pol.lambda_)]))

    args.visualize = 1
    if args.visualize:
        rows, cols = 1, 3
        labels = [predictor.input_names[0], predictor.input_names[0] + ' next', predictor.input_names[0] + ' target']
        if args.visualize > 1:
            single_feature = isinstance(predictor.feature_name, str)
            rows += 1 if single_feature else len(predictor.feature_name)
            cols += 1
            if single_feature:
                assert isinstance(predictor.next_feature_name, str)
                feature_names = [predictor.feature_name]
                next_feature_names = [predictor.next_feature_names]
            else:
                assert len(predictor.feature_name) == len(predictor.next_feature_name)
                feature_names = predictor.feature_name
                next_feature_names = predictor.next_feature_name
            labels.insert(2, '')
            for feature_name, next_feature_name in zip(feature_names, next_feature_names):
                labels += [feature_name, feature_name + ' next', next_feature_name, feature_name + ' target']
        fig = plt.figure(figsize=(4 * cols, 4 * rows), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(conditions)
        gs = gridspec.GridSpec(1, 1)
        image_visualizer = GridImageVisualizer(fig, gs[0], rows * cols, rows=rows, cols=cols, labels=labels)
        plt.show(block=False)
        if args.record_file:
            FFMpegWriter = manimation.writers['ffmpeg']
            writer = FFMpegWriter(fps=1.0 / env.dt)
            writer.setup(fig, args.record_file, fig.dpi)
    else:
        image_visualizer = None
    quiet = False
    deterministic = True
    num_trajs = args.num_trajs
    num_steps = args.num_steps
    visualize = args.visualize
    if quiet is None:
        quiet = not bool(visualize)
    if deterministic:
        random_state = np.random.get_state()
    esd_rewards = []
    for traj_iter in range(num_trajs):
        if deterministic:
            np.random.seed(traj_iter)
        print('test_rollouts. traj_iter', traj_iter)
        states, actions, rewards, next_states, target_actions = do_rollout(predictor, env, reset_pol, servoing_pol,
                                                                           target_pol,
                                                                           num_steps,
                                                                           visualize=visualize,
                                                                           image_visualizer=image_visualizer,
                                                                           target_distance=args.target_distance,
                                                                           quiet=quiet,
                                                                           reset_after_target=reset_after_target)
        esd_reward = np.array(rewards).dot(gamma ** np.arange(len(rewards)))
        esd_rewards.append(esd_reward)
    if deterministic:
        np.random.set_state(random_state)


    sample_iter = 0
    # eps_init = 1.0
    # eps_final = 0.1
    greedy_pol = pol
    # eps_greedy_pol = policy.MixedPolicy([greedy_pol, random_pol], act_probs=[1 - eps_init, eps_init], reset_probs=[1, 0])
    normal_scale_init = 0.1
    normal_scale_final = 0.01
    try:
        for i in range(100):
            print("SAMPLE_ITER", sample_iter)

            print("Q_LEARNING THETA")
            print(q_learning.theta)
            print(np.sort(q_learning.theta))
            print(np.argsort(q_learning.theta))

            exploration_random_state = np.random.get_state()
            if i == 0:
                exploitation_random_state = np.random.get_state()
            else:
                np.random.set_state(exploitation_random_state)

            # rollout and get expected sum of rewards
            all_errors, states, actions, next_states, rewards, target_actions = rollout(predictor, env, greedy_pol,
                                                                        args.num_trajs, args.num_steps,
                                                                        visualize=args.visualize, image_visualizer=image_visualizer,
                                                                        target_distance=args.target_distance)
            rms_position, rms_rotation = np.sqrt(np.mean(np.square(all_errors), axis=0))
            rms_reward = np.sqrt(np.mean(np.square(rewards), axis=0))
            rms_positions.append(rms_position)
            rms_rotations.append(rms_rotation)
            rms_rewards.append(rms_reward)
            esd_reward = (np.array(rewards).reshape((args.num_trajs, args.num_steps))
                          * (q_learning.gamma ** np.arange(args.num_steps))[None, :]).sum(axis=1).mean()
            esd_rewards.append(esd_reward)

            # compute statistics of features once
            if i == 0:
                # batch_image = np.asarray([obs[0] for (obs, target_obs) in states])
                # batch_feature = predictor.feature(batch_image)
                # batch_features = batch_feature if isinstance(batch_feature, list) else [batch_feature]
                # feature_stds = []
                # for batch_feature in batch_features:
                #     feature_std = batch_feature.transpose([0, 2, 3, 1]).reshape((-1, batch_feature.shape[1])).std(axis=0)
                #     feature_stds.append(feature_std)
                # feature_std = np.concatenate(feature_stds)
                # feature_std[feature_std == 0] = 1.0
                # # q_learning.theta[:-2] /= feature_std

                # initialize mean and covariance for cem
                if learning_type == 'cem':
                    # mean = q_learning.theta.copy()
                    if 'vgg' in args.predictor_fname:
                        mean = q_learning.theta.copy()
                        # mean = np.array([143294.94674524, 138389.72082692, 80881.93201423,
                        #                  113881.53147987, 138552.59285502, 99803.77807003,
                        #                  249041.32278199, 133521.75624396, 37892.42591511,
                        #                  32128.42220525, 16800.39153081, 19956.31642036,
                        #                  28371.43926481, 21594.88900791, 51275.90037066,
                        #                  24890.71477917, 19643.94740174, 26117.31888857,
                        #                  4637.23186999, 5544.28861698, 7616.91067105,
                        #                  5738.6285531, 12988.9744521, 6559.05641375,
                        #                  266.47325347, 1.])

                    else:
                        # camaro
                        # mean = np.array([0.10561821, 0.00000004, 0.00000003, 0.00000011, 0.00000011,
                        #                  0.0000001, 0.00000022, 0.0000004, 0.00000036, 5.25335634, 1.])
                        # mean = np.array([0.13384631, 1.62884428, 1.0217148 , 0.79811977, 1.16345499, 4.90281632, 2.44872154, 2.27626738, 7.04254259, 5.23223368, 1.])
                        mean = np.array([0.75357384, 2.49903853, 1.17271154, 2.06261216, 5.63346845, 0.93883819, 3.44667462, 3.70162917, 0.48607597, 5.30815239, 1.])  # ~90

                        # mitsubishi
                        # mean = np.array([3.14891472, 11.93392542, 13.96088388, 4.31307632,
                        #                  5.84146449, 23.46067648, 10.49607938, 4.66425788,
                        #                  12.77469346, 7.82040561, 1.])
                        # mean = np.array([3.50145672, 8.61674612, 3.70618353, 1.70249586,
                        #                  38.66924379, 6.75283425, 7.13665437, 5.07432362,
                        #                  13.83495851, 12.41419951, 1.])

                    cov = 100.0 * np.diag(np.r_[1 / feature_std ** 2, 1.0, 1.0])
                    # cov = np.eye(len(q_learning.theta)) * 0.1

            np.random.set_state(exploration_random_state)
            # rollout with noise and get (S, A, R, S') for q learning
            # TODO: use additive noise instead
            # eps = eps_init + sample_iter * (eps_final - eps_init) / 1e4 if sample_iter < 1e4 else eps_final
            # eps_greedy_pol.act_probs[:] = [1 - eps, eps]
            normal_scale = normal_scale_init + sample_iter * (normal_scale_final - normal_scale_init) / 1e4 if sample_iter < 1e4 else normal_scale_final
            print("action_normal_scale", normal_scale)

            if learning_type in ('q_learning', 'imitation_learning'):
                all_errors, states, actions, next_states, rewards, target_actions = rollout(predictor, env, greedy_pol,
                                                                            num_exploration_trajs, args.num_steps,
                                                                            action_normal_scale=normal_scale,
                                                                            visualize=args.visualize, image_visualizer=image_visualizer,
                                                                            target_distance=args.target_distance)
                if learning_type == 'q_learning':
                    bellman_error = q_learning.fit(states, actions, rewards, next_states)
                    bellman_errors.append(bellman_error)
                else:
                    training_loss = q_learning.fit_imitation(states, actions, target_actions)
                    bellman_errors.extend(training_loss)
            elif learning_type == 'cem':
                cem_thetas = []
                cem_esd_rewards = []
                for i in range(3 * len(q_learning.theta)):
                    print(i)
                    theta = np.random.multivariate_normal(mean, cov)
                    # clip negative values to zero except for bias
                    bias = theta[-2]
                    theta = np.abs(theta)
                    # theta[theta < 0] = 0.0
                    theta[-2] = bias
                    if not q_learning.learn_bias:
                        theta[-2] = q_learning.theta[-2]
                    if not q_learning.learn_lambda:
                        theta[-1] = q_learning.theta[-1]
                    cem_thetas.append(theta)
                    q_learning.theta = theta
                    all_errors, states, actions, next_states, rewards, target_actions = rollout(predictor, env, greedy_pol,
                                                                                num_exploration_trajs, args.num_steps,
                                                                                action_normal_scale=normal_scale,
                                                                                visualize=args.visualize, image_visualizer=image_visualizer,
                                                                                target_distance=args.target_distance)
                    esd_reward = (np.array(rewards).reshape((num_exploration_trajs, args.num_steps))
                                  * (q_learning.gamma ** np.arange(args.num_steps))[None, :]).sum(axis=1).mean()
                    cem_esd_rewards.append(esd_reward)
                top_inds = np.argsort(cem_esd_rewards)[:int(np.ceil(len(cem_esd_rewards) * 0.2))]  # top 20%
                top_cem_thetas = np.array(cem_thetas)[top_inds]
                mean = top_cem_thetas.mean(axis=0)
                cov = np.cov(top_cem_thetas.T)
                cov = np.diag(np.diag(cov))
                q_learning.theta = mean
            else:
                ValueError
            sample_iter += num_exploration_trajs * args.num_steps
            thetas.append(q_learning.theta.copy())

            # plotting
            rms_error_plotter.update([np.asarray(rms_positions), np.asarray(rms_rotations), rms_rewards, esd_rewards])
            if bellman_errors:
                bellman_error_plotter.update([bellman_errors])
    except Exception as e:
        print(e)
        import IPython as ipy; ipy.embed()

    if args.record_file:
        writer.finish()

    import IPython as ipy; ipy.embed()

    env.close()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
