from __future__ import division, print_function

import _tkinter
import time

import argparse
import cv2
import lasagne
import matplotlib.animation as manimation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import theano
import theano.tensor as T
import yaml

import envs
import policy
import utils
from gui.grid_image_visualizer import GridImageVisualizer
from gui.loss_plotter import LossPlotter
from algorithms import Algorithm



class DqnAlgorithm(Algorithm):
    def __init__(self, env, servoing_pol, sampling_iters, gamma,
                 algorithm_iters,
                 l2_reg=0.0, learning_rate=0.01, max_batch_size=1000, max_memory_size=0):
        super(DqnAlgorithm, self).__init__(env, servoing_pol, sampling_iters, num_trajs, num_steps, gamma=gamma)
        self.servoing_pol = servoing_pol
        self.algorithm_iters = algorithm_iters
        self.gamma = gamma
        self.l2_reg = l2_reg
        self.max_batch_size = max_batch_size
        self.max_memory_size = max_memory_size
        self.memory_sars = [[] for _ in range(4)]

    def iteration(self):
        import IPython as ipy; ipy.embed()
        _, observations, actions, rewards = utils.do_rollouts(self.env, self.noisy_pol, self.num_trajs, self.num_steps,
                                                              seeds=np.arange(self.num_trajs))


    def update(self, *sars):
        sars = [[step_data for traj_data in data for step_data in traj_data] for data in sars]
        assert len(sars) == 4
        if self.max_memory_size:
            for memory_data, data in zip(self.memory_sars, sars):
                memory_data.extend(data)
                del memory_data[:(len(memory_data) - self.max_memory_size)]
            sars = self.memory_sars

        orig_batch_size = len(sars[0])
        for data in sars[1:]:
            assert len(data) == orig_batch_size
        batch_size = min(orig_batch_size, self.max_batch_size)

        theta = np.r_[self.servoing_pol.w, self.servoing_pol.lambda_, [self.servoing_pol.bias]]
        thetas = [theta.copy()]
        bellman_errors = []
        proxy_bellman_errors = []

        # TODO: use np.array for all data in sars

        for iter_ in range(self.algorithm_iters):
            if orig_batch_size > self.max_batch_size and iter_ == 0:  # TODO: only first iteration?
                choice = np.random.choice(orig_batch_size, self.max_batch_size)
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




class TheanoDqnAlgorithm(DqnAlgorithm):
    def __init__(self, env, servoing_pol, sampling_iters, algorithm_iters, gamma,
                 l2_reg=0.0, learning_rate=0.01, max_batch_size=1000, max_memory_size=0):
        super(TheanoDqnAlgorithm, self).__init__(env, servoing_pol, sampling_iters, num_trajs, num_steps, gamma=gamma)

        self.servoing_pol = servoing_pol
        self.algorithm_iters = algorithm_iters
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

        for iter_ in range(self.algorithm_iters):
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
            print("Iteration {} of {}".format(iter_, self.algorithm_iters))
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
