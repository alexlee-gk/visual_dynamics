from __future__ import division, print_function

import time

import lasagne
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from visual_dynamics.algorithms import ServoingOptimizationAlgorithm
from visual_dynamics.gui.loss_plotter import LossPlotter
from visual_dynamics.utils.generator import iterate_minibatches_generic
from visual_dynamics.utils.rl_util import do_rollouts, split_observations
from visual_dynamics.utils.time_util import tic, toc


class ServoingFittedQIterationAlgorithm(ServoingOptimizationAlgorithm):
    def __init__(self, env, servoing_pol, sampling_iters, algorithm_iters,
                 num_trajs=None, num_steps=None, gamma=None, act_std=None,
                 iter_=0, thetas=None, mean_discounted_returns=None,
                 learning_values=None, snapshot_interval=1, snapshot_prefix='',
                 plot=True, l2_reg=0.0, max_batch_size=1000, max_memory_size=0,
                 eps=None, fit_alpha_bias=True, opt_fit_bias=False):
        super(ServoingFittedQIterationAlgorithm, self).__init__(env, servoing_pol, sampling_iters,
                                                                num_trajs=num_trajs, num_steps=num_steps,
                                                                gamma=gamma, act_std=act_std,
                                                                iter_=iter_, thetas=thetas,
                                                                mean_discounted_returns=mean_discounted_returns,
                                                                learning_values=learning_values,
                                                                snapshot_interval=snapshot_interval,
                                                                snapshot_prefix=snapshot_prefix, plot=plot)
        self.algorithm_iters = algorithm_iters
        self.l2_reg = l2_reg
        self.max_batch_size = max_batch_size
        self.max_memory_size = max_memory_size
        self.eps = eps
        self.fit_alpha_bias = fit_alpha_bias
        self.opt_fit_bias = opt_fit_bias
        self.memory_sars = [], [], [], []
        self._bias = 0.0

    @property
    def theta(self):
        return self.servoing_pol.theta

    @theta.setter
    def theta(self, theta):
        self.servoing_pol.theta[...] = theta

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, bias):
        self._bias = float(bias)

    def iteration(self):
        _, observations, actions, rewards = do_rollouts(self.env, self.noisy_pol, self.num_trajs, self.num_steps,
                                                        seeds=np.arange(self.num_trajs))

        def preprocess_obs(obs):
            for obs_name, obs_ in obs.items():
                if obs_name.endswith('image'):
                    obs[obs_name], = self.servoing_pol.predictor.preprocess([obs_])
            return obs
        preprocess_action = self.servoing_pol.predictor.transformers['action'].preprocess
        observations = [[preprocess_obs(obs) for obs in observations_] for observations_ in observations]
        actions = [[preprocess_action(action) for action in actions_] for actions_ in actions]
        rewards = [[-reward for reward in rewards_] for rewards_ in rewards]  # use costs
        observations, observations_p = split_observations(observations)
        return self.update(observations, actions, rewards, observations_p)

    def update(self, *sars):
        assert len(sars) == 4
        sars = [[step_data for traj_data in data for step_data in traj_data] for data in sars]
        if self.max_memory_size:
            for memory_data, data in zip(self.memory_sars, sars):
                memory_data.extend(data)
                del memory_data[:(len(memory_data) - self.max_memory_size)]
            sars = self.memory_sars

        orig_batch_size = len(sars[0])
        for data in sars[1:]:
            assert len(data) == orig_batch_size
        batch_size = min(orig_batch_size, self.max_batch_size)

        bellman_errors = []
        proxy_bellman_errors = []
        iter_ = 0
        while iter_ <= self.algorithm_iters:
            if orig_batch_size > self.max_batch_size:
                choice = np.random.choice(orig_batch_size, self.max_batch_size)
                S, A, R, S_p = [[data[i] for i in choice] for data in sars]
            elif iter_ == 0:  # the data is fixed across iterations so only set at the first iteration
                S, A, R, S_p = sars

            # compute phi only if (S, A) has changed
            if orig_batch_size > self.max_batch_size or iter_ == 0:
                assert len(S) == batch_size
                A = np.asarray(A)
                R = np.asarray(R)
                tic()
                phi = self.servoing_pol.phi(S, A, preprocessed=True)
                toc("\tphi")

            # compute Q_sample
            tic()
            if self.gamma == 0:
                Q_sample = R
            else:
                A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
                V_p = phi_p.dot(self.theta) + self.bias
                Q_sample = R + self.gamma * V_p
            toc("\tQ_sample")

            if self.fit_alpha_bias:
                old_objective_value = (1 / 2.) * ((phi.dot(self.theta) + self.bias - Q_sample) ** 2).mean() + \
                                      (self.l2_reg / 2.) * (self.theta ** 2).sum()
                L = np.c_[phi.dot(self.theta) - self.gamma * phi_p.dot(self.theta), (1 - self.gamma) * np.ones(batch_size)]
                D = np.diag([batch_size * self.l2_reg * (self.theta ** 2).sum(), 0])

                lsq_A_fit = L.T.dot(L) + D
                lsq_b_fit = L.T.dot(R)
                alpha, bias = np.linalg.solve(lsq_A_fit, lsq_b_fit)
                if alpha <= 0:
                    print("\tUnconstrained alpha is negative (alpha = %.2f). Solving constrained optimization." % alpha)
                    import cvxpy
                    x_var = cvxpy.Variable(2)
                    objective = cvxpy.Minimize((1 / 2.) * cvxpy.sum_squares(lsq_A_fit * x_var - lsq_b_fit))
                    constraints = [0.0 <= x_var[0]]
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
                        print("\tUnable to solve constrained optimization. Setting alpha = 0, solving for bias.")
                        alpha = 0.0
                        bias = R.mean() / (1 - self.gamma)
                self.theta *= alpha
                self.bias = bias
                if self.gamma == 0:
                    Q_sample = R
                else:
                    # the policy shouldn't have changed (because alpha >= 0), so phi_p doesn't need to be updated
                    V_p = phi_p.dot(self.theta) + self.bias
                    Q_sample = R + self.gamma * V_p
                new_objective_value = (1 / 2.) * ((phi.dot(self.theta) + self.bias - Q_sample) ** 2).mean() + \
                                      (self.l2_reg / 2.) * (self.theta ** 2).sum()
                if new_objective_value > old_objective_value:
                    print("\tObjective increased from %.6f to %.6f after fitting alpha and bias." % (old_objective_value, new_objective_value))

            objective_value = (1 / 2.) * ((phi.dot(self.theta) + self.bias - Q_sample) ** 2).mean() + \
                              (self.l2_reg / 2.) * (self.theta ** 2).sum()
            print("\tIteration {} of {}".format(iter_, self.algorithm_iters))
            print("\t    bellman error = {:.6f}".format(objective_value))
            bellman_errors.append(objective_value)

            if iter_ < self.algorithm_iters:  # don't take an update step after the last iteration
                proxy_bellman_error = self.fqi_update(S, A, R, S_p, phi=phi, Q_sample=Q_sample)
                objective_increased = proxy_bellman_error > objective_value
                print(u"\t                  {} {:.6f}".format(u"\u2191" if objective_increased else u"\u2193", proxy_bellman_error))
                proxy_bellman_errors.append(proxy_bellman_error)

            iter_ += 1

        return bellman_errors

    def fqi_update(self, S, A, R, S_p, phi=None, Q_sample=None):
        batch_size = len(S)
        if phi is None:
            tic()
            A = np.asarray(A)
            R = np.asarray(R)
            tic()
            phi = self.servoing_pol.phi(S, A, preprocessed=True)
            toc("\tphi")

        if Q_sample is None:
            tic()
            if self.gamma == 0:
                Q_sample = R
            else:
                A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
                V_p = phi_p.dot(self.theta) + self.bias
                Q_sample = R + self.gamma * V_p
            toc("\tQ_sample")

        tic()
        import cvxpy
        theta_var = cvxpy.Variable(self.theta.shape[0])
        bias_var = cvxpy.Variable(1)
        assert len(phi) == batch_size
        scale = 1.0
        solved = False
        while not solved:
            if self.opt_fit_bias:
                objective = cvxpy.Minimize(
                    (1 / 2.) * cvxpy.sum_squares(
                        (phi * np.sqrt(scale / batch_size)) * theta_var + bias_var * np.sqrt(scale / batch_size) -
                        ((Q_sample + (bias_var - self.bias) * self.gamma) * np.sqrt(scale / batch_size))) +
                    (self.l2_reg / 2.) * scale * cvxpy.sum_squares(theta_var))  # no regularization on bias
            else:
                objective = cvxpy.Minimize(
                    (1 / 2.) * cvxpy.sum_squares(
                        (phi * np.sqrt(scale / batch_size)) * theta_var + bias_var * np.sqrt(scale / batch_size) -
                        (Q_sample * np.sqrt(scale / batch_size))) +
                    (self.l2_reg / 2.) * scale * cvxpy.sum_squares(theta_var))  # no regularization on bias
            constraints = [0 <= theta_var]  # no constraint on bias

            if self.eps is not None:
                constraints.append(cvxpy.sum_squares(theta_var - self.theta) <= (len(self.theta) * self.eps))

            prob = cvxpy.Problem(objective, constraints)
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
                scale *= 0.1
                print("Unable to solve FQI optimization. Reformulating objective with a scale of %f." % scale)

        self.theta = np.squeeze(np.array(theta_var.value), axis=1)
        self.bias = bias_var.value
        proxy_bellman_error = objective.value / scale
        toc("\tcvxpy")
        return proxy_bellman_error

    def visualization_init(self):
        fig = plt.figure(figsize=(12, 6), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(self.servoing_pol.predictor.name)
        gs = gridspec.GridSpec(1, 2)
        plt.show(block=False)

        return_plotter = LossPlotter(fig, gs[0], format_dicts=[dict(linewidth=2)] * 2, ylabel='mean discounted returns')
        return_major_locator = MultipleLocator(1)
        return_major_formatter = FormatStrFormatter('%d')
        return_minor_locator = MultipleLocator(1)
        return_plotter._ax.xaxis.set_major_locator(return_major_locator)
        return_plotter._ax.xaxis.set_major_formatter(return_major_formatter)
        return_plotter._ax.xaxis.set_minor_locator(return_minor_locator)

        learning_plotter = LossPlotter(fig, gs[1],
                                       format_strings=['', 'r--'],
                                       format_dicts=[dict(linewidth=2)] * 2,
                                       ylabel='Bellman errors', yscale='log')
        # learning_plotter._ax.set_ylim((10.0, 110000))
        learning_major_locator = MultipleLocator(1)
        learning_major_formatter = FormatStrFormatter('%d')
        learning_minor_locator = MultipleLocator(0.2)
        learning_plotter._ax.xaxis.set_major_locator(learning_major_locator)
        learning_plotter._ax.xaxis.set_major_formatter(learning_major_formatter)
        learning_plotter._ax.xaxis.set_minor_locator(learning_minor_locator)
        return fig, return_plotter, learning_plotter

    def visualization_update(self, return_plotter, learning_plotter):
        return_plotter.update([self.mean_discounted_returns])
        bellman_errors = self.learning_values
        final_bellman_errors = [bellman_errors_[-1] for bellman_errors_ in bellman_errors]
        final_iters = list(range(1, self.iter_ + 2))
        bellman_errors_flat = [bellman_error for bellman_errors_ in bellman_errors
                               for bellman_error in (list(bellman_errors_) + [None])]
        iters = np.linspace(0, self.iter_ + 1, (self.algorithm_iters + 1) * (self.iter_ + 1) + 1)[1:]
        iters_flat = [iter_ for iters_ in iters.reshape((-1, self.sampling_iters + 1))
                      for iter_ in (iters_.tolist() + [None])]
        learning_plotter.update([final_bellman_errors, bellman_errors_flat],
                                [final_iters[:len(final_bellman_errors)], iters_flat[:len(bellman_errors_flat)]])

    def _get_config(self):
        config = super(ServoingFittedQIterationAlgorithm, self)._get_config()
        config.update({'algorithm_iters': self.algorithm_iters,
                       'l2_reg': self.l2_reg,
                       'max_batch_size': self.max_batch_size,
                       'max_memory_size': self.max_memory_size,
                       'eps': self.eps,
                       'fit_alpha_bias': self.fit_alpha_bias,
                       'opt_fit_bias': self.opt_fit_bias})
        return config


class ServoingPcaFittedQIterationAlgorithm(ServoingFittedQIterationAlgorithm):
    def __init__(self, pca, *args, **kwargs):
        super(ServoingPcaFittedQIterationAlgorithm, self).__init__(*args, **kwargs)
        self.pca = pca
        self._z = np.zeros(self.pca.components_.shape[0])
        self.theta = self.servoing_pol.theta

    @property
    def theta(self):
        return self.z.dot(self.pca.components_) + self.pca.mean_

    @theta.setter
    def theta(self, theta):
        self.z = (theta - self.pca.mean_).dot(self.pca.components_.T)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        self._z[...] = z
        self.servoing_pol.theta[...] = self.theta

    def fqi_update(self, S, A, R, S_p, phi=None, Q_sample=None):
        try:
            batch_size = len(S)
            if phi is None:
                tic()
                A = np.asarray(A)
                R = np.asarray(R)
                tic()
                phi = self.servoing_pol.phi(S, A, preprocessed=True)
                toc("\tphi")

            if Q_sample is None:
                tic()
                if self.gamma == 0:
                    Q_sample = R
                else:
                    A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                    phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
                    V_p = phi_p.dot(self.theta) + self.bias
                    Q_sample = R + self.gamma * V_p
                toc("\tQ_sample")

            tic()
            import cvxpy
            z_var = cvxpy.Variable(self.z.shape[0])
            theta_var = self.pca.components_.T * z_var + self.pca.mean_
            bias_var = cvxpy.Variable(1)
            assert len(phi) == batch_size
            scale = 1.0
            solved = False
            while not solved:
                if self.opt_fit_bias:
                    objective = cvxpy.Minimize(
                        (1 / 2.) * cvxpy.sum_squares(
                            (phi * np.sqrt(scale / batch_size)) * theta_var + bias_var * np.sqrt(scale / batch_size) -
                            ((Q_sample + (bias_var - self.bias) * self.gamma) * np.sqrt(scale / batch_size))) +
                        (self.l2_reg / 2.) * scale * cvxpy.sum_squares(theta_var))  # no regularization on bias
                else:
                    objective = cvxpy.Minimize(
                        (1 / 2.) * cvxpy.sum_squares(
                            (phi * np.sqrt(scale / batch_size)) * theta_var + bias_var * np.sqrt(scale / batch_size) -
                            (Q_sample * np.sqrt(scale / batch_size))) +
                        (self.l2_reg / 2.) * scale * cvxpy.sum_squares(theta_var))  # no regularization on bias
                constraints = [0 <= theta_var]  # no constraint on bias

                if self.eps is not None:
                    constraints.append(cvxpy.sum_squares(theta_var - self.theta) <= (len(self.theta) * self.eps))

                prob = cvxpy.Problem(objective, constraints)
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
                    scale *= 0.1
                    print("Unable to solve FQI optimization. Reformulating objective with a scale of %f." % scale)

            self.z = np.squeeze(np.array(z_var.value), axis=1)
            self.bias = bias_var.value
            proxy_bellman_error = objective.value / scale
            toc("\tcvxpy")
        except Exception as e:
            import IPython as ipy; ipy.embed()
        return proxy_bellman_error


class ServoingSgdFittedQIterationAlgorithm(ServoingFittedQIterationAlgorithm):
    def __init__(self, env, servoing_pol, sampling_iters, algorithm_iters,
                 num_trajs=None, num_steps=None, gamma=None, act_std=None,
                 iter_=0, thetas=None, mean_discounted_returns=None,
                 learning_values=None, snapshot_interval=1, snapshot_prefix='',
                 plot=True, l2_reg=0.0, max_batch_size=1000, max_memory_size=0,
                 fit_alpha_bias=True, opt_fit_theta_bias=False, sgd_iters=1000,
                 batch_size=100, learning_rate=0.01):
        ServoingOptimizationAlgorithm.__init__(self, env, servoing_pol, sampling_iters,
                                               num_trajs=num_trajs, num_steps=num_steps,
                                               gamma=gamma, act_std=act_std,
                                               iter_=iter_, thetas=thetas,
                                               mean_discounted_returns=mean_discounted_returns,
                                               learning_values=learning_values,
                                               snapshot_interval=snapshot_interval,
                                               snapshot_prefix=snapshot_prefix, plot=plot)
        self.algorithm_iters = algorithm_iters
        self.l2_reg = l2_reg
        self.max_batch_size = max_batch_size
        self.max_memory_size = max_memory_size
        self.fit_alpha_bias = fit_alpha_bias
        self.opt_fit_theta_bias = opt_fit_theta_bias
        self.sgd_iters = sgd_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory_sars = [], [], [], []
        self._bias = 0.0
        self.sgd_train_fn = None
        self.sqrt_theta_var = theano.shared(np.sqrt(self.theta).astype(theano.config.floatX), name='sqrt_theta')
        self.bias_var = theano.shared(self.bias)

    def fqi_update(self, S, A, R, S_p, phi=None, Q_sample=None):
        if self.sgd_train_fn is None:
            self.sgd_train_fn = self._compile_sgd_train_fn()

        self.sqrt_theta_var.set_value(np.sqrt(self.theta).astype(theano.config.floatX))
        self.bias_var.set_value(self.bias)

        data = [S, A, R, S_p]
        if not self.opt_fit_theta_bias and Q_sample is None:
            tic()
            if self.gamma == 0:
                Q_sample = R
            else:
                A_p = self.servoing_pol.pi(S_p, preprocessed=True)
                phi_p = self.servoing_pol.phi(S_p, A_p, preprocessed=True)
                V_p = phi_p.dot(self.theta) + self.bias
                Q_sample = R + self.gamma * V_p
            toc("\tQ_sample")
        if not self.opt_fit_theta_bias:
            data += [Q_sample]

        data_gen = iterate_minibatches_generic(data, batch_size=self.batch_size, shuffle=True)
        for sgd_iter, batch_data in zip(range(self.sgd_iters), data_gen):
            batch_S, batch_A, batch_R, batch_S_p = batch_data[:4]
            batch_image = np.array([obs['image'] for obs in batch_S])
            batch_target_image = np.array([obs['target_image'] for obs in batch_S])
            batch_actions = np.asarray(batch_A)
            batch_rewards = np.asarray(batch_R)
            batch_next_image = np.array([obs['image'] for obs in batch_S_p])
            batch_next_target_image = np.array([obs['target_image'] for obs in batch_S_p])
            action_lin = np.zeros(self.servoing_pol.action_space.shape)
            u_lin = self.servoing_pol.action_transformer.preprocess(action_lin)
            batch_u_lin = np.array([u_lin] * len(batch_S))
            if self.opt_fit_theta_bias:
                train_loss = float(self.sgd_train_fn(batch_image, batch_target_image,
                                                     batch_actions, batch_rewards,
                                                     batch_next_image, batch_next_target_image,
                                                     batch_u_lin, self.servoing_pol.alpha, self.learning_rate))
            else:
                batch_Q_sample = batch_data[4]
                train_loss = float(self.sgd_train_fn(batch_image, batch_target_image,
                                                     batch_actions, batch_Q_sample,
                                                     batch_u_lin, self.servoing_pol.alpha, self.learning_rate))
            if sgd_iter % 20 == 0:
                print("Iteration {} of {}".format(sgd_iter, self.sgd_iters))
                print("    training loss = {:.6f}".format(train_loss))

        self.theta = self.sqrt_theta_var.get_value() ** 2
        self.bias = self.bias_var.get_value()
        return train_loss

    def _compile_sgd_train_fn(self):
        X_var, U_var, X_target_var, U_lin_var, alpha_var = self.servoing_pol.input_vars
        X_next_var = T.tensor4('x_next')
        X_next_target_var = T.tensor4('x_next_target')
        R_var = T.vector('R')

        theta_var = self.sqrt_theta_var ** 2
        w_var = theta_var[:len(self.servoing_pol.w)]
        lambda_var = theta_var[len(self.servoing_pol.w):]
        bias_var = self.bias_var

        # depends on X_var, X_target_var and U_var
        phi_var = self.servoing_pol._get_phi_var()

        if self.opt_fit_theta_bias:
            # depends on X_var and X_target_var
            pi_var = self.servoing_pol._get_pi_var().astype(theano.config.floatX)
            pi_var = theano.clone(pi_var, replace=dict(zip(self.servoing_pol.param_vars, [w_var, lambda_var])))

            # depends on X_next_var and X_next_target_var
            pi_p_var = theano.clone(pi_var, replace={X_var: X_next_var, X_target_var: X_next_target_var})
            phi_p_var = theano.clone(phi_var, replace={X_var: X_next_var, X_target_var: X_next_target_var, U_var: pi_p_var})
            V_p_var = T.dot(phi_p_var, self.theta_var) + self.bias_var
            Q_sample_var = R_var + self.gamma * V_p_var
        else:
            Q_sample_var = T.vector('Q_sample')

        # training loss
        loss_var = ((T.dot(phi_var, theta_var) + bias_var - Q_sample_var) ** 2).mean() / 2.
        loss_var += (self.l2_reg / 2.) * (theta_var ** 2).sum()

        # training updates
        learning_rate_var = theano.tensor.scalar(name='learning_rate')
        updates = lasagne.updates.adam(loss_var,
                                       [self.sqrt_theta_var, self.bias_var],
                                       learning_rate=learning_rate_var)

        input_vars = [X_var, X_target_var, U_var]
        if self.opt_fit_theta_bias:
            input_vars += [R_var, X_next_var, X_next_target_var]
        else:
            input_vars += [Q_sample_var]
        input_vars += [U_lin_var, alpha_var, learning_rate_var]

        start_time = time.time()
        print("Compiling SGD training function...")
        sgd_train_fn = theano.function(input_vars, loss_var,
                                       updates=updates,
                                       on_unused_input='warn', allow_input_downcast=True)
        print("... finished in %.2f s" % (time.time() - start_time))
        return sgd_train_fn

    def _get_config(self):
        config = ServoingOptimizationAlgorithm._get_config(self)
        config.update({'algorithm_iters': self.algorithm_iters,
                       'l2_reg': self.l2_reg,
                       'max_batch_size': self.max_batch_size,
                       'max_memory_size': self.max_memory_size,
                       'fit_alpha_bias': self.fit_alpha_bias,
                       'opt_fit_theta_bias': self.opt_fit_theta_bias,
                       'sgd_iters': self.sgd_iters,
                       'batch_size': self.batch_size,
                       'learning_rate': self.learning_rate})
        return config
