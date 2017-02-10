from __future__ import division, print_function

import os

import numpy as np

from visual_dynamics import envs
from visual_dynamics import policies
from visual_dynamics.utils.config import ConfigObject, to_yaml
from visual_dynamics.utils.rl_util import do_rollouts, discount_returns


class Algorithm(ConfigObject):
    def run(self):
        raise NotImplementedError

    def iteration(self):
        raise NotImplementedError


class ServoingOptimizationAlgorithm(Algorithm):
    def __init__(self, env, servoing_pol, sampling_iters, num_trajs=None, num_steps=None, gamma=None, act_std=None,
                 iter_=0, thetas=None, mean_discounted_returns=None, learning_values=None, snapshot_interval=1,
                 snapshot_prefix='', plot=True):
        assert isinstance(env, envs.ServoingEnv)
        assert isinstance(servoing_pol, policies.ServoingPolicy)
        self.env = env
        self.sampling_iters = sampling_iters
        self.servoing_pol = servoing_pol
        self.num_trajs = 10 if num_trajs is None else num_trajs
        self.num_steps = 100 if num_steps is None else num_steps
        self.env.max_time_steps = self.num_steps
        self.gamma = 0.9 if gamma is None else gamma
        self.act_std = 0.2 if act_std is None else act_std
        self.noisy_pol = policies.AdditiveNormalPolicy(servoing_pol, env.action_space, None, act_std=self.act_std)

        self.iter_ = iter_
        self.thetas = [np.asarray(theta) for theta in thetas] if thetas is not None else []
        self.mean_discounted_returns = [np.asarray(ret) for ret in mean_discounted_returns] if mean_discounted_returns is not None else []
        self.learning_values = [np.asarray(value) for value in learning_values] if learning_values is not None else []
        self.snapshot_interval = snapshot_interval
        self.snapshot_prefix = snapshot_prefix
        self.plot = plot

    def run(self):
        if self.plot:
            fig_plotters = self.visualization_init()
            fig, plotters = fig_plotters[0], fig_plotters[1:]

        while self.iter_ <= self.sampling_iters:
            print("Iteration {} of {}".format(self.iter_, self.sampling_iters))
            self.thetas.append(self.servoing_pol.theta.copy())
            _, _, _, rewards = do_rollouts(self.env, self.servoing_pol, self.num_trajs, self.num_steps,
                                           seeds=np.arange(self.num_trajs))
            mean_discounted_return = np.mean(discount_returns(rewards, self.gamma))
            self.mean_discounted_returns.append(mean_discounted_return)
            print("    mean discounted return = {:.6f}".format(mean_discounted_return))

            if self.iter_ < self.sampling_iters:
                learning_value = self.iteration()
                self.learning_values.append(np.asarray(learning_value))

            if self.plot:
                self.visualization_update(*plotters)
                learning_fig_fname = self.get_snapshot_fname('_learning.pdf')
                fig.savefig(learning_fig_fname)

            if self.snapshot_interval and self.iter_ % self.snapshot_interval == 0:
                self.snapshot()

            self.iter_ += 1

    def visualization_init(self):
        raise NotImplementedError

    def visualization_update(self, return_plotter, learning_plotter):
        raise NotImplementedError

    def get_snapshot_fname(self, ext):
        algorithm_dir = os.path.split(self.snapshot_prefix)[0]
        if not os.path.exists(algorithm_dir):
            os.makedirs(algorithm_dir)
        return self.snapshot_prefix + '_iter_%s' % str(self.iter_) + ext

    def snapshot(self):
        model_fname = self.get_snapshot_fname('_model.yaml')
        algorithm_fname = self.get_snapshot_fname('_algorithm.yaml')
        print("Saving algorithm to file", algorithm_fname)
        with open(algorithm_fname, 'w') as algorithm_file:
            algorithm_config = self.get_config()
            algorithm_config['servoing_pol']['predictor']['pretrained_fname'] = self.servoing_pol.predictor.save_model(model_fname)
            to_yaml(algorithm_config, algorithm_file)

    def _get_config(self):
        config = super(ServoingOptimizationAlgorithm, self)._get_config()
        config.update({'env': self.env,
                       'servoing_pol': self.servoing_pol,
                       'sampling_iters': self.sampling_iters,
                       'num_trajs': self.num_trajs,
                       'num_steps': self.num_steps,
                       'gamma': self.gamma,
                       'act_std': self.act_std,
                       'iter_': self.iter_,
                       'thetas': [theta.tolist() for theta in self.thetas],
                       'mean_discounted_returns': [ret.tolist() for ret in self.mean_discounted_returns],
                       'learning_values': [value.tolist() for value in self.learning_values],
                       'snapshot_prefix': self.snapshot_prefix,
                       'plot': self.plot})
        return config
