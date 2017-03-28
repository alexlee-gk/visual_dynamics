from __future__ import division, print_function

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from visual_dynamics.algorithms import Algorithm, ServoingOptimizationAlgorithm
from visual_dynamics.gui.loss_plotter import LossPlotter
from visual_dynamics.utils.rl_util import do_rollouts, discount_returns


class CrossEntropyMethodAlgorithm(Algorithm):
    """
    modified from https://github.com/openai/gym/blob/master/examples/agents/cem.py
    """
    def __init__(self, f, th_mean, batch_size, sampling_iters, elite_frac=0.2, initial_std=1.0, th_std=None, th_low=None, th_high=None):
        """
        Generic implementation of the cross-entropy method for minimizing a black-box function
        f: a function mapping from vector -> scalar
        th_mean: initial mean over input distribution
        batch_size: number of samples of theta to evaluate per batch
        sampling_iters: number of batches
        elite_frac: each batch, select this fraction of the top-performing samples
        initial_std: initial standard deviation over parameter vectors
        """
        self.f = f
        self.th_mean = th_mean
        self.batch_size = batch_size
        self.sampling_iters = sampling_iters
        self.elite_frac = elite_frac
        self.initial_std = initial_std
        self.th_low = th_low
        self.th_high = th_high
        self.n_elite = int(np.round(batch_size * elite_frac))
        if th_std is not None:
            self.th_std = np.asarray(th_std)
        else:
            if isinstance(initial_std, np.ndarray) and initial_std.shape == th_mean.shape:
                self.th_std = initial_std
            else:
                self.th_std = np.ones_like(th_mean) * initial_std
        self.use_truncnorm = th_low is not None or th_high is not None
        if self.use_truncnorm:
            self.th_low = -np.inf if th_low is None else th_low
            self.th_high = np.inf if th_high is None else th_high

    def iteration(self):
        if self.use_truncnorm:
            tn = scipy.stats.truncnorm((self.th_low - self.th_mean) / self.th_std, (self.th_high - self.th_mean) / self.th_std)
            ths = np.array([self.th_mean + dth for dth in self.th_std[None, :] * tn.rvs((self.batch_size, self.th_mean.size))])
            assert np.all(self.th_low <= ths) and np.all(ths <= self.th_high)
        else:
            ths = np.array([self.th_mean + dth for dth in self.th_std[None, :] * np.random.randn(self.batch_size, self.th_mean.size)])
        ys = np.array([self.f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:self.n_elite]
        elite_ths = ths[elite_inds]
        self.th_mean = elite_ths.mean(axis=0)
        self.th_std = elite_ths.std(axis=0)
        return ys.mean()


class ServoingCrossEntropyMethodAlgorithm(ServoingOptimizationAlgorithm, CrossEntropyMethodAlgorithm):
    def __init__(self, env, servoing_pol, sampling_iters, batch_size=None,
                 elite_frac=0.2, initial_std=None, th_std=None,
                 num_trajs=None, num_steps=None, gamma=None, act_std=None,
                 iter_=0, thetas=None, mean_returns=None, std_returns=None,
                 mean_discounted_returns=None, std_discounted_returns=None,
                 learning_values=None, snapshot_prefix='', plot=True, skip_validation=False, unweighted_features=False):
        servoing_pol.unweighted_features = unweighted_features
        ServoingOptimizationAlgorithm.__init__(self, env, servoing_pol, sampling_iters,
                                               num_trajs=num_trajs, num_steps=num_steps,
                                               gamma=gamma, act_std=act_std,
                                               iter_=iter_, thetas=thetas,
                                               mean_returns=mean_returns,
                                               std_returns=std_returns,
                                               mean_discounted_returns=mean_discounted_returns,
                                               std_discounted_returns=std_discounted_returns,
                                               learning_values=learning_values,
                                               snapshot_prefix=snapshot_prefix,
                                               skip_validation=skip_validation, plot=plot)
        CrossEntropyMethodAlgorithm.__init__(self,
                                             f=self._noisy_evaluation,
                                             th_mean=self.servoing_pol.theta,
                                             batch_size=batch_size or (3 * len(self.servoing_pol.theta)),
                                             sampling_iters=sampling_iters,
                                             elite_frac=elite_frac,
                                             initial_std=initial_std if initial_std is not None else self.servoing_pol.theta,
                                             th_std=th_std,
                                             th_low=0.0)

    def _noisy_evaluation(self, theta):
        # save servoing parameters
        # w, lambda_ = self.servoing_pol.w.copy(), self.servoing_pol.lambda_.copy()
        curr_theta = self.servoing_pol.theta.copy()
        self.servoing_pol.theta = theta
        rewards = do_rollouts(self.env, self.noisy_pol, self.num_trajs, self.num_steps,
                              seeds=np.arange(self.num_trajs), ret_rewards_only=True)
        # restore servoing parameters
        # self.servoing_pol.w, self.servoing_pol.lambda_ = w, lambda_
        self.servoing_pol.theta = curr_theta
        return np.mean(discount_returns(rewards, 1.0))  # using undiscounted returns for noisy evaluation

    def iteration(self):
        mean_evaluation_values = CrossEntropyMethodAlgorithm.iteration(self)
        self.servoing_pol.theta = self.th_mean
        return mean_evaluation_values

    def visualization_init(self):
        fig = plt.figure(figsize=(12, 6), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(self.servoing_pol.predictor.name)
        gs = gridspec.GridSpec(1, 2)
        plt.show(block=False)

        return_plotter = LossPlotter(fig, gs[0],
                                     format_dicts=[dict(linewidth=2)] * 2,
                                     labels=['mean returns / 10', 'mean discounted returns'],
                                     ylabel='returns')
        return_major_locator = MultipleLocator(1)
        return_major_formatter = FormatStrFormatter('%d')
        return_minor_locator = MultipleLocator(1)
        return_plotter._ax.xaxis.set_major_locator(return_major_locator)
        return_plotter._ax.xaxis.set_major_formatter(return_major_formatter)
        return_plotter._ax.xaxis.set_minor_locator(return_minor_locator)

        learning_plotter = LossPlotter(fig, gs[1], format_dicts=[dict(linewidth=2)] * 2, ylabel='mean evaluation values')
        return fig, return_plotter, learning_plotter

    def visualization_update(self, return_plotter, learning_plotter):
        return_plotter.update([np.asarray(self.mean_returns) * 0.1, self.mean_discounted_returns])
        learning_plotter.update([self.learning_values])

    def _get_config(self):
        config = super(ServoingCrossEntropyMethodAlgorithm, self)._get_config()
        config.update({'batch_size': self.batch_size,
                       'elite_frac': self.elite_frac,
                       'initial_std': self.initial_std,
                       'th_std': self.th_std,
                       'unweighted_features': self.servoing_pol.unweighted_features})
        return config
