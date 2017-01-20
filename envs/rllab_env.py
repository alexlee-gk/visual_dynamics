import numpy as np
import rllab
from rllab.envs.normalized_env import normalize

import utils
from envs import Env, ServoingEnv
from spaces import TranslationAxisAngleSpace


class RllabEnv(Env, rllab.envs.base.Env):
    """
    Wraps an environment to make it compatible with rllab

    Applies observation and action transformers (if any) and concatenates
    observations into single numpy array.

    Implementation is currently limited to only handle ServoingEnv.
    """
    def __init__(self, env, observation_name=None, transformers=None):
        assert isinstance(env, ServoingEnv)
        self._wrapped_env = env
        self._observation_name = observation_name or 'image'
        self._transformers = transformers or {self._observation_name: utils.Transformer(),
                                              'action': utils.NormalizerTransformer(env.action_space)}

    def _apply_transform_obs(self, obs):
        obs_transformer = self._transformers[self._observation_name]
        transformed_obs = [obs_transformer.preprocess(obs_) for obs_ in [obs[self._observation_name],
                                                                         obs['target_' + self._observation_name]]]
        return np.concatenate(transformed_obs)

    def reset(self, state=None):
        obs = self._wrapped_env.reset(state=state)
        return self._apply_transform_obs(obs)

    @property
    def action_space(self):
        action_transformer = self._transformers['action']
        action_space = self._wrapped_env.action_space
        assert isinstance(action_space, TranslationAxisAngleSpace)
        assert action_space.axis is not None
        transformed_low = action_transformer.preprocess(action_space.low)
        transformed_high = action_transformer.preprocess(action_space.high)
        transformed_shape = action_transformer.preprocess_shape(action_space.shape)
        if np.isscalar(transformed_low) and np.isscalar(transformed_high):
            return rllab.spaces.Box(transformed_low, transformed_high,
                                    shape=transformed_shape)
        else:
            return rllab.spaces.Box(transformed_low, transformed_high)

    @property
    def observation_space(self):
        obs_transformer = self._transformers[self._observation_name]
        obs_space = self._wrapped_env.observation_space.spaces[self._observation_name]
        assert self._wrapped_env.observation_space.spaces['target_' + self._observation_name] == obs_space
        transformed_low = obs_transformer.preprocess(obs_space.low * np.ones(obs_space.shape))
        transformed_high = obs_transformer.preprocess(obs_space.high * np.ones(obs_space.shape))
        assert np.all(transformed_low == transformed_low.min())
        assert np.all(transformed_high == transformed_high.max())
        transformed_low = transformed_low.min()
        transformed_high = transformed_high.max()
        transformed_shape = obs_transformer.preprocess_shape(obs_space.shape)
        return rllab.spaces.Box(transformed_low, transformed_high,
                                shape=(transformed_shape[0] * 2,) + transformed_shape[1:])

    def step(self, action):
        if 'action' in self._transformers:
            scaled_action = self._transformers['action'].deprocess(action)
        else:
            scaled_action = action
        obs, reward, done, info = self._wrapped_env.step(scaled_action)
        obs = self._apply_transform_obs(obs)
        if 'action' in self._transformers:
            action[...] = self._transformers['action'].preprocess(scaled_action)
        else:
            action[...] = scaled_action
        return obs, reward, done, info
