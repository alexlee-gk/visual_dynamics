import yaml
import envs
import policy
import utils
from utils.transformer import ImageTransformer, NormalizerTransformer, CompositionTransformer
from envs import SimpleQuadPanda3dEnv

from citysim3d.utils import is_in_view


from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np


class SimpleQuadPanda3dRllabEnv(SimpleQuadPanda3dEnv, Env):
    def __init__(self):
        env_fname = '/home/alex/rll/visual_dynamics/config/environment/simplequad.yaml'
        pol_fname = '/home/alex/rll/visual_dynamics/config/policy/back.yaml'
        transformer_fname = '/home/alex/rll/visual_dynamics/config/transformer/transformer_32.yaml'

        with open(env_fname) as yaml_string:
            env_config = yaml.load(yaml_string)
            env = utils.from_config(env_config)

        with open(pol_fname) as yaml_string:
            policy_config = yaml.load(yaml_string)
            replace_config = {'env': env,
                              'action_space': env.action_space}
            try:
                replace_config['target_env'] = env.car_env
            except AttributeError:
                pass
            pol = utils.from_config(policy_config, replace_config=replace_config)
            assert len(pol.policies) == 2
            target_pol, random_pol = pol.policies
            assert isinstance(target_pol, policy.TargetPolicy)
            assert isinstance(random_pol, policy.RandomPolicy)

        self.__dict__.update(env.__dict__)

        with open(transformer_fname) as yaml_string:
            transformer_config = yaml.load(yaml_string)
            self.image_transformer = utils.from_config(transformer_config['image'],
                                                       replace_config={'space': env.observation_space.spaces[0]})

        self.target_pol = target_pol
        self._target_image = None
        self._t = 0
        self._T = 100

    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(6, 32, 32))

    @property
    def action_space(self):
        # TODO: use TranslationAxisAngleSpace
        return Box(low=self._action_space.low, high=self._action_space.high)

    def reset(self):
        # reset_state = self.target_pol.reset()
        # super(SimpleQuadPanda3dRllabEnv, self).reset(state=reset_state)
        super(SimpleQuadPanda3dRllabEnv, self).reset()
        self._t = 0
        obs = self.observe()
        self._target_image = self.image_transformer.preprocess(obs[0])
        return np.concatenate([self._target_image] * 2)

    def _get_target_direction(self):
        import utils.transformations as tf
        from policy.camera_target_policy import getTransform
        target_T = getTransform(self.target_pol.target_node)
        # agent transform in world coordinates
        agent_T = getTransform(self.target_pol.agent_node)
        # camera transform relative to the agent
        agent_to_camera_T = getTransform(self.target_pol.camera_node)
        # camera transform in world coordinates
        camera_T = agent_T.dot(agent_to_camera_T)
        # target relative to camera
        camera_to_target_T = tf.inverse_matrix(camera_T).dot(target_T)
        # target direction relative to camera
        target_direction = -camera_to_target_T[:3, 3]
        return target_direction

    def _get_image_formation_error(self, target_direction=None):
        if target_direction is None:
            target_direction = self._get_target_direction()
        x_error = (target_direction[0] / target_direction[2])
        y_error = (target_direction[1] / target_direction[2])
        z_error = (1.0 / target_direction[2] - 1.0 / self.target_pol.offset[2])
        # TODO: use z-value or norm of vector?
        return self.camera_sensor.focal_length * np.linalg.norm([x_error, y_error, z_error])

    def step(self, action):
        super(SimpleQuadPanda3dRllabEnv, self).step(action)
        obs = self.observe()
        image = self.image_transformer.preprocess(obs[0])
        target_direction = self._get_target_direction()
        self._t += 1
        in_view = is_in_view(self.camera_sensor.cam, self.car_env._car_local_node,
                             scale_size=self.image_transformer.transformers[0].scale_size,
                             crop_size=self.image_transformer.transformers[0].crop_size)
        done = self._t >= self._T or in_view or np.linalg.norm(target_direction) < 4.0
        reward = - self._get_image_formation_error(target_direction)
        if done:
            reward *= self._T - self._t + 1
        return Step(observation=np.concatenate([image, self._target_image]), reward=reward, done=done)


# class Vgg1SimpleQuadPanda3dRllabEnv(SimpleQuadPanda3dRllabEnv):
#     def __init__(self):
#         super(Vgg1SimpleQuadPanda3dRllabEnv, self).__init__()
#         transformer_fname = '/home/alex/rll/visual_dynamics/config/transformer/transformer_32_vgg.yaml'
#         predictor_fname = '/home/alex/rll/visual_dynamics/models/theano/multiscale_dilated_stdvgg_local_level1_scales012_transformer_32_vgg/adam_gamma0.9_losslevel1scales012_simplequad/_iter_10000_model.yaml'
#
#         with open(transformer_fname) as yaml_string:
#             transformer_config = yaml.load(yaml_string)
#             self.image_transformer = utils.from_config(transformer_config['image'], replace_config={'space': self._observation_space.spaces[0]})
#
#         self.predictor = utils.from_yaml(open(predictor_fname))
#
#     @property
#     def observation_space(self):
#         return Box(low=-1, high=1, shape=(64 * 3,))
#
#     def reset(self):
#         reset_state = self.target_pol.reset()
#         super(SimpleQuadPanda3dRllabEnv, self).reset(state=reset_state)
#         self._t = 0
#         obs = self.observe()
#         self._target_image = self.image_transformer.preprocess(obs[0])
#         self._target_feature_maps = self.predictor.feature(self._target_image, preprocessed=True)
#         feature_maps = self._target_feature_maps
#         observation = np.concatenate([((feature_map - target_feature_map) ** 2).sum(axis=(1, 2)) for feature_map, target_feature_map in zip(feature_maps, self._target_feature_maps)])
#         return observation
#
#     def step(self, action):
#         super(SimpleQuadPanda3dRllabEnv, self).step(action)
#         obs = self.observe()
#         image = self.image_transformer.preprocess(obs[0])
#         feature_maps = self.predictor.feature(image, preprocessed=True)
#         target_direction = self._get_target_direction()
#         self._t += 1
#         done = self._t >= self._T or \
#                np.all(self.image_transformer.preprocess(obs[1]) == self.image_transformer.preprocess(np.zeros_like(obs[1]))) or \
#                np.linalg.norm(target_direction) < 4.0
#         reward = - self._get_image_formation_error(target_direction)
#         if done:
#             reward *= self._T - self._t + 1
#         observation = np.concatenate([((feature_map - target_feature_map) ** 2).sum(axis=(1, 2)) for feature_map, target_feature_map in zip(feature_maps, self._target_feature_maps)])
#         return Step(observation=observation, reward=reward, done=done)
#
#     def render(self):
#         super(SimpleQuadPanda3dRllabEnv, self).render()


def main():
    from rllab.algos.trpo import TRPO
    from rllab.envs.normalized_env import normalize
    from rllab.policies.gaussian_conv_policy import GaussianConvPolicy
    from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
    from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

    env = SimpleQuadPanda3dRllabEnv()
    # env.reset()
    env = normalize(env)
    policy = GaussianConvPolicy(
        env_spec=env.spec,
        conv_filters=[32, 32],
        conv_filter_sizes=[3, 3],
        conv_strides=[2, 2],
        conv_pads=['same'] * 2,
        hidden_sizes=[32],
        std_conv_filters=[8, 8],
        std_conv_filter_sizes=[3, 3],
        std_conv_strides=[2, 2],
        std_conv_pads=['same'] * 2,
    )

    baseline = GaussianConvBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            step_size=0.01,
            normalize_inputs=True,
            normalize_outputs=True,
            hidden_sizes=[16, 16],
            conv_filters=[],
            conv_filter_sizes=[],
            conv_strides=[],
            conv_pads=[],
            batchsize=6000,
        )
    )

    # policy = GaussianMLPPolicy(
    #     env_spec=env.spec,
    #     hidden_sizes=(32, 32),
    # )

    # baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=100,
        n_itr=40,
        discount=0.9,
        step_size=0.01,
    )
    algo.train()


if __name__ == '__main__':
    main()
