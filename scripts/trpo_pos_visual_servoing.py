import argparse
import citysim3d.utils.panda3d_util as putil
import numpy as np
from citysim3d.envs import SimpleQuadPanda3dEnv
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from envs import ServoingEnv, RllabEnv
from spaces import BoxSpace, TranslationAxisAngleSpace


class Point3dSimpleQuadPanda3dEnv(SimpleQuadPanda3dEnv):
    def __init__(self, *args, **kwargs):
        super(Point3dSimpleQuadPanda3dEnv, self).__init__(*args, **kwargs)

        self._observation_space.spaces.clear()
        self._observation_space.spaces['pos'] = BoxSpace(-np.inf, np.inf, shape=(1, 3))

    def observe(self):
        return {'pos': np.array(self.car_node.getTransform(self.camera_node).getPos())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[32, 32])
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--n_itr', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--normalize_obs', action='store_true')
    parser.add_argument('--normalize_reward', action='store_true')
    args = parser.parse_args()

    action_space = TranslationAxisAngleSpace(low=[-20, -10, -10, -np.pi/2],
                                             high=[20, 10, 10, np.pi/2],
                                             axis=[0, 0, 1])
    camera_size, camera_hfov = putil.scale_crop_camera_parameters((640, 480), 60.0, scale_size=0.125, crop_size=(32, 32))
    env = Point3dSimpleQuadPanda3dEnv(action_space, camera_size=camera_size, camera_hfov=camera_hfov,
                                      car_model_names=['camaro2', 'mazda6', 'sport', 'kia_rio_blue', 'kia_rio_red', 'kia_rio_white'])
    env = ServoingEnv(env, max_time_steps=100)
    env = RllabEnv(env, observation_name='pos')
    env = normalize(env, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=args.hidden_sizes,
        init_std=args.init_std,
    )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            step_size=args.step_size,
            normalize_inputs=True,
            normalize_outputs=True,
            hidden_sizes=args.hidden_sizes,
            batchsize=args.batch_size * 10,
        )
    )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.batch_size,
        max_path_length=100,
        n_itr=args.n_itr,
        discount=0.9,
        step_size=args.step_size,
    )
    algo.train()
    import IPython as ipy; ipy.embed()


if __name__ == '__main__':
    main()
