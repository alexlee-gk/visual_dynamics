import argparse

from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from visual_dynamics.envs import Point3dSimpleQuadPanda3dEnv, GeometricCarPanda3dEnv
from visual_dynamics.envs import ServoingEnv, RllabEnv
from visual_dynamics.spaces import TranslationAxisAngleSpace, BoxSpace
from visual_dynamics.utils.transformer import NormalizerTransformer
from visual_dynamics.utils.transformer import Transformer


stub(globals())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from', type=str)
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[32, 32])
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--n_itr', type=int, default=500)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--custom_local_flags', type=str, default=None)
    args = parser.parse_args()

    env = Point3dSimpleQuadPanda3dEnv(action_space=TranslationAxisAngleSpace(low=[-10., -10., -10., -1.5707963267948966],
                                                                             high=[10., 10., 10., 1.5707963267948966],
                                                                             axis=[0., 0., 1.]),
                                      sensor_names=[],
                                      camera_size=[256, 256],
                                      camera_hfov=26.007823885645635,
                                      car_env_class=GeometricCarPanda3dEnv,
                                      car_action_space=BoxSpace(low=[0., 0.],
                                                                high=[0., 0.]),
                                      car_model_names=['mazda6', 'chevrolet_camaro', 'nissan_gt_r_nismo',
                                                       'lamborghini_aventador', 'golf5'],
                                      dt=0.1)
    env = ServoingEnv(env)
    transformers = {'pos': Transformer(),
                    'action': NormalizerTransformer(space=env.action_space)}
    env = RllabEnv(env, observation_name='pos', transformers=transformers)
    env = normalize(env)

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

    if args.resume_from:
        run_experiment_lite(algo.train(),
                            snapshot_mode='gap',
                            snapshot_gap=100,
                            seed=args.seed,
                            custom_local_flags=args.custom_local_flags,
                            resume_from=args.resume_from)
    else:
        run_experiment_lite(algo.train(),
                            snapshot_mode='gap',
                            snapshot_gap=100,
                            custom_local_flags=args.custom_local_flags,
                            seed=args.seed)


if __name__ == '__main__':
    main()
