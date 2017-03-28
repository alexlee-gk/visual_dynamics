import matplotlib
matplotlib.use('Agg')

import argparse

import lasagne.nonlinearities as LN
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.policies.gaussian_conv_policy import GaussianConvPolicy

from visual_dynamics.envs import ServoingEnv, RllabEnv
from visual_dynamics.envs import SimpleQuadPanda3dEnv, GeometricCarPanda3dEnv
from visual_dynamics.policies import TheanoServoingPolicy
from visual_dynamics.policies.servoing_policy_network import ServoingPolicyNetwork
from visual_dynamics.policies.vgg_conv_network import VggConvNetwork
from visual_dynamics.spaces import TranslationAxisAngleSpace, BoxSpace
from visual_dynamics.utils.transformer import CompositionTransformer, ImageTransformer, OpsTransformer, NormalizerTransformer


stub(globals())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--algorithm_fname', type=str, default=None)
    parser.add_argument('--resume_from', type=str)
    parser.add_argument('--num_encoding_levels', type=int, default=5)
    parser.add_argument('--conv_filters', nargs='*', type=int, default=[16, 16])
    parser.add_argument('--conv_filter_sizes', nargs='*', type=int, default=[4, 4])
    parser.add_argument('--conv_strides', nargs='*', type=int, default=[2, 2])
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[32, 32])
    parser.add_argument('--init_std', type=float, default=0.2)
    parser.add_argument('--n_itr', type=int, default=500)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--custom_local_flags', type=str, default=None)
    parser.add_argument('--w_init', type=float, nargs='+', default=1.0)
    parser.add_argument('--lambda_init', type=float, nargs='+', default=1.0)
    args = parser.parse_args()

    env = SimpleQuadPanda3dEnv(action_space=TranslationAxisAngleSpace(low=[-10., -10., -10., -1.5707963267948966],
                                                                      high=[10., 10., 10., 1.5707963267948966],
                                                                      axis=[0., 0., 1.]),
                               sensor_names=['image'],
                               camera_size=[256, 256],
                               camera_hfov=26.007823885645635,
                               car_env_class=GeometricCarPanda3dEnv,
                               car_action_space=BoxSpace(low=[0., 0.],
                                                         high=[0., 0.]),
                               car_model_names=['mazda6', 'chevrolet_camaro', 'nissan_gt_r_nismo',
                                                'lamborghini_aventador', 'golf5'],
                               dt=0.1)
    # TODO: this should be true but it doesn't work with stubbed objects
    # assert env.get_config() == environment_config
    env = ServoingEnv(env)
    transformers = {'image': CompositionTransformer([ImageTransformer(scale_size=0.5),
                                                     OpsTransformer(transpose=(2, 0, 1))]),
                    'action': NormalizerTransformer(space=env.action_space)}
    env = RllabEnv(env, transformers=transformers)
    env = normalize(env)

    servoing_pol = TheanoServoingPolicy(predictor=args.predictor_fname, w=args.w_init, lambda_=args.lambda_init, algorithm_or_fname=args.algorithm_fname)
    mean_network = ServoingPolicyNetwork(env.observation_space.shape, env.action_space.flat_dim, servoing_pol)

    policy = GaussianConvPolicy(
        env_spec=env.spec,
        init_std=args.init_std,
        mean_network=mean_network,
        learn_std=False,
    )

    assert len(args.conv_filters) == len(args.conv_filter_sizes)
    assert len(args.conv_filters) == len(args.conv_strides)
    network_kwargs = dict(
        num_encoding_levels=args.num_encoding_levels,
        conv_filters=args.conv_filters,
        conv_filter_sizes=args.conv_filter_sizes,
        conv_strides=args.conv_strides,
        conv_pads=[0] * len(args.conv_filters),
        hidden_sizes=args.hidden_sizes,
        hidden_nonlinearity=LN.rectify,
        output_nonlinearity=None,
        name="mean_network"
    )

    conv_baseline_kwargs = dict(env_spec=env.spec,
                                regressor_args=dict(
                                    mean_network=VggConvNetwork(
                                        input_shape=env.observation_space.shape,
                                        output_dim=1,
                                        **network_kwargs),
                                    use_trust_region=True,
                                    step_size=args.step_size,
                                    normalize_inputs=True,
                                    normalize_outputs=True,
                                    hidden_sizes=None,
                                    conv_filters=None,
                                    conv_filter_sizes=None,
                                    conv_strides=None,
                                    conv_pads=None,
                                    batchsize=200,
                                    optimizer=PenaltyLbfgsOptimizer(n_slices=50),
                                ))
    baseline = GaussianConvBaseline(**conv_baseline_kwargs)

    algo = TRPO(env=env,
                policy=policy,
                baseline=baseline,
                batch_size=args.batch_size,
                max_path_length=100,
                n_itr=args.n_itr,
                discount=0.9,
                step_size=args.step_size,
                optimizer=ConjugateGradientOptimizer(num_slices=50),
                )

    if args.resume_from:
        run_experiment_lite(algo.train(),
                            snapshot_mode='gap',
                            snapshot_gap=1,
                            seed=args.seed,
                            custom_local_flags=args.custom_local_flags,
                            resume_from=args.resume_from)
    else:
        run_experiment_lite(algo.train(),
                            snapshot_mode='gap',
                            snapshot_gap=1,
                            seed=args.seed,
                            custom_local_flags=args.custom_local_flags)


if __name__ == '__main__':
    main()
