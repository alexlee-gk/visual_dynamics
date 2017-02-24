import argparse
from collections import OrderedDict

import lasagne
import lasagne.init as LI
import lasagne.layers as L
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.nonlinearities as nl
import numpy as np
import numpy as np
import theano
import theano
import theano.tensor as T
import theano.tensor as T
import yaml
from citysim3d.utils import panda3d_util as putil
from lasagne import init
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.core.network import ConvNetwork
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.policies.gaussian_conv_policy import GaussianConvPolicy
from rllab.regressors.gaussian_conv_regressor import GaussianConvRegressor
from rllab.sampler.utils import rollout

from visual_dynamics import envs
from visual_dynamics.envs import ServoingEnv, RllabEnv
from visual_dynamics.envs import SimpleQuadPanda3dEnv, GeometricCarPanda3dEnv
from visual_dynamics.policies import TheanoServoingPolicy
from visual_dynamics.policies.servoing_policy_network import ServoingPolicyNetwork
from visual_dynamics.policies.vgg_conv_network import VggConvNetwork
from visual_dynamics.predictors import layers_theano as LT
from visual_dynamics.spaces import TranslationAxisAngleSpace, BoxSpace
from visual_dynamics.utils.config import from_config, from_yaml
from visual_dynamics.utils.transformer import OpsTransformer, NormalizerTransformer
from visual_dynamics.utils.transformer import transfer_image_transformer



stub(globals())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from', type=str)
    parser.add_argument('--encoding_levels', type=int, nargs='+')
    parser.add_argument('--num_encoding_levels', type=int, default=5)
    parser.add_argument('--conv_filters', nargs='*', type=int, default=[16, 16])
    parser.add_argument('--conv_filter_sizes', nargs='*', type=int, default=[4, 4])
    parser.add_argument('--conv_strides', nargs='*', type=int, default=[2, 2])
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[32, 32])
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--n_itr', type=int, default=500)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    camera_size, camera_hfov = putil.scale_crop_camera_parameters((640, 480), 60.0, scale_size=0.5, crop_size=(128,) * 2)
    env = SimpleQuadPanda3dEnv(action_space=TranslationAxisAngleSpace(low=[-20., -10., -10., -1.57079633],
                                                                      high=[20., 10., 10., 1.57079633],
                                                                      axis=[0., 0., 1.]),
                               sensor_names=['image'],
                               camera_size=camera_size,
                               camera_hfov=camera_hfov,
                               offset=[0., -25.98076211, 15.],
                               car_env_class=GeometricCarPanda3dEnv,
                               car_action_space=BoxSpace(low=[0., 0.],
                                                         high=[0., 0.]),
                               car_model_names=['camaro2', 'mazda6', 'sport', 'kia_rio_blue', 'kia_rio_red', 'kia_rio_white'],
                               dt=0.1)
    env = ServoingEnv(env)
    transformers = {'image': OpsTransformer(transpose=(2, 0, 1)),
                    'action': NormalizerTransformer(space=env.action_space)}
    env = RllabEnv(env, transformers=transformers)
    env = normalize(env)

    assert len(args.conv_filters) == len(args.conv_filter_sizes)
    assert len(args.conv_filters) == len(args.conv_strides)
    network_kwargs = dict(
        encoding_levels=args.encoding_levels,
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
    mean_network = VggConvNetwork(
        input_shape=env.observation_space.shape,
        output_dim=env.action_space.flat_dim,
        **network_kwargs)

    policy = GaussianConvPolicy(
        env_spec=env.spec,
        init_std=args.init_std,
        mean_network=mean_network,
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
                                    batchsize=500,
                                    # TODO: try other max_opt_itr
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
                            snapshot_gap=10,
                            mode="ec2",
                            use_gpu=True,
                            seed=args.seed,
                            resume_from=args.resume_from)
    else:
        run_experiment_lite(algo.train(),
                            snapshot_mode='gap',
                            snapshot_gap=10,
                            mode="ec2",
                            use_gpu=True,
                            seed=args.seed)

    import IPython as ipy; ipy.embed()


if __name__ == '__main__':
    main()
