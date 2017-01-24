import argparse
import lasagne.init as LI
import lasagne.layers as L
import lasagne.nonlinearities as LN
import numpy as np
import yaml
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.core.network import ConvNetwork
from rllab.core.network import wrapped_conv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_conv_policy import GaussianConvPolicy

from visual_dynamics import envs
from visual_dynamics.envs import ServoingEnv, RllabEnv
from visual_dynamics.utils.config import from_config


class SiameseQuadraticErrorNetwork(object):
    def __init__(self, input_shape, output_dim, hidden_sizes,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 hidden_nonlinearity=LN.rectify,
                 output_nonlinearity=LN.softmax,
                 name=None, input_var=None):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if len(input_shape) == 3:
            l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var)
            l_hid = L.reshape(l_in, ([0],) + input_shape)
        elif len(input_shape) == 2:
            l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var)
            input_shape = (1,) + input_shape
            l_hid = L.reshape(l_in, ([0],) + input_shape)
        else:
            l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
            l_hid = l_in

        assert input_shape[0] % 2 == 0
        l_hid0 = L.SliceLayer(l_hid, slice(None, input_shape[0] // 2), axis=1)
        l_hid1 = L.SliceLayer(l_hid, slice(input_shape[0] // 2, None), axis=1)
        l_hids = [l_hid0, l_hid1]

        for idx, conv_filter, filter_size, stride, pad in zip(
                range(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
        ):
            for ihid in range(len(l_hids)):
                if ihid > 0:
                    conv_kwargs = dict(W=l_hids[0].W,
                                       b=l_hids[0].b)
                else:
                    conv_kwargs = dict()
                l_hids[ihid] = L.Conv2DLayer(
                    l_hids[ihid],
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad,
                    nonlinearity=hidden_nonlinearity,
                    name="%sconv_hidden_%d_%d" % (prefix, idx, ihid),
                    convolution=wrapped_conv,
                    **conv_kwargs
                )

        l_hid = L.ElemwiseSumLayer(l_hids, coeffs=[-1, 1])
        l_hid = L.ExpressionLayer(l_hid, lambda X: X * X)

        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_fname', type=str, help='config file with environment arguments')
    parser.add_argument('transformers_fname', type=str)
    parser.add_argument('mean_network_type', type=str, choices=['conv', 'siamese'])
    parser.add_argument('--conv_filters', nargs='*', type=int, default=[16, 32])
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[16])
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--n_itr', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--use_static_car', action='store_true')
    parser.add_argument('--use_init_heuristic', action='store_true')
    args = parser.parse_args()

    with open(args.env_fname) as yaml_string:
        env_config = yaml.load(yaml_string)
        if issubclass(env_config['class'], envs.RosEnv):
            import rospy
            rospy.init_node("generate_data")
        env = from_config(env_config)

    if args.use_static_car:
        env.car_env.speed_offset_space.low = \
        env.car_env.speed_offset_space.high = np.array([0.0, 4.0])

    # transformers
    with open(args.transformers_fname) as transformers_file:
        transformers_config = yaml.load(transformers_file)
    transformers = dict()
    for data_name, transformer_config in transformers_config.items():
        if data_name == 'action':
            replace_config = {'space': env.action_space}
        elif data_name in env.observation_space.spaces:
            replace_config = {'space': env.observation_space.spaces[data_name]}
        else:
            replace_config = {}
        transformers[data_name] = from_config(transformers_config[data_name], replace_config=replace_config)

    env = ServoingEnv(env)
    env = RllabEnv(env, transformers=transformers)
    env = normalize(env)

    network_kwargs = dict(
        input_shape=env.observation_space.shape,
        output_dim=env.action_space.flat_dim,
        conv_filters=args.conv_filters,
        conv_filter_sizes=[3] * len(args.conv_filters),
        conv_strides=[2] * len(args.conv_filters),
        conv_pads=[0] * len(args.conv_filters),
        hidden_sizes=args.hidden_sizes,
        hidden_nonlinearity=LN.rectify,
        output_nonlinearity=None,
        name="mean_network",
    )
    if args.mean_network_type == 'conv':
        mean_network = ConvNetwork(**network_kwargs)
    elif args.mean_network_type == 'siamese':
        mean_network = SiameseQuadraticErrorNetwork(**network_kwargs)
    else:
        raise NotImplementedError

    policy = GaussianConvPolicy(
        env_spec=env.spec,
        init_std=args.init_std,
        mean_network=mean_network,
    )
    if args.use_init_heuristic:
        W_var = policy.get_params()[0]
        W = W_var.get_value()
        W[:, 3:, :, :] = -W[:, :3, :, :]
        W_var.set_value(W)
    baseline = GaussianConvBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            step_size=args.step_size,
            normalize_inputs=True,
            normalize_outputs=True,
            hidden_sizes=args.hidden_sizes,
            conv_filters=args.conv_filters,
            conv_filter_sizes=[3] * len(args.conv_filters),
            conv_strides=[2] * len(args.conv_filters),
            conv_pads=[0] * len(args.conv_filters),
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
