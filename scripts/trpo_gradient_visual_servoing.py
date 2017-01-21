import argparse
import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T
import yaml
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_conv_policy import GaussianConvPolicy

import envs
import utils
from envs import ServoingEnv, RllabEnv
from policy import TheanoServoingPolicy


class ServoingPolicyNetwork(object):
    def __init__(self, input_shape, output_dim, servoing_pol,
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

        l_out = TheanoServoingPolicyLayer(l_hid, servoing_pol)
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


class TheanoServoingPolicyLayer(L.Layer):
    def __init__(self, incoming, servoing_pol, **kwargs):
        assert isinstance(servoing_pol, TheanoServoingPolicy)
        super(TheanoServoingPolicyLayer, self).__init__(incoming, **kwargs)

        assert len(self.input_shape) == 4 and self.input_shape[1] == 6
        self.action_space = servoing_pol.action_space

        self.w_presoftplus_var = self.add_param(np.log(np.exp(servoing_pol.w) - 1).astype(theano.config.floatX), servoing_pol.w.shape, name='w')
        self.lambda_presoftplus_var = self.add_param(np.log(np.exp(servoing_pol.lambda_) - 1).astype(theano.config.floatX), servoing_pol.lambda_.shape, name='lambda')
        self.w_var = T.log(1 + T.exp(self.w_presoftplus_var))
        self.lambda_var = T.log(1 + T.exp(self.lambda_presoftplus_var))

        self.X_var, U_var, self.X_target_var, self.U_lin_var, alpha_var = servoing_pol.input_vars
        w_var, lambda_var = servoing_pol.param_vars
        pi_var = servoing_pol._get_pi_var()
        self.pi_var = theano.clone(pi_var, replace={w_var: self.w_var,
                                                    lambda_var: self.lambda_var,
                                                    alpha_var: np.array(servoing_pol.alpha, dtype=theano.config.floatX)})

    def get_output_shape_for(self, input_shape):
        return self.action_space.shape

    def get_output_for(self, input, **kwargs):
        return theano.clone(self.pi_var, replace={self.X_var: input[:, :3, :, :],
                                                  self.X_target_var: input[:, 3:, :, :],
                                                  self.U_lin_var: T.zeros((input.shape[0],) + self.action_space.shape)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--image_transformer_fname', type=str)
    parser.add_argument('--algorithm_fname', type=str)
    parser.add_argument('--conv_filters', nargs='*', type=int, default=[16, 32])
    parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[16])
    parser.add_argument('--init_std', type=float, default=1.0)
    parser.add_argument('--n_itr', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--use_static_car', action='store_true')
    args = parser.parse_args()

    # instantiate predictor later since the transformers might need to be updated
    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    # apply transformations from predictor to environment
    if args.image_transformer_fname:
        with open(args.image_transformer_fname) as image_transformer_file:
            image_transformer = utils.from_yaml(image_transformer_file)
    else:
        image_transformer = None
    if issubclass(predictor_config['environment_config']['class'], envs.Panda3dEnv):
        utils.transfer_image_transformer(predictor_config, image_transformer)

    # predictor
    predictor = utils.from_config(predictor_config)

    # environment
    if issubclass(predictor.environment_config['class'], envs.RosEnv):
        import rospy
        rospy.init_node("learn_visual_servoing")
    env = utils.from_config(predictor.environment_config)
    if args.use_static_car:
        env.car_env.speed_offset_space.low = \
        env.car_env.speed_offset_space.high = np.array([0.0, 4.0])
    env = ServoingEnv(env)
    env = RllabEnv(env, transformers=predictor.transformers)
    env = normalize(env)

    # policy
    servoing_pol = TheanoServoingPolicy(predictor)
    if args.algorithm_fname:
        with open(args.algorithm_fname) as algorithm_file:
            algorithm_config = yaml.load(algorithm_file)
        best_iter = np.argmax(algorithm_config['mean_discounted_returns'])
        best_theta = np.asarray(algorithm_config['thetas'][best_iter])
        servoing_pol.theta = best_theta

    mean_network = ServoingPolicyNetwork(env.observation_space.shape, env.action_space.flat_dim, servoing_pol)

    policy = GaussianConvPolicy(
        env_spec=env.spec,
        init_std=args.init_std,
        mean_network=mean_network,
    )
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
