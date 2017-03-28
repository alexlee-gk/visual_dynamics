import lasagne.layers as L
import numpy as np
import theano
import theano.tensor as T

from visual_dynamics.policies import TheanoServoingPolicy


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

        self.sqrt_w_var = self.add_param(np.sqrt(servoing_pol.w).astype(theano.config.floatX), servoing_pol.w.shape, name='sqrt_w')
        self.sqrt_lambda_var = self.add_param(np.sqrt(servoing_pol.lambda_).astype(theano.config.floatX), servoing_pol.lambda_.shape, name='sqrt_lambda')
        self.w_var = self.sqrt_w_var ** 2
        self.lambda_var = self.sqrt_lambda_var ** 2

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
