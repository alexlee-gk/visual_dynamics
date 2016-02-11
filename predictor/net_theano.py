from __future__ import division

from collections import OrderedDict
import numpy as np
import theano.tensor as T
import lasagne
import lasagne.layers as L
from lasagne.layers.dnn import dnn
from lasagne import init
from lasagne.utils import as_tuple


# deconv_length and Deconv2DLayer adapted from https://github.com/ebenolson/Lasagne/blob/deconv/lasagne/layers/dnn.py
def deconv_length(output_length, filter_size, stride, pad=0):
    if output_length is None:
        return None

    output_length = output_length * stride
    if pad == 'valid':
        input_length = output_length + filter_size - 1
    elif pad == 'full':
        input_length = output_length - filter_size + 1
    elif pad == 'same':
        input_length = output_length
    elif isinstance(pad, int):
        input_length = output_length - 2 * pad + filter_size - stride
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    return input_length

class Deconv2DLayer(L.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=(2, 2),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 flip_filters=False, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = lasagne.nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad == 'full':
            self.pad = 'full'
        elif pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
            self.pad = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2],
                                self.output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = deconv_length(input_shape[2],
                                    self.filter_size[0],
                                    self.stride[0],
                                    pad[0])

        output_columns = deconv_length(input_shape[3],
                                       self.filter_size[1],
                                       self.stride[1],
                                       pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'

        image = T.alloc(0., input.shape[0], *self.output_shape[1:])
        conved = dnn.dnn_conv(img=image,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=self.pad,
                              conv_mode=conv_mode
                              )

        grad = T.grad(conved.sum(), wrt=image, known_grads={conved: input})

        if self.b is None:
            activation = grad
        elif self.untie_biases:
            activation = grad + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = grad + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)


class BilinearLayer(L.MergeLayer):
    def __init__(self, incomings, axis=1, M=init.Normal(std=0.001),
                 N=init.Normal(std=0.001), b=init.Constant(0.), **kwargs):
        """
        axis: The first axis of Y to be lumped into a single bilinear model.
            The bilinear model are computed independently for each element wrt the preceding axes.
        """
        super(BilinearLayer, self).__init__(incomings, **kwargs)
        assert axis >= 1
        self.axis = axis

        self.y_shape, self.u_shape = [input_shape[1:] for input_shape in self.input_shapes]
        self.y_dim = int(np.prod(self.y_shape[self.axis-1:]))
        self.u_dim,  = self.u_shape

        self.M = self.add_param(M, (self.y_dim, self.y_dim, self.u_dim), name='M')
        self.N = self.add_param(N, (self.y_dim, self.u_dim), name='N')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.y_dim,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        Y_shape, U_shape = input_shapes
        assert Y_shape[0] == U_shape[0]
        return Y_shape

    def get_output_for(self, inputs, **kwargs):
        Y, U = inputs
        if Y.ndim > (self.axis + 1):
            Y = Y.flatten(self.axis + 1)
        assert Y.ndim == self.axis + 1

        outer_YU = Y.dimshuffle(range(Y.ndim) + ['x']) * U.dimshuffle([0] + ['x']*self.axis + [1])
        bilinear = T.dot(outer_YU.reshape((-1, self.y_dim * self.u_dim)), self.M.reshape((self.y_dim, self.y_dim * self.u_dim)).T)
        if self.axis > 1:
            bilinear = bilinear.reshape((-1,) + self.y_shape[:self.axis-1] + (self.y_dim,))
        linear = T.dot(U, self.N.T)
        if self.axis > 1:
            linear = linear.dimshuffle([0] + ['x']*(self.axis-1) + [1])
        activation = bilinear + linear
        if self.b is not None:
            activation += self.b.dimshuffle(['x']*self.axis + [0])

        activation = activation.reshape((-1,) + self.y_shape)
        return activation


def build_bilinear_net(input_shapes):
    x_shape, u_shape = input_shapes
    X_var = T.tensor4('X')
    U_var = T.matrix('U')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)

    l_x_diff_pred = BilinearLayer([l_x, l_u], b=None)
    l_x_next_pred = L.ElemwiseMergeLayer([l_x, l_x_diff_pred], T.add)
    l_y = L.flatten(l_x)
    l_y_diff_pred = L.flatten(l_x_diff_pred)

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'BilinearNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('Y_diff_pred', l_y_diff_pred), ('Y', l_y), ('X_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss


def build_small_action_cond_encoder_net(input_shapes):
    x_shape, u_shape = input_shapes
    x2_c_dim = x1_c_dim = x_shape[0]
    x1_shape = (x1_c_dim, x_shape[1]//2, x_shape[2]//2)
    x2_shape = (x2_c_dim, x1_shape[1]//2, x1_shape[2]//2)
    y2_dim = 64
    X_var = T.tensor4('X')
    U_var = T.matrix('U')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)

    l_x1 = L.Conv2DLayer(l_x, x1_c_dim, filter_size=6, stride=2, pad=2,
                         W=init.Normal(std=0.01),
                         nonlinearity=lasagne.nonlinearities.rectify)
    l_x2 = L.Conv2DLayer(l_x1, x2_c_dim, filter_size=6, stride=2, pad=2,
                         W=init.Normal(std=0.01),
                         nonlinearity=lasagne.nonlinearities.rectify)

    l_y2 = L.DenseLayer(l_x2, y2_dim, nonlinearity=None)
    l_y2_diff_pred = BilinearLayer([l_y2, l_u], b=None)
    l_y2_next_pred = L.ElemwiseMergeLayer([l_y2, l_y2_diff_pred], T.add)
    l_x2_next_pred_flat = L.DenseLayer(l_y2_next_pred, np.prod(x2_shape), nonlinearity=None)
    l_x2_next_pred = L.ReshapeLayer(l_x2_next_pred_flat, ([0],) + x2_shape)

    l_x1_next_pred = Deconv2DLayer(l_x2_next_pred, x2_c_dim, filter_size=6, stride=2, pad=2,
                                   W=init.Normal(std=0.01),
                                   nonlinearity=lasagne.nonlinearities.rectify)
    l_x_next_pred = Deconv2DLayer(l_x1_next_pred, x1_c_dim, filter_size=6, stride=2, pad=2,
                                  W=init.Normal(std=0.01),
                                  nonlinearity=lasagne.nonlinearities.tanh)

    l_y = l_y2
    l_y_diff_pred = l_y2_diff_pred

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'SmallActionCondEncoderNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('Y_diff_pred', l_y_diff_pred), ('Y', l_y), ('X_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss

def build_fcn_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, num_downsample=0, share_bilinear_weights=True, ladder_loss=True, batch_normalization=False, concat=False, tanh=True):
    if not share_bilinear_weights:
        raise NotImplementedError
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)

    # preprocess
    if num_downsample > 0:
        raise NotImplementedError
    l_x0 = l_x
    x0_shape = x_shape

    # encoding
    l_xlevels = OrderedDict()
    l_ylevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x0
        else:
            if level == 1:
                xlevelm1_c_dim = x0_shape[0]
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevelm1_c_dim
            l_xlevel_1 = L.Conv2DLayer(l_xlevels[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       W=lasagne.init.Normal(std=0.01), b=lasagne.init.Constant(0.0),
                                       nonlinearity=lasagne.nonlinearities.rectify)
            if batch_normalization:
                l_xlevel_1 = L.batch_norm(l_xlevel_1)
            l_xlevel_2 = L.Conv2DLayer(l_xlevel_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       W=init.Normal(std=0.01), b=lasagne.init.Constant(0.0),
                                       nonlinearity=lasagne.nonlinearities.rectify)
            if batch_normalization:
                l_xlevel_2 = L.batch_norm(l_xlevel_2)
            l_xlevel = L.MaxPool2DLayer(l_xlevel_2, pool_size=2, stride=2, pad=0)
        l_xlevels[level] = l_xlevel
        l_ylevels[level] = L.FlattenLayer(l_xlevel)
    l_y = L.ConcatLayer(l_ylevels.values())

    # bilinear
    l_xlevels_next_pred_s0 = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_pred = BilinearLayer([l_xlevel, l_u], axis=2) # TODO: add linear term in y
        l_xlevel_next_pred_s0 = L.ElemwiseMergeLayer([l_xlevel, l_xlevel_diff_pred], T.add)
        l_xlevels_next_pred_s0[level] = l_xlevel_next_pred_s0

    # decoding
    l_xlevels_next_pred = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            l_xlevel_next_pred = l_xlevels_next_pred_s0[level]
        else:
            if level == 0:
                xlevel_c_dim = x1_c_dim
                xlevelm1_c_dim = x0_shape[0]
            elif level < levels[-1]-1:
                xlevel_c_dim = xlevelm1_c_dim
                xlevelm1_c_dim = xlevel_c_dim // 2
            l_xlevel_next_pred_2 = Deconv2DLayer(l_xlevels_next_pred[level+1], xlevel_c_dim, filter_size=2, stride=2, pad=0,
                                                 W=init.Normal(std=0.01), b=lasagne.init.Constant(0.0),
                                                 nonlinearity=None) # TODO initialize with bilinear weights # TODO should rectify? # TODO: channel-wise (groups) # TODO: no bias term
            l_xlevel_next_pred_1 = Deconv2DLayer(l_xlevel_next_pred_2, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                                 W=init.Normal(std=0.01), b=lasagne.init.Constant(0.0),
                                                 nonlinearity=lasagne.nonlinearities.rectify)
            if batch_normalization:
                l_xlevel_next_pred_1 = L.batch_norm(l_xlevel_next_pred_1)
            if concat:
                if level in l_xlevels_next_pred_s0:
                    l_xlevel_next_pred_1 = L.ConcatLayer([l_xlevels_next_pred_s0[level], l_xlevel_next_pred_1])
                l_xlevel_next_pred = Deconv2DLayer(l_xlevel_next_pred_1, xlevelm1_c_dim, filter_size=3, stride=1, pad=1,
                                                   W=init.Normal(std=0.01), b=lasagne.init.Constant(0.0),
                                                   nonlinearity=lasagne.nonlinearities.tanh if (level == 0 and tanh) else lasagne.nonlinearities.rectify)
                if batch_normalization: # TODO batch normat level 0?
                    l_xlevel_next_pred = L.batch_norm(l_xlevel_next_pred)
            else:
                l_xlevel_next_pred_s1 = Deconv2DLayer(l_xlevel_next_pred_1, xlevelm1_c_dim, filter_size=3, stride=1, pad=1,
                                                      W=init.Normal(std=0.01), b=lasagne.init.Constant(0.0),
                                                      nonlinearity=None)
                if batch_normalization:
                    l_xlevel_next_pred_s1 = L.batch_norm(l_xlevel_next_pred_s1)
                if level in l_xlevels_next_pred_s0:
                    l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevels_next_pred_s0[level], l_xlevel_next_pred_s1], coeffs=[0.5, 0.5]) # TODO weights should be learned
                else:
                    l_xlevel_next_pred = l_xlevel_next_pred_s1
                l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred, nonlinearity=lasagne.nonlinearities.tanh if (level == 0 and tanh) else lasagne.nonlinearities.rectify)
        l_xlevels_next_pred[level] = l_xlevel_next_pred
        l_ylevels_diff_pred[level] = L.FlattenLayer(L.ElemwiseSumLayer([l_xlevel_next_pred, l_xlevels[level]], coeffs=[1.0, -1.0]))
    l_y_diff_pred = L.ConcatLayer(l_ylevels_diff_pred.values())

    l_x0_next_pred = l_xlevels_next_pred[0]
    X0_next_pred_var = lasagne.layers.get_output(l_x0_next_pred)

    # preprocess
    X0_next_var = X_next_var

    loss = ((X0_next_var - X0_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    if ladder_loss:
        raise NotImplementedError

    net_name = 'FcnActionCondEncoderNet'
    net_name +='_levels' + ''.join([str(level) for level in levels])
    net_name += '_x1cdim' + str(x1_c_dim)
    net_name += '_numds' + str(num_downsample)
    net_name += '_share' + str(int(share_bilinear_weights))
    net_name += '_ladder' + str(int(ladder_loss))
    net_name += '_bn' + str(int(batch_normalization))
    if concat:
        net_name += '_concat' + str(int(concat))

    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('Y_diff_pred', l_y_diff_pred), ('Y', l_y), ('X_next_pred', l_x0_next_pred)]) # TODO distinction between x and preprocessed x0
    return net_name, input_vars, pred_layers, loss
