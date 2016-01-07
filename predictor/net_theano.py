from __future__ import division

from collections import OrderedDict
import numpy as np
import theano.tensor as T
import lasagne
import lasagne.layers as L
from lasagne import init
from lasagne.utils import as_tuple


class Deconv2DLayer(L.Conv2DLayer):
    def __init__(self, incoming, channels, filter_size, stride=(1, 1),
                 pad=0, W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        self.original_channels = channels
        self.original_filter = as_tuple(filter_size, 2)
        self.original_stride = as_tuple(stride, 2)

        if pad == 'valid':
            self.original_pad = (0, 0)
        elif pad in ('full', 'same'):
            self.original_pad = pad
        else:
            self.original_pad = as_tuple(pad, 2, int)

        super(Deconv2DLayer, self).__init__(incoming, channels, filter_size,
                                            stride=(1,1), pad='full', W=W, b=b,
                                            nonlinearity=nonlinearity, **kwargs)

    def get_output_shape_for(self, input_shape):
        _, _, width, height = input_shape
        original_width = ((width - 1) * self.original_stride[0]) - 2 * self.original_pad[0] + self.original_filter[0]
        original_height = ((height - 1) * self.original_stride[1]) - 2 * self.original_pad[1] + self.original_filter[1]
        return (input_shape[0], self.original_channels, original_width, original_height)

    def get_output_for(self, input, **kwargs):
        # first we upsample to compensate for strides
        if self.original_stride != 1:
            _, _, width, height = input.shape
            unstrided_width = width * self.original_stride[0]
            unstrided_height = height * self.original_stride[1]
            placeholder = T.zeros((input.shape[0], input.shape[1], unstrided_width, unstrided_height))
            upsampled = T.set_subtensor(placeholder[:, :, ::self.original_stride[0], ::self.original_stride[1]], input)
        else:
            upsampled = input
        # then we conv to deconv
        deconv = super(Deconv2DLayer, self).get_output_for(upsampled, input_shape=(None, self.input_shape[1], self.input_shape[2]*self.original_stride[0], self.input_shape[3]*self.original_stride[1]), **kwargs)
        # lastly we cut off original padding
        pad = self.original_pad
        _, _, original_width, original_height = self.get_output_shape_for(input.shape)
        t = deconv[:, :, pad[0]:(pad[0] + original_width), pad[1]:(pad[1] + original_height)]
        return t


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

def build_fcn_action_cond_encoder_net(input_shapes, levels=None):
    x_shape, u_shape = input_shapes
    x_c_dim = x_shape[0]
    x1_c_dim = 16
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('X')
    U_var = T.matrix('U')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)

    # encoding
    l_xlevels = {}
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x
        else:
            if level == 1:
                xlevel_c_dim = x1_c_dim
            else:
                xlevel_c_dim *= 2
            l_xlevel_1 = L.Conv2DLayer(l_xlevels[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       W=init.Normal(std=0.01),
                                       nonlinearity=lasagne.nonlinearities.rectify)
            l_xlevel_2 = L.Conv2DLayer(l_xlevel_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       W=init.Normal(std=0.01),
                                       nonlinearity=lasagne.nonlinearities.rectify)
            l_xlevel = L.MaxPool2DLayer(l_xlevel_2, pool_size=2, stride=2, pad=0)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_pred_0 = {}
    l_ylevels = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in levels:
        l_ylevel = l_xlevels[level]
        l_ylevels[level] = l_ylevel
        l_ylevel_diff_pred = BilinearLayer([l_ylevel, l_u], b=None, axis=2)
        l_ylevels_diff_pred[level] = l_ylevel_diff_pred
        l_ylevel_next_pred = L.ElemwiseMergeLayer([l_ylevel, l_ylevel_diff_pred], T.add)
        l_xlevel_next_pred = l_ylevel_next_pred
        l_xlevels_next_pred_0[level] = l_xlevel_next_pred

    # decoding
    l_xlevels_next_pred = {}
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            l_xlevel_next_pred = l_xlevels_next_pred_0[level]
        else:
            l_xlevel_next_pred_2 = Deconv2DLayer(l_xlevels_next_pred[level+1], xlevel_c_dim, filter_size=2, stride=2, pad=0,
                                             W=init.Normal(std=0.01),
                                             nonlinearity=None) # TODO initialize with bilinear # TODO should rectify?
            l_xlevel_next_pred_1 = Deconv2DLayer(l_xlevel_next_pred_2, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                             W=init.Normal(std=0.01),
                                             nonlinearity=lasagne.nonlinearities.rectify)
            if level == 0:
                xlevel_c_dim = x_c_dim
            else:
                xlevel_c_dim = xlevel_c_dim // 2
            l_xlevel_next_pred_1 = Deconv2DLayer(l_xlevel_next_pred_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                          W=init.Normal(std=0.01),
                                          nonlinearity=lasagne.nonlinearities.rectify if level > 0 else lasagne.nonlinearities.tanh)
            if level in l_xlevels_next_pred_0:
                l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevels_next_pred_0[level], l_xlevel_next_pred_1], coeffs=[0.5, 0.5]) # TODO should be learned
            else:
                l_xlevel_next_pred = l_xlevel_next_pred_1
        l_xlevels_next_pred[level] = l_xlevel_next_pred

    l_x_next_pred = l_xlevels_next_pred[0]
    l_y = L.ConcatLayer(l_ylevels.values())
    l_y_diff_pred = L.ConcatLayer(l_ylevels_diff_pred.values())

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'FcnActionCondEncoderNet_levels' + ''.join(str(level) for level in levels)
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('Y_diff_pred', l_y_diff_pred), ('Y', l_y), ('X_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss
