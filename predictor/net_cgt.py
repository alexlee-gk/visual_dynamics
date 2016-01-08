from __future__ import division

from collections import OrderedDict
import numpy as np
import cgt
from cgt import nn


class SpatialDeconvolution(nn.SpatialConvolution):
    def __init__(self, input_channels, output_channels,
                 kernelshape, pad, stride=(1,1), name=None,
                 weight_init=nn.Constant(0), bias_init=nn.Constant(0)):
        self.original_input_channels = input_channels
        self.original_output_channels = output_channels
        self.original_kernelshape = tuple(map(int, kernelshape))
        self.original_pad = tuple(map(int,pad))
        self.original_stride = tuple(map(int,stride))
        super(SpatialDeconvolution, self).__init__(input_channels, output_channels,
                                                   kernelshape, (kernelshape[0]-1, kernelshape[1]-1), stride=(1,1), name=name,
                                                   weight_init=weight_init, bias_init=bias_init)

    def __call__(self, x):
        if self.original_stride != 1:
            _, _, width, height = cgt.infer_shape(x)
            unstrided_width = width * self.original_stride[0]
            unstrided_height = height * self.original_stride[1]
            # workaround for this
            # cgt.inc_subtensor(upsampled, (slice(None), slice(None), slice(None, None, self.original_stride[0])), slice(None, None, self.original_stride[1])), x)
            placeholder = cgt.zeros((x.shape[0], x.shape[1], width, unstrided_height)) # (None, 64, 4, 8)
            cgt.inc_subtensor(placeholder, (slice(None), slice(None), slice(None), slice(None, None, self.original_stride[1])), x)
            upsampled = cgt.zeros((x.shape[0], x.shape[1], unstrided_width, unstrided_height)) # (None, 64, 8, 8)
            cgt.inc_subtensor(upsampled, (slice(None), slice(None), slice(None, None, self.original_stride[0]), slice(None)), placeholder)
        else:
            upsampled = x
        # then we conv to deconv
        deconv = super(SpatialDeconvolution, self).__call__(upsampled)
        # lastly we cut off original padding
        pad = self.original_pad
        original_width = ((width - 1) * self.original_stride[0]) - 2 * self.original_pad[0] + self.original_kernelshape[0]
        original_height = ((height - 1) * self.original_stride[1]) - 2 * self.original_pad[1] + self.original_kernelshape[1]
        t = deconv[:, :, pad[0]:(pad[0] + original_width), pad[1]:(pad[1] + original_height)]
        return t


class Bilinear(object):
    def __init__(self, input_shapes, axis=1, name=None,
                 M=nn.IIDGaussian(std=0.001),
                 N=nn.IIDGaussian(std=0.001),
                 b=nn.Constant(0)):
        assert axis >= 1
        self.axis = axis
        name = "unnamed" if name is None else name

        self.y_shape, self.u_shape = input_shapes
        self.y_dim = int(np.prod(self.y_shape[self.axis-1:]))
        self.u_dim,  = self.u_shape

        self.M = nn.parameter(nn.init_array(M, (self.y_dim, self.y_dim, self.u_dim)),
                              name=name+".M")
        self.N = nn.parameter(nn.init_array(N, (self.y_dim, self.u_dim)),
                              name=name+".N")
        if b is None:
            self.b = None
        else:
            self.b = nn.parameter(nn.init_array(b, (self.y_dim,)),
                                  name=name+".b") # TODO: not regularizable

    def __call__(self, Y, U):
        if Y.ndim > (self.axis + 1):
            Y = Y.reshape(Y.shape[:self.axis] + [cgt.mul_multi(Y.shape[self.axis:])])

        outer_YU = cgt.broadcast('*',
                                 Y.dimshuffle(range(Y.ndim) + ['x']),
                                 U.dimshuffle([0] + ['x']*self.axis + [1]),
                                 ''.join(['x']*Y.ndim + ['1', ',', 'x'] + ['1']*self.axis + ['x']))
        bilinear = cgt.dot(outer_YU.reshape((outer_YU.shape[0], cgt.mul_multi(outer_YU.shape[1:]))),
                           self.M.reshape((self.y_dim, self.y_dim * self.u_dim)).T)
        if self.axis > 1:
            bilinear = bilinear.reshape((-1,) + self.y_shape[:self.axis-1] + (self.y_dim,))
        linear = cgt.dot(U, self.N.T)
        if self.axis > 1:
            linear = linear.dimshuffle([0] + ['x']*(self.axis-1) + [1])
        activation = bilinear + linear
        if self.b is not None:
            activation += cgt.broadcast('+',
                                        activation,
                                        self.b.dimshuffle(['x']*self.axis + [0]),
                                        ''.join(['x']*activation.ndim + [','] + ['1']*(activation.ndim-1) + ['x']))
        activation = activation.reshape((-1,) + self.y_shape)
        return activation


def build_bilinear_net(input_shapes, **kwargs):
    x_shape, u_shape = input_shapes
    X = cgt.tensor4('X', fixed_shape=(None,) + x_shape)
    U = cgt.matrix('U', fixed_shape=(None,) + u_shape)

    X_diff_pred = Bilinear(input_shapes, b=None, name='bilinear')(X, U)
    X_next_pred = X + X_diff_pred
    Y = X.reshape((X.shape[0], cgt.mul_multi(X.shape[1:])))
    Y_diff_pred = X_diff_pred.reshape((X_diff_pred.shape[0], cgt.mul_multi(X_diff_pred.shape[1:])))

    X_diff = cgt.tensor4('X_diff', fixed_shape=(None,) + x_shape)
    X_next = X + X_diff
    loss = ((X_next - X_next_pred) ** 2).mean(axis=0).sum() / 2.

    net_name = 'BilinearNet'
    input_vars = OrderedDict([(var.name, var) for var in [X, U, X_diff]])
    pred_vars = OrderedDict([('Y_diff_pred', Y_diff_pred), ('Y', Y), ('X_next_pred', X_next_pred)])
    return net_name, input_vars, pred_vars, loss


def build_fcn_action_cond_encoder_net(input_shapes, levels=None):
    x_shape, u_shape = input_shapes
    x_c_dim = x_shape[0]
    x1_c_dim = 16
    levels = levels or [3]
    levels = sorted(set(levels))

    X = cgt.tensor4('X', fixed_shape=(None,) + x_shape)
    U = cgt.matrix('U', fixed_shape=(None,) + u_shape)

    # encoding
    Xlevels = {}
    for level in range(levels[-1]+1):
        if level == 0:
            Xlevel = X
        else:
            if level == 1:
                xlevelm1_c_dim = x_c_dim
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevel_c_dim
            Xlevel_1 = nn.rectify(nn.SpatialConvolution(xlevelm1_c_dim, xlevel_c_dim, kernelshape=(3,3), pad=(1,1), stride=(1,1), name='conv%d_1'%level,
                                                        weight_init=nn.IIDGaussian(std=0.01))(Xlevels[level-1]))
            Xlevel_2 = nn.rectify(nn.SpatialConvolution(xlevel_c_dim, xlevel_c_dim, kernelshape=(3,3), pad=(1,1), stride=(1,1), name='conv%d_2'%level,
                                                        weight_init=nn.IIDGaussian(std=0.01))(Xlevel_1))
            Xlevel = nn.max_pool_2d(Xlevel_2, kernelshape=(2,2), pad=(0,0), stride=(2,2))
        Xlevels[level] = Xlevel 

    # bilinear
    Xlevels_next_pred_0 = {}
    Ylevels = {}
    Ylevels_diff_pred = {}
    for level in levels:
        Xlevel = Xlevels[level]
        Xlevel_diff_pred = Bilinear(input_shapes, b=None, axis=2, name='bilinear%d'%level)(Xlevel, U)
        Xlevels_next_pred_0[level] = Xlevel + Xlevel_diff_pred
        Ylevels[level] = Xlevel.reshape((Xlevel.shape[0], cgt.mul_multi(Xlevel.shape[1:])))
        Ylevels_diff_pred[level] = Xlevel_diff_pred.reshape((Xlevel_diff_pred.shape[0], cgt.mul_multi(Xlevel_diff_pred.shape[1:])))

    # decoding
    Xlevels_next_pred = {}
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            Xlevel_next_pred = Xlevels_next_pred_0[level]
        else:
            if level == 0:
                xlevelm1_c_dim = x_c_dim
            elif level < levels[-1]-1:
                xlevel_c_dim = xlevelm1_c_dim
                xlevelm1_c_dim = xlevelm1_c_dim // 2
            Xlevel_next_pred_2 = SpatialDeconvolution(xlevel_c_dim, xlevel_c_dim, 
                                                      kernelshape=(2,2), pad=(0,0), stride=(2,2), name='upsample%d'%(level+1),
                                                      weight_init=nn.IIDGaussian(std=0.01))(Xlevels_next_pred[level+1]) # TODO initialize with bilinear # TODO should rectify?
            Xlevel_next_pred_1 = nn.rectify(SpatialDeconvolution(xlevel_c_dim, xlevel_c_dim, 
                                                                 kernelshape=(3,3), pad=(1,1), stride=(1,1), name='deconv%d_2'%(level+1),
                                                                 weight_init=nn.IIDGaussian(std=0.01))(Xlevel_next_pred_2))
            nonlinearity = nn.rectify if level > 0 else cgt.tanh
            Xlevel_next_pred = nonlinearity(SpatialDeconvolution(xlevel_c_dim, xlevelm1_c_dim, 
                                                                 kernelshape=(3,3), pad=(1,1), stride=(1,1), name='deconv%d_1'%(level+1),
                                                                 weight_init=nn.IIDGaussian(std=0.01))(Xlevel_next_pred_1))
            if level in Xlevels_next_pred_0:
                coefs = nn.parameter(nn.init_array(nn.Constant(0.5), (2,)), name='sum%d.coef'%level)
                Xlevel_next_pred = coefs[0] * Xlevel_next_pred + coefs[1] * Xlevels_next_pred_0[level]
        Xlevels_next_pred[level] = Xlevel_next_pred

    X_next_pred = Xlevels_next_pred[0]
    Y = cgt.concatenate(Ylevels.values(), axis=1)
    Y_diff_pred = cgt.concatenate(Ylevels_diff_pred.values(), axis=1)

    X_diff = cgt.tensor4('X_diff', fixed_shape=(None,) + x_shape)
    X_next = X + X_diff
    loss = ((X_next - X_next_pred) ** 2).mean(axis=0).sum() / 2.

    net_name = 'FcnActionCondEncoderNet_levels' + ''.join(str(level) for level in levels)
    input_vars = OrderedDict([(var.name, var) for var in [X, U, X_diff]])
    pred_vars = OrderedDict([('Y_diff_pred', Y_diff_pred), ('Y', Y), ('X_next_pred', X_next_pred)])
    return net_name, input_vars, pred_vars, loss
