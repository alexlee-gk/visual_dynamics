from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as nl
from lasagne import init
from . import layers_theano as LT


def build_bilinear_net(input_shapes, X_var=None, U_var=None, X_diff_var=None, axis=1):
    x_shape, u_shape = input_shapes
    X_var = X_var or T.tensor4('X')
    U_var = U_var or T.matrix('U')
    X_diff_var = X_diff_var or T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)

    l_x_diff_pred = LT.BilinearLayer([l_x, l_u], axis=axis)
    l_x_next_pred = L.ElemwiseMergeLayer([l_x, l_x_diff_pred], T.add)
    l_y = L.flatten(l_x)
    l_y_diff_pred = L.flatten(l_x_diff_pred)

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'BilinearNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred), ('y', l_y), ('x0_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss


def build_action_cond_encoder_net(input_shapes, **kwargs):
    x_shape, u_shape = input_shapes

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x0 = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    l_x1 = L.Conv2DLayer(l_x0, 64, filter_size=6, stride=2, pad=0,
                         nonlinearity=nl.rectify,
                         name='x1')
    l_x2 = L.Conv2DLayer(l_x1, 64, filter_size=6, stride=2, pad=2,
                         nonlinearity=nl.rectify,
                         name='x2')
    l_x3 = L.Conv2DLayer(l_x2, 64, filter_size=6, stride=2, pad=2,
                         nonlinearity=nl.rectify,
                         name='x3')
    l_x3_shape = lasagne.layers.get_output_shape(l_x3)

    l_y4 = L.DenseLayer(l_x3, 1024, nonlinearity=nl.rectify, name='y')
    l_y4d = L.DenseLayer(l_y4, 2048, W=init.Uniform(1.0), nonlinearity=None)
    l_ud = L.DenseLayer(l_u, 2048, W=init.Uniform(0.1), nonlinearity=None)

    l_y4d_diff_pred = L.ElemwiseMergeLayer([l_y4d, l_ud], T.mul)
    l_y4_diff_pred = L.DenseLayer(l_y4d_diff_pred, 1024, W=init.Uniform(1.0), nonlinearity=None, name='y_diff_pred')

    l_y4_next_pred = L.ElemwiseMergeLayer([l_y4, l_y4_diff_pred], T.add, name='y_next_pred')

    l_y3_next_pred = L.DenseLayer(l_y4_next_pred, np.prod(l_x3_shape[1:]), nonlinearity=nl.rectify)
    l_x3_next_pred = L.ReshapeLayer(l_y3_next_pred, ([0],) + l_x3_shape[1:],
                                   name='x3_next_pred')

    l_x2_next_pred = LT.Deconv2DLayer(l_x3_next_pred, 64, filter_size=6, stride=2, pad=2,
                                   nonlinearity=nl.rectify,
                                   name='x2_next_pred')
    l_x1_next_pred = LT.Deconv2DLayer(l_x2_next_pred, 64, filter_size=6, stride=2, pad=2,
                                   nonlinearity=nl.rectify,
                                   name='x1_next_pred')
    l_x0_next_pred = LT.Deconv2DLayer(l_x1_next_pred, 3, filter_size=6, stride=2, pad=0,
                                   nonlinearity=None,
                                   name='x0_next_pred')

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = loss_fn(X_next_var, lasagne.layers.get_output(l_x0_next_pred))

    net_name = 'ActionCondEncoderNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('x0_next_pred', l_x0_next_pred)])
    return net_name, input_vars, pred_layers, loss


def build_small_action_cond_encoder_net(input_shapes):
    x_shape, u_shape = input_shapes
    x2_c_dim = x1_c_dim = x_shape[0]
    x1_shape = (x1_c_dim, x_shape[1]//2, x_shape[2]//2)
    x2_shape = (x2_c_dim, x1_shape[1]//2, x1_shape[2]//2)
    y2_dim = 64
    X_var = T.tensor4('X')
    U_var = T.matrix('U')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    l_x1 = L.Conv2DLayer(l_x, x1_c_dim, filter_size=6, stride=2, pad=2,
                         W=init.Normal(std=0.01),
                         nonlinearity=nl.rectify)
    l_x2 = L.Conv2DLayer(l_x1, x2_c_dim, filter_size=6, stride=2, pad=2,
                         W=init.Normal(std=0.01),
                         nonlinearity=nl.rectify)

    l_y2 = L.DenseLayer(l_x2, y2_dim, nonlinearity=None, name='y')
    l_y2_diff_pred = LT.BilinearLayer([l_y2, l_u], name='y_diff_pred')
    l_y2_next_pred = L.ElemwiseMergeLayer([l_y2, l_y2_diff_pred], T.add)
    l_x2_next_pred_flat = L.DenseLayer(l_y2_next_pred, np.prod(x2_shape), nonlinearity=None)
    l_x2_next_pred = L.ReshapeLayer(l_x2_next_pred_flat, ([0],) + x2_shape)

    l_x1_next_pred = LT.Deconv2DLayer(l_x2_next_pred, x2_c_dim, filter_size=6, stride=2, pad=2,
                                   W=init.Normal(std=0.01),
                                   nonlinearity=nl.rectify)
    l_x_next_pred = LT.Deconv2DLayer(l_x1_next_pred, x1_c_dim, filter_size=6, stride=2, pad=2,
                                  W=init.Normal(std=0.01),
                                  nonlinearity=nl.tanh,
                                  name='x_next_pred')

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'SmallActionCondEncoderNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y2_diff_pred), ('y', l_y2), ('x0_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss


def build_vgg_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, bilinear_type='share', tanh=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    xlevels_c_dim = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            xlevels_c_dim[level] = x_shape[0]
        else:
            xlevels_c_dim[level] = x1_c_dim * 2**(level-1)

    # encoding
    l_xlevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x
        else:
            l_xlevel = LT.VggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_pred = LT.create_bilinear_layer(l_xlevel, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_pred' % level)
        l_xlevels_next_pred[level] = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
                                                        name='x%d_next_pred' % level)
        if tanh:
            l_xlevels_next_pred[level].name += '_unconstrained'
            l_xlevels_next_pred[level] = L.NonlinearityLayer(l_xlevels_next_pred[level], nl.tanh,
                                                             name='x%d_next_pred' % level)

    pred_layers = OrderedDict([('x', l_xlevels[0]),
                               *[('x%d' % level, l_xlevels[level]) for level in l_xlevels.keys()],
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               *[('x%d_next_pred' % level, l_xlevels_next_pred[level]) for level in l_xlevels_next_pred.keys()]
                               ])
    return pred_layers


def build_vgg_fcn_action_cond_encoder_net(input_shapes, levels=None, bilinear_type='share', tanh=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    xlevels_c_dim = OrderedDict(zip(range(levels[-1]+1), [x_shape[0], 64, 128, 256, 512, 512]))

    # encoding
    l_xlevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x
        elif level < 3:
            l_xlevel = LT.VggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
            # TODO: non-trainable encodings
            LT.set_layer_param_tags(l_xlevel, trainable=False)
        else:
            l_xlevel = LT.VggEncoding3Layer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
            # TODO: non-trainable encodings
            LT.set_layer_param_tags(l_xlevel, trainable=False)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_u_outer = LT.OuterProductLayer([l_xlevel, l_u], name='x%d_u_outer')
        l_xlevel_diff_pred = L.Conv2DLayer(l_xlevel_u_outer, xlevels_c_dim[level], filter_size=7, stride=1, pad=3, nonlinearity=None,
                                           name='x%d_diff_pred' % level)
        l_xlevels_next_pred[level] = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
                                                        name='x%d_next_pred' % level)
        if tanh:
            l_xlevels_next_pred[level].name += '_unconstrained'
            l_xlevels_next_pred[level] = L.NonlinearityLayer(l_xlevels_next_pred[level], nl.tanh,
                                                             name='x%d_next_pred' % level)

    pred_layers = OrderedDict([('x', l_xlevels[0]),
                               *[('x%d' % level, l_xlevels[level]) for level in l_xlevels.keys()],
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               *[('x%d_next_pred' % level, l_xlevels_next_pred[level]) for level in l_xlevels_next_pred.keys()]
                               ])
    return pred_layers


def build_multiscale_action_cond_encoder_net(input_shapes, levels=None, bilinear_type='share', tanh=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    # multi-scale pyramid
    l_xlevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x
        else:
            l_xlevel = LT.Downscale2DLayer(l_xlevels[level-1], scale_factor=2, name='x%d' % level)
            # l_xlevel = L.Pool2DLayer(l_xlevels[level-1], pool_size=2, mode='average_inc_pad', name='x%d' % level)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_pred = LT.create_bilinear_layer(l_xlevel, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_pred' % level)
        l_xlevels_next_pred[level] = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
                                                        name='x%d_next_pred' % level)
        if tanh:
            l_xlevels_next_pred[level].name += '_unconstrained'
            l_xlevels_next_pred[level] = L.NonlinearityLayer(l_xlevels_next_pred[level], nl.tanh,
                                                             name='x%d_next_pred' % level)

    pred_layers = OrderedDict([('x', l_xlevels[0]),
                               *[('x%d' % level, l_xlevels[level]) for level in l_xlevels.keys()],
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               *[('x%d_next_pred' % level, l_xlevels_next_pred[level]) for level in l_xlevels_next_pred.keys()]
                               ])
    return pred_layers


def build_multiscale_dilated_vgg_action_cond_encoder_net(input_shapes,
                                                         num_encoding_levels=5,
                                                         scales=None,
                                                         bilinear_type=None):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    scales = sorted(set(scales))

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    xlevels_c_dim = OrderedDict(zip(range(num_encoding_levels+1), [x_shape[0], 64, 128, 256, 512, 512]))

    # encoding
    l_xlevels = OrderedDict()
    for level in range(num_encoding_levels+1):
        if level == 0:
            l_xlevel = l_x
        elif level < 3:
            l_xlevel = LT.DilatedVggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
        else:
            l_xlevel = LT.DilatedVggEncoding3Layer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
        l_xlevels[level] = l_xlevel

    # multi-scale pyramid
    l_xlevel_scales = OrderedDict()
    for scale in range(scales[-1]+1):
        if scale == 0:
            l_xlevel_scale = l_xlevels[num_encoding_levels]
        else:
            l_xlevel_scale = LT.Downscale2DLayer(l_xlevel_scales[scale-1],
                                                 scale_factor=2,
                                                 name='x%d_%d' % (num_encoding_levels, scale))
        l_xlevel_scales[scale] = l_xlevel_scale

    # bilinear
    l_xlevel_scales_next_pred = OrderedDict()
    for scale in scales:
        l_xlevel_scale = l_xlevel_scales[scale]
        l_xlevel_scale_diff_pred = LT.create_bilinear_layer(l_xlevel_scale,
                                                            l_u,
                                                            scale,
                                                            bilinear_type=bilinear_type,
                                                            name='x%d_%d_diff_pred' % (num_encoding_levels, scale))
        l_xlevel_scales_next_pred[scale] = L.ElemwiseSumLayer([l_xlevel_scale, l_xlevel_scale_diff_pred],
                                                              name='x%d_%d_next_pred' % (num_encoding_levels, scale))

    pred_layers = OrderedDict([('x', l_x),
                               *[('x%d_%d' % (num_encoding_levels, scale), l_xlevel_scales[scale]) for scale in l_xlevel_scales.keys()],
                               ('x_next', l_x_next),
                               *[('x%d_%d_next_pred' % (num_encoding_levels, scale), l_xlevel_scales_next_pred[scale]) for scale in l_xlevel_scales_next_pred.keys()]
                               ])
    return pred_layers


# def build_dilated_vgg_action_cond_encoder_net(input_shapes,
#                                               levels=None,
#                                               num_encoding_levels=5,
#                                               num_decoding_levels=5,
#                                               bilinear_type='convolution'):
#     x_shape, u_shape = input_shapes
#     assert len(x_shape) == 3
#     assert len(u_shape) == 1
#     assert 0 <= num_encoding_levels <= 5
#     assert 0 <= num_decoding_levels <= 5
#     assert len(levels) == 1
#     assert levels[0] == num_encoding_levels
#
#     X_var = T.tensor4('x')
#     U_var = T.matrix('u')
#     X_next_var = T.tensor4('x_next')
#
#     l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
#     l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
#     l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')
#
#     xlevels_c_dim = OrderedDict(zip(range(num_encoding_levels+1), [x_shape[0], 64, 128, 256, 512, 512]))
#
#     # encoding
#     l_xlevels = OrderedDict()
#     for level in range(num_encoding_levels+1):
#         if level == 0:
#             l_xlevel = l_x
#         elif level < 3:
#             l_xlevel = LT.DilatedVggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
#             # TODO: non-trainable encodings
#             LT.set_layer_param_tags(l_xlevel, trainable=False)
#         else:
#             l_xlevel = LT.DilatedVggEncoding3Layer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
#             # TODO: non-trainable encodings
#             LT.set_layer_param_tags(l_xlevel, trainable=False)
#         l_xlevels[level] = l_xlevel
#
#     # bilinear
#     l_xlevels_next_pred = OrderedDict()
#     for level in [num_encoding_levels]:
#         l_xlevel = l_xlevels[level]
#         l_xlevel_diff_pred = LT.create_bilinear_layer(l_xlevel, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_pred' % level)
#         l_xlevels_next_pred[level] = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
#                                                         name='x%d_next_pred' % level)
#         # if tanh:
#         #     l_xlevels_next_pred[level].name += '_unconstrained'
#         #     l_xlevels_next_pred[level] = L.NonlinearityLayer(l_xlevels_next_pred[level], nl.tanh,
#         #                                                      name='x%d_next_pred' % level)
#
#     pred_layers = OrderedDict([('x', l_xlevels[0]),
#                                *[('x%d' % level, l_xlevels[level]) for level in l_xlevels.keys()],
#                                ('x_next', l_x_next),
#                                ('x0_next', l_x_next),
#                                # ('x_next_pred', l_xlevels_next_pred[0]),
#                                *[('x%d_next_pred' % level, l_xlevels_next_pred[level]) for level in l_xlevels_next_pred.keys()]
#                                ])
#     return pred_layers


def build_dilated_vgg_action_cond_encoder_net(input_shapes,
                                              levels=None,
                                              num_encoding_levels=5,
                                              num_decoding_levels=5,
                                              bilinear_type='convolution',
                                              last_fcn_layer=False,
                                              tf_nl_layer=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    assert 0 <= num_encoding_levels <= 5
    assert 0 <= num_decoding_levels <= 5
    assert len(levels) == 1
    assert levels[0] == num_encoding_levels

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    xlevels_c_dim = OrderedDict(zip(range(num_encoding_levels+1), [x_shape[0], 64, 128, 256, 512, 512]))

    # encoding
    l_xlevels = OrderedDict()
    for level in range(num_encoding_levels+1):
        if level == 0:
            l_xlevel = l_x
        elif level < 3:
            l_xlevel = LT.DilatedVggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
        else:
            if level < num_encoding_levels or not last_fcn_layer:
                l_xlevel = LT.DilatedVggEncoding3Layer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d' % level)
            else:
                l_xlevel_ = LT.DilatedVggEncoding3Layer(l_xlevels[level-1], xlevels_c_dim[level], name='x%d_' % level)
                l_xlevel = L.Conv2DLayer(l_xlevel_, xlevels_c_dim[level], filter_size=1, stride=1, pad='same',
                                         nonlinearity=None,
                                         name='x%d' % level)
                LT.set_layer_param_tags(l_xlevel, encoding=True, last_fcn=True)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    for level in [num_encoding_levels]:
        l_xlevel = l_xlevels[level]

        if bilinear_type == 'cross_conv':
            # kernel decoder
            # l_u_fc = L.DenseLayer(l_u, 128 * 5 * 5, name='u_fc')
            # l_u_fcr = L.ReshapeLayer(l_u_fc, (-1, 128, 5, 5), name='u_fcr')
            # l_u_conv1 = L.Conv2DLayer(l_u_fcr, 128, filter_size=5, stride=1, pad='same', name='u_conv1')
            # l_u_conv2 = L.Conv2DLayer(l_u_conv1, 512 * 512, filter_size=5, stride=1, pad='same', name='u_conv2')
            # l_u_conv2r = L.ReshapeLayer(l_u_conv2, (-1, 512, 512, 5, 5), name='u_conv2r')

            l_u_fc = L.DenseLayer(l_u, 512 * 512 * 5 * 5, name='u_fc')
            l_u_fcr = L.ReshapeLayer(l_u_fc, (-1, 512, 512, 5, 5), name='u_fcr')

            l_x_shape = L.get_output_shape(l_xlevel)
            l_xlevel_diff_pred = LT.CrossConv2DLayer([l_xlevel, l_u_fcr],
                                                      num_filters=l_x_shape[1],
                                                      filter_size=5,
                                                      stride=1,
                                                      pad='same',
                                                      name='x%d_diff_pred' % level)
        else:
            l_xlevel_diff_pred = LT.create_bilinear_layer(l_xlevel, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_pred' % level)
        # l_xlevels_next_pred[level] = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
        #                                                 name='x%d_next_pred' % level)
        if not tf_nl_layer:
            l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
                                                    name='x%d_next_pred' % level)
        else:
            l_xlevel_next_pred_ = L.ElemwiseSumLayer([l_xlevel, l_xlevel_diff_pred],
                                                     name='x%d_next_pred_' % level)
            l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred_, nonlinearity=nl.rectify,
                                                     name='x%d_next_pred' % level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred

    # decoding
    l_xlevels_dec = OrderedDict()
    for level in range(num_decoding_levels+1)[::-1]:
        if level == num_decoding_levels:
            l_xlevel_dec = l_xlevels_next_pred[num_encoding_levels]
        else:
            # TODO: use encoding channel dimensions
            l_xlevel_dec = L.Conv2DLayer(l_xlevels_dec[level+1], xlevels_c_dim[level], filter_size=3, stride=1, pad='same',
                                         nonlinearity=nl.rectify if level else None,
                                         name='x%d_dec_conv' % level)
            LT.set_layer_param_tags(l_xlevel_dec, decoding=True)
        l_xlevels_dec[level] = l_xlevel_dec

    pred_layers = OrderedDict([('x', l_xlevels[0]),
                               *[('x%d' % level, l_xlevels[level]) for level in l_xlevels.keys()],
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_xlevels_dec[0]),
                               ('x0_next_pred', l_xlevels_dec[0]),
                               *[('x%d_next_pred' % level, l_xlevels_next_pred[level]) for level in l_xlevels_next_pred.keys()],
                               *[('x%d_dec' % level, l_xlevels_dec[level]) for level in l_xlevels_dec.keys()]
                               ])
    return pred_layers


# l_x = predictor.pred_layers['x5']
# l_u = predictor.pred_layers['u']
# l_x_u_outer = predictor.pred_layers['x5_u_outer']
# l_x_diff_pred = predictor.pred_layers['x5_diff_pred']
#
# x = np.random.random((1,) + L.get_output_shape(l_x)[1:]).astype(np.float32)
# u = np.random.random((1,) + L.get_output_shape(l_u)[1:]).astype(np.float32)
# W = l_x_diff_pred.get_params()[0].get_value()[0]  # first channel for now
# b = l_x_diff_pred.get_params()[1].get_value()[0]
#
# T_x, T_u, T_x_u_outer, T_x_diff_pred = L.get_output([l_x, l_u, l_x_u_outer, l_x_diff_pred])
# input_value_dict = {T_x: x, T_u: u}
# x_u_outer = T_x_u_outer.eval(input_value_dict)
# x_diff_pred = T_x_diff_pred.eval(input_value_dict)
#
# k = 5
# n = 6
# w_sym = np.array(['%d%d' % (i+1, j+1) for i in range(k) for j in range(k)]).reshape((k, k))
# np.c_[w_sym, np.zeros((k, n-k), dtype='<U2')].flatten()
#
# np.r_[np.c_[w_sym, np.zeros((k, n-k), dtype='<U2')],
#       np.c_[np.zeros((n-k, k), dtype='<U2'), np.zeros((n-k, n-k), dtype='<U2')]].flatten()
#
# M = np.array(['00'] * (n * n * n)).reshape((n, n, n))
# for i in range(n):
#     m_slice = slice(max(i - k // 2, 0), min(i - k // 2 + k, n))
#     print(m_slice)
#     w_slice = slice(max(i - k // 2, 0), min(i - k // 2 + k, k))
#     M[i, m_slice, m_slice] = w_sym[w_slice, w_slice]
#
# img = np.squeeze(x_u_outer)
# kern = W
# out = np.zeros((x_diff_pred.shape[2:]), dtype=img.dtype)
# assert img.shape[0] == kern.shape[0]
#
# for i in range(out.shape[0]):
#     for j in range(out.shape[1]):
#         for c in range(img.shape[0]):
#             for k in range(kern.shape[1]):
#                 for l in range(kern.shape[2]):
#                     if (0 <= kern.shape[1] - k - 1 < kern.shape[1]) and \
#                         (0 <= kern.shape[2] - l - 1 < kern.shape[2]) and \
#                         (0 <= i - kern.shape[1] // 2 + k < img.shape[1]) and \
#                         (0 <= j - kern.shape[2] // 2 + l < img.shape[2]):
#                         out[i, j] += kern[c, kern.shape[1] - k - 1, kern.shape[2] - l - 1] * \
#                                      img[c, i - kern.shape[1] // 2 + k, j - kern.shape[2] // 2 + l]
# out += b

# x_shape = (512, 32, 32)
# u_shape = (6,)
# X_var = T.tensor4('x')
# U_var = T.matrix('u')
# l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
# l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
# l_x_u_outer = LT.OuterProductLayer([l_x, l_u], name='x_u_outer')
# l_x_diff_pred = L.Conv2DLayer(l_x_u_outer, x_shape[0], filter_size=3, stride=1, pad='same',
#                               untie_biases=True, nonlinearity=None, name='x_diff_pred')
# x = np.random.random((1,) + L.get_output_shape(l_x)[1:]).astype(np.float32)
# u = np.random.random((1,) + L.get_output_shape(l_u)[1:]).astype(np.float32)
# T_x, T_u, T_x_u_outer, T_x_diff_pred = L.get_output([l_x, l_u, l_x_u_outer, l_x_diff_pred])
# input_value_dict = {T_x: x, T_u: u}
#
# import time
#
# jac_var = theano.gradient.jacobian(T_x_diff_pred.flatten(), U_var)
# jac_fn = theano.function([X_var, U_var], jac_var)
# start_time = time.time()
# J = jac_fn(x, u).squeeze(axis=1)
# print(time.time() - start_time)
#
# g_var = theano.gradient.Rop(T_x_diff_pred.flatten(), U_var, np.eye(6)[0][None, :])
# g_fn = theano.function([X_var, U_var], g_var)
# start_time = time.time()
# g = g_fn(x, u)
# print(time.time() - start_time)
#
# # TODO: u might be part of computation graph
# g_var, updates = theano.scan(lambda eval_points, T_x_diff_pred, U_var: theano.gradient.Rop(T_x_diff_pred, U_var, eval_points),
#                          sequences=np.eye(6)[:, None, :],
#                          non_sequences=[T_x_diff_pred.flatten(), U_var])
# g_fn = theano.function([X_var, U_var], g_var)
# start_time = time.time()
# g = g_fn(x, u)
# print(time.time() - start_time)
#
# jac_var = theano.gradient.Rop(T_x_diff_pred, U_var, T.eye(U_var.shape[-1]))
# jac_fn = theano.function([X_var, U_var], jac_var)
# start_time = time.time()
# J = jac_fn(np.concatenate([x] * 6), np.concatenate([u] * 6))
# J = J.reshape((6, -1))
# Q = J.dot(J.T)
# print(time.time() - start_time)
#
# jac_var = theano.gradient.Rop(T_x_diff_pred, U_var, T.eye(U_var.shape[-1]))
# x_diff_pred_jac_fn = theano.function([X_var, U_var], [T_x_diff_pred, jac_var])
# start_time = time.time()
# x_diff_pred, J = x_diff_pred_jac_fn(np.concatenate([x] * 6), np.concatenate([u] * 6))
# print(time.time() - start_time)
#
# jac_flatten_var = T.flatten(jac_var, outdim=2)
# Q_var = jac_flatten_var.dot(jac_flatten_var.T)
# Q_fn = theano.function([X_var, U_var], Q_var)
# start_time = time.time()
# Q = Q_fn(np.concatenate([x] * 6), np.concatenate([u] * 6))
# print(time.time() - start_time)
#
# jac_rep_var = theano.clone(jac_var, replace={X_var: T.tile(X_var, (6, 1, 1, 1)), U_var: T.tile(U_var, (6, 1))})
# jac_rep_fn = theano.function([X_var, U_var], jac_rep_var)
# start_time = time.time()
# J_rep = jac_rep_fn(x, u)
# print(time.time() - start_time)
#
# x_diff_pred_fn = theano.function([X_var, U_var], T_x_diff_pred)
# start_time = time.time()
# x_diff_pred = x_diff_pred_fn(x, u)
# print(time.time() - start_time)
#
# start_time = time.time()
# y = x_diff_pred[0].flatten()
# u = np.linalg.solve(J.dot(J.T), J.dot(y))
# print(time.time() - start_time)
#
# u_var = T.slinalg.solve(T.dot(jac_flatten_var, jac_flatten_var.T), jac_flatten_var.dot(T.flatten(T_x_diff_pred, outdim=2).T))
# u_fn = theano.function([X_var, U_var], u_var)
# start_time = time.time()
# u2 = u_fn(np.concatenate([x] * 6), np.concatenate([u] * 6))
# print(time.time() - start_time)
#
#
# u_prev = np.random.random((6,))
# image = (np.random.random((480, 640, 3))*255).astype(np.uint8)
# J = predictor.feature_jacobian(image, u_prev)



# X_var = T.tensor4('x')
# batch_size = 32
# x_shape = 8, 16, 16
# num_filters = 16
# l_x = L.InputLayer(shape=(batch_size,) + x_shape, input_var=X_var, name='x')
# # l_x1 = L.Conv2DLayer(l_x, num_filters, filter_size=5, stride=2, pad='same')
# # x = L.get_output(l_x)
# # x1 = L.get_output(l_x1)
#
# tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
# W_var = tensor5('W')
# W_shape = (num_filters, x_shape[0], 5, 5)
# l_W = L.InputLayer(shape=(batch_size,) + W_shape, input_var=W_var, name='W')
#
# l_x1 = MultiFilterConv2DLayer([l_x, l_W], num_filters, filter_size=5, stride=2, pad='same')
#
# x_value = np.random.random((batch_size, *x_shape)).astype(np.float32)
# x1_value = x1.eval({x: x_value})
#
# W = l_x1.get_params()[0]
# output_shape = l_x1.output_shape
# output_size = output_shape[2:]
# stride = l_x1.stride
#
# op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
#     imshp=output_shape,
#     kshp=tuple(W.shape.eval()),
#     subsample=stride,
#     border_mode='half',
#     filter_flip=True)
# x_diff = op(W, x1, output_size)


def build_cross_conv_net(input_shapes, num_scales=4):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    # multi-scale pyramid
    l_xscales = OrderedDict()
    for scale in range(num_scales):
        if scale == 0:
            l_xscale = l_x
        else:
            l_xscale = LT.Downscale2DLayer(l_xscales[scale-1], scale_factor=2, name='x%d' % scale)
            # l_xscale = L.Pool2DLayer(l_xscales[scale-1], pool_size=2, mode='average_inc_pad', name='x%d' % scale)
        l_xscales[scale] = l_xscale

    # image encoder
    xlevels_c_dim = [64, 64, 64, 32]
    l_xscales_levels = OrderedDict()
    for scale in range(num_scales):
        l_xlevel = l_xscales[scale]
        l_xscales_levels[scale] = OrderedDict()
        for level in range(len(xlevels_c_dim)):
            l_xlevel = L.Conv2DLayer(l_xlevel, xlevels_c_dim[level], filter_size=5, stride=1, pad='same',
                                     name='x%d_%d' % (scale, level))
            l_xlevel = L.batch_norm(l_xlevel)
            if (level + 1) % 2 == 0:
                l_xlevel = L.MaxPool2DLayer(l_xlevel, pool_size=2, stride=2, pad=0,
                                            name='x%d_%d_pool' % (scale, level))
            l_xscales_levels[scale][level] = l_xlevel

    # kernel decoder
    l_u_fc = L.DenseLayer(l_u, 128 * 5 * 5, name='u_fc')
    l_u_fcr = L.ReshapeLayer(l_u_fc, (-1, 128, 5, 5), name='u_fcr')
    l_u_conv1 = L.Conv2DLayer(l_u_fcr, 128, filter_size=5, stride=1, pad='same', name='u_conv1')
    l_u_conv2 = L.Conv2DLayer(l_u_conv1, 128, filter_size=5, stride=1, pad='same', name='u_conv2')
    l_u_conv2_slices = []
    for scale in range(num_scales):
        l_u_conv2_slice = L.SliceLayer(l_u_conv2, indices=slice(scale, scale + 32), axis=1,
                                       name='u_conv2_%d' % scale)
        l_u_conv2_slice = L.DimshuffleLayer(l_u_conv2_slice, (0, 'x', 1, 2, 3))
        l_u_conv2_slices.append(l_u_conv2_slice)

    l_xscales_cross_conv = OrderedDict()
    for scale in range(num_scales):
        l_xscale = list(l_xscales_levels[scale].values())[-1]
        l_u_conv2_scale = l_u_conv2_slices[scale]
        # (N, 32, 64, 64)
        # (1, 32, 5, 5)
        # (N, 1, 64, 64)
        #
        # (1, N, 32, 64, 64)
        # (1, N, 32, 5, 5)
        # (1, 1, 1, 64, 64)
        # [Ns, Ts, C, Hs, Ws]
        # [Nf, Tf, C, Hf, Wf]
        # (Ns, Ts - Tf + 1, Nf, Hs - Hf + 1, Ws - Wf + 1)
        #
        # (N, 32, 64, 64)
        # (N, 1, 32, 5, 5)
        # (N, 1, 64, 64)
        #
        # (N, 32, 64, 64)
        # (N, 32, 5, 5)
        # (N, N, 64, 64)
        #
        # (N, 32N, 64, 64)
        # (1, 32N, 5, 5)
        # (N, 1, 64, 64)
        #
        # (1, 32N, 64, 64)
        # (1, N, 5, 5)
        # (1, N, 64, 64)
        #
        # from theano.tensor.nnet import conv3d2d
        # from theano.tensor.nnet import conv2d
        # # conv2d(input, filters, input_shape=None, filter_shape=None,
        # #        border_mode='valid', subsample=(1, 1), filter_flip=True,
        # #        image_shape=None, filter_dilation=(1, 1), **kwargs):
        #
        #
        # xscale = L.get_output(l_xscale)
        # u_conv2_scale = L.get_output(l_u_conv2_scale)
        #
        # def batched_conv2d(t):
        #     xscale, u_conv2_scale = t
        #     outputs, updates = theano.scan(
        #         fn=lambda xscale, u_conv2_scale: conv2d(xscale.dimshuffle(('x', 0, 1, 2)),
        #                                                 u_conv2_scale.dimshuffle(('x', 0, 1, 2)),
        #                                                 filter_shape=(5, 5),
        #                                                 border_mode='half'),
        #         outputs_info=None,
        #         sequences=[xscale, u_conv2_scale])
        #     return T.concatenate(outputs)
        #
        # output_shape = (None, 1, *L.get_output_shape(l_xscale)[2:])
        # l_xscale_cross_conv = L.ExpressionLayer((l_xscale, l_u_conv2_scale), batched_conv2d, output_shape=output_shape)

        l_xscale_cross_conv = LT.CrossConv2DLayer([l_xscale, l_u_conv2_scale],
                                                  num_filters=1,
                                                  filter_size=5,
                                                  stride=1,
                                                  pad='same',
                                                  name='x%d_cross_conv' % scale)
        # l_xscale_conv = L.Conv2DLayer(l_xscale, batch_size, filter_size=5, stride=1, pad='same',
        #                               W=L.get_output(l_u_conv2_scale),
        #                               name='x%d_cross_conv' % scale)
        # l_xscale_conv_re = L.ReshapeLayer(l_xscale_conv, (-1, [2], [3]))
        # l_xscale_cross_conv_re = L.SliceLayer(l_xscale_conv_re, indices=slice(0, None, batch_size+1), axis=0)
        # l_xscale_cross_conv = L.ReshapeLayer(l_xscale_cross_conv_re, ([0], 1, [1], [2]))

        # # l_xscale = L.ReshapeLayer(l_xscale, (32, [1], [2], [3]))
        # # l_u_conv2_scale = L.ReshapeLayer(l_u_conv2_slices[scale], (32, [1], [2], [3]))
        # l_xscale_samples_cross_conv = []
        # for sample in range(batch_size):
        #     l_xscale_sample = L.SliceLayer(l_xscale, indices=slice(sample, sample + 1), axis=0)
        #     l_u_conv2_sample = L.SliceLayer(l_u_conv2_slices[scale], indices=slice(sample, sample + 1), axis=0)
        #     l_xscale_sample_cross_conv = L.Conv2DLayer(l_xscale_sample, 1, filter_size=5, stride=1, pad='same',
        #                                                W=L.get_output(l_u_conv2_sample),
        #                                                name='x%d_%d_cross_conv' % (scale, sample))
        #     l_xscale_samples_cross_conv.append(l_xscale_sample_cross_conv)
        # l_xscale_cross_conv = L.ConcatLayer(l_xscale_samples_cross_conv, axis=0)

        # ideally, want to a per-sample convolution as in the following snippet
        # but Conv2DLayer doesn't support that
        # l_u_conv2 = L.ReshapeLayer(l_u_conv2_slices[scale], (-1, 1, 32, 5, 5))
        # l_xscale_cross_conv = L.Conv2DLayer(l_xscale, 1, filter_size=5, stride=1, pad='same',
        #                                     W=L.get_output(l_u_conv2),
        #                                     name='x%d_cross_conv' % scale)
        if scale > 0:
            l_xscale_cross_conv = L.Upscale2DLayer(l_xscale_cross_conv, scale_factor=2 ** scale)
        l_xscales_cross_conv[scale] = l_xscale_cross_conv
    l_x_cross_conv = L.ConcatLayer(l_xscales_cross_conv.values(), axis=1,
                                   name='x_cross_conv')

    l_x2_diff_pred = L.Conv2DLayer(l_x_cross_conv, 128, filter_size=9, stride=1, pad='same',
                                   name='l_x2_diff_pred')
    l_x2_diff_pred = L.batch_norm(l_x2_diff_pred)
    l_x1_diff_pred = L.Conv2DLayer(l_x2_diff_pred, 128, filter_size=1, stride=1, pad='same',
                                   name='l_x1_diff_pred')
    l_x1_diff_pred = L.batch_norm(l_x1_diff_pred)
    l_x_diff_pred = L.Conv2DLayer(l_x1_diff_pred, 3, filter_size=1, stride=1, pad='same',
                                  nonlinearity=None,
                                  name='l_x_diff_pred')
    l_x_diff_pred = L.batch_norm(l_x_diff_pred)

    l_x_next_pred = L.ElemwiseSumLayer([l_xscales[2], l_x_diff_pred], name='x_next_pred')

    # TODO: remove last mappings after the names have been fixed in the model file
    pred_layers = OrderedDict([('x', l_x),
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_x_next_pred),
                               ('x0_next_pred', l_x_next_pred),
                               ('u', l_u),
                               ('x2_diff_pred', l_x2_diff_pred),
                               ('x1_diff_pred', l_x1_diff_pred),
                               ('x_diff_pred', l_x_diff_pred)
                               ])
    return pred_layers


def build_dilated_cross_conv_net(input_shapes, num_scales=4):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    # multi-scale pyramid
    l_xscales = OrderedDict()
    for scale in range(num_scales):
        if scale == 0:
            l_xscale = l_x
        else:
            l_xscale = LT.Downscale2DLayer(l_xscales[scale-1], scale_factor=2, name='x%d' % scale)
            # l_xscale = L.Pool2DLayer(l_xscales[scale-1], pool_size=2, mode='average_inc_pad', name='x%d' % scale)
        l_xscales[scale] = l_xscale

    # image encoder
    xlevels_c_dim = [64, 64, 64, 32]
    l_xscales_levels = OrderedDict()
    for scale in range(num_scales):
        l_xlevel = l_xscales[scale]
        l_xscales_levels[scale] = OrderedDict()
        for level in range(len(xlevels_c_dim)):
            l_xlevel = L.Conv2DLayer(l_xlevel, xlevels_c_dim[level], filter_size=5, stride=1, pad='same',
                                     filter_dilation=2 if ((level + 1) % 2 == 0) else 1,
                                     name='x%d_%d' % (scale, level))
            l_xlevel = L.batch_norm(l_xlevel)
            l_xscales_levels[scale][level] = l_xlevel

    # kernel decoder
    l_u_fc = L.DenseLayer(l_u, 128 * 5 * 5, name='u_fc')
    l_u_fcr = L.ReshapeLayer(l_u_fc, (-1, 128, 5, 5), name='u_fcr')
    l_u_conv1 = L.Conv2DLayer(l_u_fcr, 128, filter_size=5, stride=1, pad='same', name='u_conv1')
    l_u_conv2 = L.Conv2DLayer(l_u_conv1, 128, filter_size=5, stride=1, pad='same', name='u_conv2')
    l_u_conv2_slices = []
    for scale in range(num_scales):
        l_u_conv2_slice = L.SliceLayer(l_u_conv2, indices=slice(scale, scale + 32), axis=1,
                                       name='u_conv2_%d' % scale)
        l_u_conv2_slice = L.DimshuffleLayer(l_u_conv2_slice, (0, 'x', 1, 2, 3))
        l_u_conv2_slices.append(l_u_conv2_slice)

    l_xscales_cross_conv = OrderedDict()
    for scale in range(num_scales):
        l_xscale = list(l_xscales_levels[scale].values())[-1]
        l_u_conv2_scale = l_u_conv2_slices[scale]
        l_xscale_cross_conv = LT.CrossConv2DLayer([l_xscale, l_u_conv2_scale],
                                                  num_filters=1,
                                                  filter_size=5,
                                                  stride=1,
                                                  pad='same',
                                                  name='x%d_cross_conv' % scale)
        if scale > 0:
            l_xscale_cross_conv = L.Upscale2DLayer(l_xscale_cross_conv, scale_factor=2 ** scale)
        l_xscales_cross_conv[scale] = l_xscale_cross_conv
    l_x_cross_conv = L.ConcatLayer(l_xscales_cross_conv.values(), axis=1,
                                   name='x_cross_conv')

    l_x2_diff_pred = L.Conv2DLayer(l_x_cross_conv, 128, filter_size=9, stride=1, pad='same',
                                   name='l_x2_diff_pred')
    l_x2_diff_pred = L.batch_norm(l_x2_diff_pred)
    l_x1_diff_pred = L.Conv2DLayer(l_x2_diff_pred, 128, filter_size=1, stride=1, pad='same',
                                   name='l_x1_diff_pred')
    l_x1_diff_pred = L.batch_norm(l_x1_diff_pred)
    l_x_diff_pred = L.Conv2DLayer(l_x1_diff_pred, 3, filter_size=1, stride=1, pad='same',
                                  nonlinearity=None,
                                  name='l_x_diff_pred')
    l_x_diff_pred = L.batch_norm(l_x_diff_pred)

    l_x_next_pred = L.ElemwiseSumLayer([l_x, l_x_diff_pred], name='x_next_pred')

    # TODO: remove last mappings after the names have been fixed in the model file
    pred_layers = OrderedDict([('x', l_x),
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_x_next_pred),
                               ('x0_next_pred', l_x_next_pred),
                               ('u', l_u),
                               ('x2_diff_pred', l_x2_diff_pred),
                               ('x1_diff_pred', l_x1_diff_pred),
                               ('x_diff_pred', l_x_diff_pred)
                               ])
    return pred_layers


def build_prednet_net(input_shapes, num_levels=1):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1

    tensor5 = T.TensorType(theano.config.floatX, (False,) * 5)
    X_var = tensor5('x')  # (n_batch, n_time_steps, num_input_channels, input_rows, input_columns)
    U_var = T.matrix('u')

    # TODO
    x_shape = tuple(int(dim) for dim in x_shape)

    seq_len = 2
    l_x = L.InputLayer(shape=(None, seq_len) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    xlevels_c_dim = [x_shape[0], 32, 64, 128, 256]

    l_hid_to_hid = [None] * num_levels
    l_error_to_hid = [None] * num_levels
    l_upper_hid_to_hid = [None] * (num_levels - 1)
    for level in range(num_levels)[::-1]:
        # should contain operations for the 4 gates
        l_in_hid = L.InputLayer(shape=(None,
                                       xlevels_c_dim[level],
                                       x_shape[-2] // (2 ** level),
                                       x_shape[-1] // (2 ** level)))
        l_in_error = L.InputLayer(shape=(None,
                                         2 * xlevels_c_dim[level],
                                         x_shape[-2] // (2 ** level),
                                         x_shape[-1] // (2 ** level)))
        l_hid_to_hid[level] = L.Conv2DLayer(l_in_hid, num_filters=4*xlevels_c_dim[level], filter_size=3, pad='same')
        l_error_to_hid[level] = L.Conv2DLayer(l_in_error, num_filters=4*xlevels_c_dim[level], filter_size=3, pad='same', b=None)
        if level < num_levels - 1:
            l_in_upper_hid = L.InputLayer(shape=(None,
                                                 xlevels_c_dim[level + 1],
                                                 x_shape[-2] // (2 ** (level + 1)),
                                                 x_shape[-1] // (2 ** (level + 1))))
            l_upper_hid_to_hid[level] = L.Conv2DLayer(L.Upscale2DLayer(l_in_upper_hid, scale_factor=2),
                                                      num_filters=4*xlevels_c_dim[level], filter_size=3, pad='same', b=None)

    l_input_and_hid_to_error = [None] * num_levels
    l_error_to_upper_input = [None] * (num_levels - 1)
    for level in range(num_levels):
        l_in_input = L.InputLayer(shape=(None,
                                         xlevels_c_dim[level],
                                         x_shape[-2] // (2 ** level),
                                         x_shape[-1] // (2 ** level)))
        l_in_hid = L.InputLayer(shape=(None,
                                       xlevels_c_dim[level],
                                       x_shape[-2] // (2 ** level),
                                       x_shape[-1] // (2 ** level)))
        l_in_error = L.InputLayer(shape=(None,
                                         2 * xlevels_c_dim[level],
                                         x_shape[-2] // (2 ** level),
                                         x_shape[-1] // (2 ** level)))
        if level == 0:
            nonlinearity = lambda x: T.clip(x, -1, 1)
        else:
            nonlinearity = nl.rectify
        l_input_hat = L.NonlinearityLayer(L.Conv2DLayer(l_in_hid, num_filters=xlevels_c_dim[level], filter_size=3, pad='same'),
                                          nonlinearity=nonlinearity, name='x%d_hat' % level)
        l_error_pos = L.ElemwiseSumLayer([l_in_input, l_input_hat], coeffs=[1.0, -1.0], name='e%d_pos' % level)
        l_error_neg = L.ElemwiseSumLayer([l_in_input, l_input_hat], coeffs=[-1.0, 1.0], name='e%d_neg' % level)
        l_error_pos_neg = L.ConcatLayer([l_error_pos, l_error_neg], axis=1, name='e%d_pos_neg' % level)
        l_input_and_hid_to_error[level] = L.NonlinearityLayer(l_error_pos_neg, nonlinearity=nl.rectify, name='e%d' % level)
        if level < num_levels - 1:
            l_error_to_upper_input[level] = L.Pool2DLayer(
                L.Conv2DLayer(l_in_error, xlevels_c_dim[level + 1], filter_size=3, pad='same'),
                pool_size=2, stride=2, pad=0, mode='max')

    l_x_next_pred = LT.PredNetLSTMLayer(l_x,
                                        l_hid_to_hid,
                                        l_error_to_hid,
                                        l_upper_hid_to_hid,
                                        l_input_and_hid_to_error,
                                        l_error_to_upper_input,
                                        only_return_final=True)

    # TODO: remove last mappings after the names have been fixed in the model file
    pred_layers = OrderedDict([('x', l_x),
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_x_next_pred),
                               ('x0_next_pred', l_x_next_pred),
                               ('u', l_u),
                               ])
    return pred_layers


def build_laplacian_action_cond_encoder_net(input_shapes, levels=None, bilinear_type='share', tanh=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    # encoding
    l_xlevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x
        else:
            l_xlevel = L.Pool2DLayer(l_xlevels[level-1], pool_size=2, mode='average_inc_pad', name='x%d' % level)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    l_xlevels_res = OrderedDict()
    l_xlevels_next_pred_res = OrderedDict()
    for level in levels[::-1]:
        l_xlevel = l_xlevels[level]
        if level == levels[-1]:
            l_xlevels_res[level] = l_xlevel
        else:
            l_xlevel_blur = L.Upscale2DLayer(l_xlevels[level+1], scale_factor=2, name='x%d_blur' % level)
            l_xlevels_res[level] = L.ElemwiseSumLayer([l_xlevel, l_xlevel_blur], coeffs=[1.0, -1.0],
                                                      name='x%d_res' % level)
        l_xlevel_diff_pred_res = LT.create_bilinear_layer(l_xlevels_res[level], l_u, level, bilinear_type=bilinear_type,
                                                          name='x%d_diff_pred_res' % level)
        l_xlevels_next_pred_res[level] = L.ElemwiseSumLayer([l_xlevels_res[level], l_xlevel_diff_pred_res],
                                                            name='x%d_next_pred_res' % level)
        if level == levels[-1]:
            l_xlevels_next_pred[level] = l_xlevels_next_pred_res[level]
            if tanh:
                l_xlevels_next_pred[level] = L.NonlinearityLayer(l_xlevels_next_pred[level], nl.tanh,
                                                                 name='x%d_next_pred' % level)
        else:
            l_xlevel_next_pred_blur = L.Upscale2DLayer(l_xlevels_next_pred[level+1], scale_factor=2,
                                                       name='x%d_next_pred_blur' % level)
            l_xlevels_next_pred[level] = L.ElemwiseSumLayer([l_xlevels_next_pred_res[level], l_xlevel_next_pred_blur],
                                                            name='x%d_next_pred' % level)
            if tanh:
                l_xlevels_next_pred[level].name += '_unconstrained'
                l_xlevels_next_pred[level] = L.NonlinearityLayer(l_xlevels_next_pred[level], nl.tanh,
                                                                 name='x%d_next_pred' % level)

    pred_layers = OrderedDict([('x', l_xlevels[0]),
                               *[('x%d' % level, l_xlevels[level]) for level in l_xlevels.keys()],
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               *[('x%d_next_pred' % level, l_xlevels_next_pred[level]) for level in l_xlevels_next_pred.keys()],
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               *[('x%d_res' % level, l_xlevels_res[level]) for level in l_xlevels_res.keys()],
                               *[('x%d_next_pred_res' % level, l_xlevels_next_pred_res[level]) for level in l_xlevels_next_pred_res.keys()],
                               ])
    return pred_layers


def build_fcn_action_cond_encoder_only_net(input_shapes, levels=None, x1_c_dim=16, num_downsample=0, bilinear_type='share', ladder_loss=True, batch_norm=False, **kwargs):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

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
                                       nonlinearity=None,
                                       name='x%d_conv1'%level)
            if batch_norm:
                l_xlevel_1 = L.BatchNormLayer(l_xlevel_1, name='x%d_bn1'%level)
            l_xlevel_1 = L.NonlinearityLayer(l_xlevel_1, nonlinearity=nl.rectify, name='x%d_1'%level)
            l_xlevel_2 = L.Conv2DLayer(l_xlevel_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       nonlinearity=None,
                                       name='x%d_conv2'%level)
            if batch_norm:
                l_xlevel_2 = L.BatchNormLayer(l_xlevel_2, name='x%d_bn2'%level)
            l_xlevel_2 = L.NonlinearityLayer(l_xlevel_2, nonlinearity=nl.rectify, name='x%d_2'%level)
            l_xlevel = L.MaxPool2DLayer(l_xlevel_2, pool_size=2, stride=2, pad=0, name='x%d'%level)
        l_xlevels[level] = l_xlevel
        l_ylevels[level] = L.FlattenLayer(l_xlevel, name='y%d'%level)
    l_y = L.ConcatLayer(l_ylevels.values(), name='y')

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_pred = LT.create_bilinear_layer(l_xlevel, l_u, bilinear_type=bilinear_type, name='x%d_diff_pred'%level)
        l_xlevel_next_pred = L.ElemwiseMergeLayer([l_xlevel, l_xlevel_diff_pred], T.add, name='x%d_next_pred'%level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred
        l_ylevels_diff_pred[level] = L.FlattenLayer(L.ElemwiseSumLayer([l_xlevel_next_pred, l_xlevels[level]], coeffs=[1.0, -1.0], name='x%d_diff_pred'%level), name='y%d_diff_pred'%level)
    l_y_diff_pred = L.ConcatLayer(l_ylevels_diff_pred.values(), name='y_diff_pred')

    l_x0_next_pred = l_xlevels_next_pred[0]

    # preprocess
    l_x0_next = l_x_next

    encoder_layers = OrderedDict((layer.name, layer) for layer in lasagne.layers.get_all_layers(l_xlevels[levels[-1]]) if layer.name is not None)
    # encoding of next image
    l_xlevels_next = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel_next = l_x0_next
        else:
            if level == 1:
                xlevelm1_c_dim = x0_shape[0]
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevelm1_c_dim
            l_xlevel_next_1 = L.Conv2DLayer(l_xlevels_next[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                            W=encoder_layers['x%d_conv1'%level].W,
                                            b=encoder_layers['x%d_conv1'%level].b,
                                            nonlinearity=None,
                                            name='x%d_next_conv1'%level)
            if batch_norm: # TODO: share batch normalization variables?
                l_xlevel_next_1 = L.BatchNormLayer(l_xlevel_next_1,
                                                   beta=encoder_layers['x%d_bn1'%level].beta,
                                                   gamma=encoder_layers['x%d_bn1'%level].gamma,
                                                   mean=encoder_layers['x%d_bn1'%level].mean,
                                                   inv_std=encoder_layers['x%d_bn1'%level].inv_std,
                                                   name='x%d_next_bn1'%level)
            l_xlevel_next_1 = L.NonlinearityLayer(l_xlevel_next_1, nonlinearity=nl.rectify, name='x%d_next_1'%level)
            l_xlevel_next_2 = L.Conv2DLayer(l_xlevel_next_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                            W=encoder_layers['x%d_conv2'%level].W,
                                            b=encoder_layers['x%d_conv2'%level].b,
                                            nonlinearity=None,
                                            name='x%d_next_conv2'%level)
            if batch_norm:
                l_xlevel_next_2 = L.BatchNormLayer(l_xlevel_next_2,
                                                   beta=encoder_layers['x%d_bn2'%level].beta,
                                                   gamma=encoder_layers['x%d_bn2'%level].gamma,
                                                   mean=encoder_layers['x%d_bn2'%level].mean,
                                                   inv_std=encoder_layers['x%d_bn2'%level].inv_std,
                                                   name='x%d_next_bn2'%level)
            l_xlevel_next_2 = L.NonlinearityLayer(l_xlevel_next_2, nonlinearity=nl.rectify, name='x%d_next_2'%level)
            l_xlevel_next = L.MaxPool2DLayer(l_xlevel_next_2, pool_size=2, stride=2, pad=0, name='x%d_next'%level)
        l_xlevels_next[level] = l_xlevel_next

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = 0
    loss_deterministic = 0
    for level in levels:
        loss += loss_fn(lasagne.layers.get_output(l_xlevels_next[level]),
                        lasagne.layers.get_output(l_xlevels_next_pred[level]))
        loss_deterministic += loss_fn(lasagne.layers.get_output(l_xlevels_next[level], deterministic=True),
                                      lasagne.layers.get_output(l_xlevels_next_pred[level], deterministic=True))

    net_name = 'FcnActionCondEncoderOnlyNet'
    net_name +='_levels' + ''.join([str(level) for level in levels])
    net_name += '_x1cdim' + str(x1_c_dim)
    net_name += '_numds' + str(num_downsample)
    net_name += '_bi' + bilinear_type
    net_name += '_ladder' + str(int(ladder_loss))
    net_name += '_bn' + str(int(batch_norm))

    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred),
                               ('y', l_y)])
    for level in set([0, *levels]):
        pred_layers.update([('x%d_next_pred'%level, l_xlevels_next_pred[level])])
    return net_name, input_vars, pred_layers, loss, loss_deterministic


def build_small_cifar10nin_net(input_shapes, **kwargs):
    x_shape, u_shape = input_shapes

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x0 = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    l_x1 = L.Conv2DLayer(l_x0,
                         num_filters=192,
                         filter_size=5,
                         pad=2,
                         flip_filters=False,
                         name='x1')
    l_x2 = L.Conv2DLayer(l_x1, num_filters=160, filter_size=1, flip_filters=False,
                         name='x2')
    l_x3 = L.Conv2DLayer(l_x2, num_filters=96, filter_size=1, flip_filters=False,
                         name='x3')
    l_x4 = L.MaxPool2DLayer(l_x3,
                            pool_size=3,
                            stride=2,
                            ignore_border=False,
                            name='x4')

    l_x4_diff_pred = LT.BilinearLayer([l_x4, l_u], axis=2, name='x4_diff_pred')
    l_x4_next_pred = L.ElemwiseMergeLayer([l_x4, l_x4_diff_pred], T.add, name='x4_next_pred')

    l_x3_next_pred = LT.Deconv2DLayer(l_x4_next_pred,
                                   num_filters=96,
                                   filter_size=3,
                                   stride=2,
                                   nonlinearity=None,
                                   name='x3_next_pred')
    l_x2_next_pred = LT.Deconv2DLayer(l_x3_next_pred, num_filters=160, filter_size=1, flip_filters=False,
                                   name='x2_next_pred')
    l_x1_next_pred = LT.Deconv2DLayer(l_x2_next_pred, num_filters=192, filter_size=1, flip_filters=False,
                                   name='x1_next_pred')
    l_x0_next_pred = LT.Deconv2DLayer(l_x1_next_pred,
                                   num_filters=3,
                                   filter_size=5,
                                   pad=2,
                                   flip_filters=False,
                                   nonlinearity=None,
                                   name='x0_next_pred')

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = loss_fn(X_next_var, lasagne.layers.get_output(l_x0_next_pred))

    net_name = 'SmallCifar10ninNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('x0_next_pred', l_x0_next_pred)])
    return net_name, input_vars, pred_layers, loss
