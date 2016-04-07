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


def build_laplacian_vgg_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, bilinear_type='share', tanh=False):
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
    for level in range(levels[-1] + 1):
        if level == 0:
            xlevels_c_dim[level] = x_shape[0]
        else:
            xlevels_c_dim[level] = x1_c_dim * 2 ** (level - 1)

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
    l_xlevels_res = OrderedDict()
    l_xlevels_next_pred_res = OrderedDict()
    for level in levels[::-1]:
        l_xlevel = l_xlevels[level]
        if level == levels[-1]:
            l_xlevels_res[level] = l_xlevel
        else:
            # TODO: should use deconv because number of channels are different
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
