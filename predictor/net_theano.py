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


def build_fcn_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, num_downsample=0, bilinear_type='share', ladder_loss=True, batch_norm=False, concat=False, tanh=False):
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
    l_x0 = l_x
    x0_shape = x_shape
    if num_downsample > 0:
        import cv2
        ds_kernel = cv2.getGaussianKernel(ksize=2, sigma=-1)
        ds_weight_filler = ds_kernel.dot(ds_kernel.T).astype(theano.config.floatX)
        ds_weight_filler = ds_weight_filler.reshape((1, 1, *ds_weight_filler.shape))
    for i_ds in range(num_downsample):
        l_x0_channels = []
        for channel in range(x0_shape[0]):
            l_x0_channel = L.SliceLayer(l_x0, indices=slice(channel, channel+1), axis=1)
            l_x0_channel = L.Conv2DLayer(l_x0_channel, 1, filter_size=2, stride=2, pad=0,
                                         W=ds_weight_filler,
                                         b=init.Constant(0.),
                                         nonlinearity=None)
            LT.set_layer_param_tags(l_x0_channel, trainable=False)
            l_x0_channels.append(l_x0_channel)
        l_x0 = L.ConcatLayer(l_x0_channels, axis=1, name='x0' if i_ds == (num_downsample-1) else 'x0_ds%d'%(i_ds+1))
        x0_shape = (x0_shape[0], x0_shape[1]//2, x0_shape[2]//2)

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
            LT.set_layer_param_tags(l_xlevel_1, encoding=True, **dict([('level%d'%level, True)]))
            if batch_norm:
                l_xlevel_1 = L.BatchNormLayer(l_xlevel_1, name='x%d_bn1'%level)
                LT.set_layer_param_tags(l_xlevel_1, encoding=True, **dict([('level%d'%level, True)]))
            l_xlevel_1 = L.NonlinearityLayer(l_xlevel_1, nonlinearity=nl.rectify, name='x%d_1'%level)
            l_xlevel_2 = L.Conv2DLayer(l_xlevel_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       nonlinearity=None,
                                       name='x%d_conv2'%level)
            LT.set_layer_param_tags(l_xlevel_2, encoding=True, **dict([('level%d'%level, True)]))
            if batch_norm:
                l_xlevel_2 = L.BatchNormLayer(l_xlevel_2, name='x%d_bn2'%level)
                LT.set_layer_param_tags(l_xlevel_2, encoding=True, **dict([('level%d'%level, True)]))
            l_xlevel_2 = L.NonlinearityLayer(l_xlevel_2, nonlinearity=nl.rectify, name='x%d_2'%level)
            l_xlevel = L.MaxPool2DLayer(l_xlevel_2, pool_size=2, stride=2, pad=0, name='x%d'%level)
        l_xlevels[level] = l_xlevel
        if level in levels:
            l_ylevels[level] = L.FlattenLayer(l_xlevel, name='y%d'%level)
    l_y = L.ConcatLayer(l_ylevels.values(), name='y')

    # bilinear
    l_xlevels_next_trans = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_trans = LT.create_bilinear_layer(l_xlevel, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_trans'%level)
        l_xlevel_next_trans = L.ElemwiseMergeLayer([l_xlevel, l_xlevel_diff_trans], T.add, name='x%d_next_trans'%level)
        l_xlevels_next_trans[level] = l_xlevel_next_trans

    # decoding
    l_xlevels_next_pred = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            l_xlevel_next_pred = l_xlevels_next_trans[level]
        else:
            if level == 0:
                xlevel_c_dim = x1_c_dim
                xlevelm1_c_dim = x0_shape[0]
            elif level < levels[-1]-1:
                xlevel_c_dim = xlevelm1_c_dim
                xlevelm1_c_dim = xlevel_c_dim // 2
            l_xlevel_next_pred_2 = LT.Deconv2DLayer(l_xlevels_next_pred[level+1], xlevel_c_dim, filter_size=2, stride=2, pad=0,
                                                 nonlinearity=None,
                                                 name='x%d_next_pred_2'%(level+1)) # TODO initialize with bilinear weights # TODO should rectify? # TODO: channel-wise (groups) # TODO: no bias term
            LT.set_layer_param_tags(l_xlevel_next_pred_2, decoding=True, **dict([('level%d'%(level+1), True)]))
            l_xlevel_next_pred_1 = LT.Deconv2DLayer(l_xlevel_next_pred_2, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                                 nonlinearity=None,
                                                 name='x%d_next_pred_deconv2'%(level+1))
            LT.set_layer_param_tags(l_xlevel_next_pred_1, decoding=True, **dict([('level%d'%(level+1), True)]))
            if batch_norm:
                l_xlevel_next_pred_1 = L.BatchNormLayer(l_xlevel_next_pred_1, name='x%d_next_pred_bn2'%(level+1))
                LT.set_layer_param_tags(l_xlevel_next_pred_1, decoding=True, **dict([('level%d'%(level+1), True)]))
            l_xlevel_next_pred_1 = L.NonlinearityLayer(l_xlevel_next_pred_1, nonlinearity=nl.rectify, name='x%d_next_pred_1'%(level+1))
            if concat:
                if level in l_xlevels_next_trans:
                    l_xlevel_next_pred_1 = L.ConcatLayer([l_xlevels_next_trans[level], l_xlevel_next_pred_1], name='x%d_next_pred_concat1'%(level+1))
                l_xlevel_next_pred = LT.Deconv2DLayer(l_xlevel_next_pred_1, xlevelm1_c_dim, filter_size=3, stride=1, pad=1,
                                                   nonlinearity=None,
                                                   name='x%d_next_pred_deconv1'%(level+1))
                LT.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
                if batch_norm: # TODO batch normat level 0?
                    l_xlevel_next_pred = L.BatchNormLayer(l_xlevel_next_pred, name='x%d_next_pred_bn1'%(level+1))
                    LT.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
            else:
                l_xlevel_next_pred = LT.Deconv2DLayer(l_xlevel_next_pred_1, xlevelm1_c_dim, filter_size=3, stride=1, pad=1,
                                                      nonlinearity=None,
                                                      name='x%d_next_pred_deconv1'%(level+1))
                LT.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
                if batch_norm:
                    l_xlevel_next_pred = L.BatchNormLayer(l_xlevel_next_pred, name='x%d_next_pred_bn1'%(level+1))
                    LT.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
                if level in l_xlevels_next_trans:
                    l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevels_next_trans[level], l_xlevel_next_pred], coeffs=[0.5, 0.5], name='x%d_next_pred_sum1'%(level+1)) # TODO weights should be learned
                    LT.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
            l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred, nonlinearity=nl.tanh if (level == 0 and tanh) else nl.rectify, name='x%d_next_pred'%level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred
        if level in levels:
            l_ylevels_diff_pred[level] = L.FlattenLayer(L.ElemwiseSumLayer([l_xlevel_next_pred, l_xlevels[level]], coeffs=[1.0, -1.0], name='x%d_diff_pred'%level), name='y%d_diff_pred'%level)
    l_y_diff_pred = L.ConcatLayer(list(l_ylevels_diff_pred.values())[::-1], name='y_diff_pred')
    l_y_next_pred = L.ElemwiseSumLayer([l_y, l_y_diff_pred], name='y_next_pred')

    l_x0_next_pred = l_xlevels_next_pred[0]

    # preprocess
    l_x0_next = l_x_next
    for i_ds in range(num_downsample):
        l_x0_next_channels = []
        for channel in range(x0_shape[0]):
            l_x0_next_channel = L.SliceLayer(l_x0_next, indices=slice(channel, channel+1), axis=1)
            l_x0_next_channel = L.Conv2DLayer(l_x0_next_channel, 1, filter_size=2, stride=2, pad=0,
                                              W=ds_weight_filler,
                                              b=init.Constant(0.),
                                              nonlinearity=None)
            LT.set_layer_param_tags(l_x0_next_channel, trainable=False)
            l_x0_next_channels.append(l_x0_next_channel)
        l_x0_next = L.ConcatLayer(l_x0_next_channels, axis=1, name='x0_next' if i_ds == (num_downsample-1) else 'x0_next_ds%d'%(i_ds+1))

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = loss_fn(lasagne.layers.get_output(l_x0_next),
                   lasagne.layers.get_output(l_x0_next_pred))
    loss_deterministic = loss_fn(lasagne.layers.get_output(l_x0_next, deterministic=True),
                                 lasagne.layers.get_output(l_x0_next_pred, deterministic=True))

    if ladder_loss:
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

        for level in levels:
            if level == 0:
                continue
            loss += loss_fn(lasagne.layers.get_output(l_xlevels_next[level]),
                            lasagne.layers.get_output(l_xlevels_next_pred[level]))
            loss_deterministic += loss_fn(lasagne.layers.get_output(l_xlevels_next[level], deterministic=True),
                                          lasagne.layers.get_output(l_xlevels_next_pred[level], deterministic=True))

    net_name = 'FcnActionCondEncoderNet'
    net_name +='_levels' + ''.join([str(level) for level in levels])
    net_name += '_x1cdim' + str(x1_c_dim)
    net_name += '_numds' + str(num_downsample)
    net_name += '_bi' + bilinear_type
    net_name += '_ladder' + str(int(ladder_loss))
    net_name += '_bn' + str(int(batch_norm))
    net_name += '_concat' + str(int(concat))

    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred),
                               ('y', l_y),
                               ('y_next_pred', l_y_next_pred),
                               ('x0_next_pred', l_x0_next_pred),
                               ('x%d_next_pred'%levels[-1], l_xlevels_next_pred[levels[-1]])])
    if ladder_loss:
        pred_layers.update([('x%d_next'%levels[-1], l_xlevels_next[levels[-1]])])
    return net_name, input_vars, pred_layers, loss, loss_deterministic


def build_laplacian_fcn_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, bilinear_type='share', batch_norm=False, tanh=False):
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
            l_xlevel = LT.VggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], batch_norm=batch_norm, name='x%d' % level)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_res = OrderedDict()
    l_xlevels_dec = OrderedDict()
    l_ylevels = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        if level == levels[-1]:
            l_xlevel_res = l_xlevel
        else:
            l_xlevels_dec[level] = LT.VggDecodingLayer(l_xlevels[level+1], xlevels_c_dim[level],
                                                       batch_norm=batch_norm, name='x%d_dec' % level)
            l_xlevel_res = L.ElemwiseSumLayer([l_xlevel, l_xlevels_dec[level]], coeffs=[1.0, -1.0], name='x%d_res' % level)
        l_xlevel_diff_res = LT.create_bilinear_layer(l_xlevel_res, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_res' % level)
        l_xlevels_next_res[level] = L.ElemwiseSumLayer([l_xlevel_res, l_xlevel_diff_res], name='x%d_next_res' % level)
        l_ylevels[level] = L.FlattenLayer(l_xlevel_res, name='y%d' % level)
        l_ylevels_diff_pred[level] = L.FlattenLayer(l_xlevel_diff_res, name='y%d_diff_pred' % level)
    l_y = L.ConcatLayer(l_ylevels.values(), name='y')
    l_y_diff_pred = L.ConcatLayer(l_ylevels_diff_pred.values(), name='y_diff_pred')
    l_y_next_pred = L.ElemwiseSumLayer([l_y, l_y_diff_pred], name='y_next_pred')

    # decoding
    l_xlevels_next_pred = OrderedDict()
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            l_xlevel_next_pred = l_xlevels_next_res[level]
        else:
            if level in l_xlevels_dec:
                param_kwargs = l_xlevels_dec[level].get_param_kwargs()
            else:
                param_kwargs = dict()
            l_xlevel_next_pred = LT.VggDecodingLayer(l_xlevels_next_pred[level+1], xlevels_c_dim[level],
                                                     **param_kwargs,
                                                     batch_norm=batch_norm, name='x%d_next_dec' % level)
            if level in l_xlevels_next_res:
                l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevels_next_res[level], l_xlevel_next_pred], name='x%d_next_dec_res_sum' % level)
            if level != 0:
                l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred, nl.rectify, name='x%d_next_pred' % level)
            elif tanh:
                l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred, nl.tanh, name='x%d_next_pred' % level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred

    # encoding of next image
    l_xlevels_next = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel_next = l_x_next
        else:
            l_xlevel_next = LT.VggEncodingLayer(l_xlevels_next[level-1], xlevels_c_dim[level],
                                                **l_xlevels[level].get_param_kwargs(),
                                                batch_norm=batch_norm, name='x%d_next'%level)
        l_xlevels_next[level] = l_xlevel_next

    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred),
                               ('y', l_y),
                               ('y_next_pred', l_y_next_pred),
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               ('x0', l_xlevels[0]),
                               ('x0_next_pred', l_xlevels_next_pred[0]),
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x%d_next_pred' % levels[-1], l_xlevels_next_pred[levels[-1]]),
                               ('x%d_next' % levels[-1], l_xlevels_next[levels[-1]])])
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
