from __future__ import division

import numpy as np
import copy
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2 as pb2

def compute_jacobian(net, output, input_):
    assert output in net.outputs
    assert input_ in net.inputs
    input_data = net.blobs[input_].data
    assert input_data.ndim == 2
    assert input_data.shape[0] == 1
    output_data = net.blobs[output].data
    assert output_data.ndim == 2
    assert output_data.shape[0] == 1
    doutput_dinput = np.array([net.backward(y_diff_pred=e[None,:])[input_].flatten() for e in np.eye(output_data.shape[1])])
    return doutput_dinput

def traverse_layers_to_keep(layer, layers_dict, layers_to_keep):
    if layer in layers_to_keep: # has already been traversed
        return
    layers_to_keep.append(layer)
    for layer_name in layer.bottom:
        if layer_name in layers_dict:
            traverse_layers_to_keep(layers_dict[layer_name], layers_dict, layers_to_keep)

def remove_non_descendants(layers, ancestor_layers, exception_layers=[]):
    """
    Remove layers that are not descendants of ancestor_layers except for exception_layers.
    """
    layers_dict = {layer.name: layer for layer in layers}
    layers_to_keep = [layer for layer in exception_layers]
    for ancestor_layer in ancestor_layers:
        traverse_layers_to_keep(ancestor_layer, layers_dict, layers_to_keep)
    layer_names_to_keep = [layer.name for layer in layers_to_keep]
    for layer_name, layer in layers_dict.items():
        if layer in layers_to_keep:
            continue
        for top_layer in layer.top:
            if top_layer in layer_names_to_keep:
                layer_names_to_keep.append(layer_name)
                layers_to_keep.append(layer)
                break
    for layer_name, layer in layers_dict.items():
        if layer_name not in layer_names_to_keep:
            layers.remove(layer)

def deploy_net(net, inputs, input_shapes, outputs, batch_size=1, force_backward=True):
    # remove all layers that are not descendants of output layers
    output_layers = [layer for layer in net.layer if layer.name in outputs]
    remove_non_descendants(net.layer, output_layers)
    net.input.extend(inputs)
    net.input_shape.extend([pb2.BlobShape(dim=(batch_size,)+shape) for shape in input_shapes])
    net.force_backward = force_backward
    return net

def train_val_net(net):
    # remove all layers that are not descendants of loss layers
    loss_layers = [layer for layer in net.layer if layer.name.endswith('loss')]
    exception_layers = [layer for layer in net.layer if 'data' in layer.name]
    remove_non_descendants(net.layer, loss_layers, exception_layers)
    return net

def approx_bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='ApproxBilinearNet', phase=None):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    _, height, width = image_shape
    y_dim = height * width

    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     num_output=y_dim,
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.y = L.Flatten(n.image_curr, name='flatten1')
    n.y_diff = L.Flatten(n.image_diff, name='flatten2')
    n.fc1_y = L.InnerProduct(n.y, name='fc1', **fc_kwargs)
    n.fc2_u = L.InnerProduct(u, name='fc2', **fc_kwargs)
    n.fc3_u = L.InnerProduct(u, name='fc3', **fc_kwargs)
    n.prod_y_u = L.Eltwise(n.fc1_y, n.fc2_u, name='prod', operation=P.Eltwise.PROD)
    n.y_diff_pred = L.Eltwise(n.prod_y_u, n.fc3_u, name='sum', operation=P.Eltwise.SUM)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def Bilinear(n, y, u, y_dim, u_dim, name='bilinear', **fc_kwargs):
    re_y = n.tops[name+'_re_y'] = L.Reshape(y, shape=dict(dim=[0, 1, -1]))
    tile_re_y = n.tops[name+'_tile_re_y'] = L.Tile(re_y, axis=1, tiles=u_dim)
    re_u = n.tops[name+'_re_u'] = L.Reshape(u, shape=dict(dim=[0, -1, 1]))
    tile_re_u = n.tops[name+'_tile_re_u'] = L.Tile(re_u, axis=2, tiles=y_dim)
    outer_yu = n.tops[name+'_outer_yu'] = L.Eltwise(tile_re_y, tile_re_u, operation=P.Eltwise.PROD)
    fc_outer_yu = n.tops[name+'_fc_outer_yu'] = L.InnerProduct(outer_yu, num_output=y_dim, **fc_kwargs)
    fc_u = n.tops[name+'_fc_u'] = L.InnerProduct(u, num_output=y_dim, **fc_kwargs)
    return L.Eltwise(fc_outer_yu, fc_u, operation=P.Eltwise.SUM)

def bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearNet', phase=None, **kwargs):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    _, height, width = image_shape
    y_dim = height * width
    u_dim = vel_shape[0]

    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = Bilinear(n, n.y, u, y_dim, u_dim, **fc_kwargs)
    n.y_diff = L.Flatten(n.image_diff)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    n.image_diff_pred = L.Reshape(n.y_diff_pred, shape=dict(dim=[batch_size] + list(image_shape)))
    n.image_next_pred = L.Eltwise(n.image_curr, n.image_diff_pred, operation=P.Eltwise.SUM)

    net = n.to_proto()
    net.name = net_name
    return net

def bilinear_constrained_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearConstrainedNet', phase=None, **kwargs):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    _, height, width = image_shape
    y_dim = height * width
    u_dim = vel_shape[0]

    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = Bilinear(n, n.y, u, y_dim, u_dim, **fc_kwargs)
    n.image_diff_pred = L.Reshape(n.y_diff_pred, shape=dict(dim=[batch_size] + list(image_shape)))
    n.image_next_pred_unconstrained = L.Eltwise(n.image_curr, n.image_diff_pred, operation=P.Eltwise.SUM)
    n.image_next_pred = L.TanH(n.image_next_pred_unconstrained)
    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)
    n.loss = L.EuclideanLoss(n.image_next, n.image_next_pred, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='ActionCondEncoderNet', phase=None):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    y_dim = 1024
    u_dim = vel_shape[0]

    conv_kwargs = dict(num_output=64, kernel_size=6, stride=2)
    deconv_kwargs = conv_kwargs
    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)

    n.conv1 = L.Convolution(n.image_curr, name='conv1', **conv_kwargs)
    n.relu1 = L.ReLU(n.conv1, name='relu1', in_place=True)
    n.conv2 = L.Convolution(n.relu1, name='conv2', pad=2, **conv_kwargs)
    n.relu2 = L.ReLU(n.conv2, name='relu2', in_place=True)
    n.conv3 = L.Convolution(n.relu2, name='conv3', pad=2, **conv_kwargs)
    n.relu3 = L.ReLU(n.conv3, name='relu3', in_place=True)
    n.y = L.InnerProduct(n.relu3, num_output=y_dim, weight_filler=dict(type='xavier'))

    u = n.vel
    n.y_diff_pred = Bilinear(n, n.y, u, y_dim, u_dim, **fc_kwargs)
    n.y_next_pred = L.Eltwise(n.y, n.y_diff_pred, operation=P.Eltwise.SUM)

    n.ip2 = L.InnerProduct(n.y_next_pred, name='ip2', num_output=6400, weight_filler=dict(type='xavier'))
    n.re_y_next_pred = L.Reshape(n.ip2, shape=dict(dim=[batch_size, 64, 10, 10]))
    n.deconv3 = L.Deconvolution(n.re_y_next_pred, convolution_param=dict(deconv_kwargs.items() + dict(pad=2).items()))
    n.derelu3 = L.ReLU(n.deconv3, in_place=True)
    n.deconv2 = L.Deconvolution(n.derelu3, convolution_param=dict(deconv_kwargs.items() + dict(pad=2).items()))
    n.derelu2 = L.ReLU(n.deconv2, in_place=True)
    n.deconv1 = L.Deconvolution(n.derelu2, convolution_param=dict(deconv_kwargs.items() + dict(num_output=1).items()))
    n.image_next_pred = L.ReLU(n.deconv1, in_place=True)

    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.loss = L.EuclideanLoss(n.image_next_pred, n.image_next, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def small_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, constrained=True, **kwargs):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert image0_shape[1] == 32
    assert image0_shape[2] == 32
    assert len(vel_shape) == 1
    image0_num_channel = image0_shape[0]
    num_channel = kwargs.get('num_channel') or 16
    image1_num_channel = num_channel
    image2_num_channel = num_channel
    image1_shape = (image1_num_channel, 16, 16)
    image2_shape = (image2_num_channel, 8, 8)
    y0_dim = image0_shape[1] * image0_shape[2] # 1024
    y2_dim = kwargs.get('y2_dim') or 64
    u_dim = vel_shape[0]

    conv1_kwargs = dict(param=[dict(name='conv1', lr_mult=1, decay_mult=1), dict(name='conv1_bias', lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image1_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    conv2_kwargs = dict(param=[dict(name='conv2', lr_mult=1, decay_mult=1), dict(name='conv2_bias', lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image2_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    deconv0_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                          convolution_param=dict(num_output=image0_num_channel,
                                                 kernel_size=6,
                                                 stride=2,
                                                 pad=2,
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)))
    deconv1_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                          convolution_param=dict(num_output=image1_num_channel,
                                                 kernel_size=6,
                                                 stride=2,
                                                 pad=2,
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)))
    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    image0 = n.image_curr
    u = n.vel

    n.image1 = L.Convolution(image0, **conv1_kwargs)
    n.image1 = L.ReLU(n.image1, in_place=True)
    n.image2 = L.Convolution(n.image1, **conv2_kwargs)
    n.image2 = L.ReLU(n.image2, in_place=True)

    y2 = n.y = L.InnerProduct(n.image2, num_output=y2_dim, weight_filler=dict(type='xavier'))
    y2_diff_pred = n.y_diff_pred = Bilinear(n, y2, u, y2_dim, u_dim, name='bilinear2', **fc_kwargs)
    n.y2_next_pred = L.Eltwise(y2, y2_diff_pred, operation=P.Eltwise.SUM)
    n.image2_next_pred_flat = L.InnerProduct(n.y2_next_pred, num_output=np.prod(image2_shape), weight_filler=dict(type='xavier'))
    n.image2_next_pred = L.Reshape(n.image2_next_pred_flat, shape=dict(dim=[batch_size]+list(image2_shape)))

    n.image1_next_pred = L.Deconvolution(n.image2_next_pred, **deconv1_kwargs)
    n.image1_next_pred = L.ReLU(n.image1_next_pred, in_place=True)
    if constrained:
        n.image_next_pred_unconstrained = L.Deconvolution(n.image1_next_pred, **deconv0_kwargs)
        image0_next_pred = n.image_next_pred = L.TanH(n.image_next_pred_unconstrained)
    else:
        n.image_next_pred = L.Deconvolution(n.image1_next_pred, **deconv0_kwargs)

    image0_next = n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.image0_next_loss = L.EuclideanLoss(image0_next, image0_next_pred)

    net = n.to_proto()
    if net_name is None:
        net_name = 'SmallActionCondEncoderNet'
        if constrained:
            net_name += '_constrained'
        net_name +='_num_channel' + str(num_channel)
        net_name += '_y2_dim' + str(y2_dim)
    net.name = net_name
    return net

def downsampled_small_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='DownsampledSmallActionCondEncoderNet', phase=None):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    y_dim = 32
    u_dim = vel_shape[0]
    conv_num_output = 16
    conv2_wh = 4

    blur_conv_kwargs = dict(param=[dict(lr_mult=0, decay_mult=0)],
                            convolution_param=dict(num_output=1,
                                                   kernel_size=5,
                                                   stride=2,
                                                   pad=2,
                                                   bias_term=False))
    conv_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                       convolution_param=dict(num_output=conv_num_output,
                                              kernel_size=6,
                                              stride=2,
                                              pad=2,
                                              weight_filler=dict(type='gaussian', std=0.01),
                                              bias_filler=dict(type='constant', value=0)))
    deconv_kwargs = conv_kwargs
    deconv_kwargs1 = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=1,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)

    n.image_curr_ds = L.Convolution(n.image_curr, name='blur_conv1', **blur_conv_kwargs)
    n.image_diff_ds = L.Convolution(n.image_diff, name='blur_conv2', **blur_conv_kwargs)

    n.conv1 = L.Convolution(n.image_curr_ds, **conv_kwargs)
    n.relu1 = L.ReLU(n.conv1, name='relu1', in_place=True)
    n.conv2 = L.Convolution(n.relu1, **conv_kwargs)
    n.relu2 = L.ReLU(n.conv2, name='relu2', in_place=True)
    n.y = L.InnerProduct(n.relu2, num_output=y_dim, weight_filler=dict(type='xavier'))

    u = n.vel
    n.y_diff_pred = Bilinear(n, n.y, u, y_dim, u_dim, **fc_kwargs)
    n.y_next_pred = L.Eltwise(n.y, n.y_diff_pred, operation=P.Eltwise.SUM)

    n.ip2 = L.InnerProduct(n.y_next_pred, name='ip2', num_output=conv_num_output*conv2_wh**2, weight_filler=dict(type='xavier'))
    n.re_y_next_pred = L.Reshape(n.ip2, shape=dict(dim=[batch_size, conv_num_output, conv2_wh, conv2_wh]))
    n.deconv2 = L.Deconvolution(n.re_y_next_pred, **deconv_kwargs)
    n.derelu2 = L.ReLU(n.deconv2, in_place=True)
    n.deconv1 = L.Deconvolution(n.derelu2, **deconv_kwargs1)
    n.image_next_pred = L.ReLU(n.deconv1, in_place=True)

    n.image_next = L.Eltwise(n.image_curr_ds, n.image_diff_ds, operation=P.Eltwise.SUM)

    n.loss = L.EuclideanLoss(n.image_next_pred, n.image_next, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def ladder_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, constrained=True, **kwargs):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert image0_shape[1] == 32
    assert image0_shape[2] == 32
    assert len(vel_shape) == 1
    image0_num_channel = image0_shape[0]
    num_channel = kwargs.get('num_channel') or 16
    image1_num_channel = num_channel
    image2_num_channel = num_channel
    image1_shape = (image1_num_channel, 16, 16)
    image2_shape = (image2_num_channel, 8, 8)
    y0_dim = image0_shape[1] * image0_shape[2] # 1024
    y1_dim = kwargs.get('y1_dim') or 128
    y2_dim = kwargs.get('y2_dim') or 32
    u_dim = vel_shape[0]

    conv1_kwargs = dict(param=[dict(name='conv1', lr_mult=1, decay_mult=1), dict(name='conv1_bias', lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image1_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    conv2_kwargs = dict(param=[dict(name='conv2', lr_mult=1, decay_mult=1), dict(name='conv2_bias', lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image2_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    deconv0_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                          convolution_param=dict(num_output=image0_num_channel,
                                                 kernel_size=6,
                                                 stride=2,
                                                 pad=2,
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)))
    deconv1_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                          convolution_param=dict(num_output=image1_num_channel,
                                                 kernel_size=6,
                                                 stride=2,
                                                 pad=2,
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)))
    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    image0 = n.image_curr
    u = n.vel

    n.image1 = L.Convolution(image0, **conv1_kwargs)
    n.image1 = L.ReLU(n.image1, in_place=True)
    n.image2 = L.Convolution(n.image1, **conv2_kwargs)
    n.image2 = L.ReLU(n.image2, in_place=True)

    n.y2 = L.InnerProduct(n.image2, num_output=y2_dim, weight_filler=dict(type='xavier'))
    n.y1 = L.InnerProduct(n.image1, num_output=y1_dim, weight_filler=dict(type='xavier'))
    n.y0 = L.Flatten(image0)
    n.y01 = L.Concat(n.y0, n.y1, axis=1)
    n.y = L.Concat(n.y01, n.y2, axis=1)

    n.y2_diff_pred = Bilinear(n, n.y2, u, y2_dim, u_dim, name='bilinear2', **fc_kwargs)
    n.y1_diff_pred = Bilinear(n, n.y1, u, y1_dim, u_dim, name='bilinear1', **fc_kwargs)
    n.y0_diff_pred = Bilinear(n, n.y0, u, y0_dim, u_dim, name='bilinear0', **fc_kwargs)
    n.y01_diff_pred = L.Concat(n.y0_diff_pred, n.y1_diff_pred, axis=1)
    n.y_diff_pred = L.Concat(n.y01_diff_pred, n.y2_diff_pred, axis=1)

    n.y2_next_pred = L.Eltwise(n.y2, n.y2_diff_pred, operation=P.Eltwise.SUM)
    n.image2_next_pred_flat = L.InnerProduct(n.y2_next_pred, num_output=np.prod(image2_shape), weight_filler=dict(type='xavier'))
    n.image2_next_pred = L.Reshape(n.image2_next_pred_flat, shape=dict(dim=[batch_size]+list(image2_shape)))

    n.y1_next_pred = L.Eltwise(n.y1, n.y1_diff_pred, operation=P.Eltwise.SUM)
    n.image1_next_pred1_flat = L.InnerProduct(n.y1_next_pred, num_output=np.prod(image1_shape), weight_filler=dict(type='xavier'))
    n.image1_next_pred1 = L.Reshape(n.image1_next_pred1_flat, shape=dict(dim=[batch_size]+list(image1_shape)))
    n.image1_next_pred2 = L.Deconvolution(n.image2_next_pred, **deconv1_kwargs)
    n.image1_next_pred = L.Eltwise(n.image1_next_pred1, n.image1_next_pred2, operation=P.Eltwise.SUM)
    n.image1_next_pred = L.ReLU(n.image1_next_pred, in_place=True)

    n.y0_next_pred = L.Eltwise(n.y0, n.y0_diff_pred, operation=P.Eltwise.SUM)
    n.image0_next_pred0 = L.Reshape(n.y0_next_pred, shape=dict(dim=[batch_size]+list(image0_shape)))
    n.image0_next_pred1 = L.Deconvolution(n.image1_next_pred, **deconv0_kwargs)
    if constrained:
        n.image_next_pred_unconstrained = L.Eltwise(n.image0_next_pred0, n.image0_next_pred1, operation=P.Eltwise.SUM)
        image0_next_pred = n.image_next_pred = L.TanH(n.image_next_pred_unconstrained)
    else:
        image0_next_pred = n.image_next_pred = L.Eltwise(n.image0_next_pred0, n.image0_next_pred1, operation=P.Eltwise.SUM)

    image0_next = n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.image1_next = L.Convolution(image0_next, **conv1_kwargs)
    n.image1_next = L.ReLU(n.image1_next, in_place=True)
    n.image2_next = L.Convolution(n.image1_next, **conv2_kwargs)
    n.image2_next = L.ReLU(n.image2_next, in_place=True)

    n.image0_next_loss = L.EuclideanLoss(image0_next, image0_next_pred)
    n.image1_next_loss = L.EuclideanLoss(n.image1_next, n.image1_next_pred)
    n.image2_next_loss = L.EuclideanLoss(n.image2_next, n.image2_next_pred)

    net = n.to_proto()
    if net_name is None:
        net_name = 'LadderActionCondEncoderNet'
        if constrained:
            net_name += '_constrained'
        net_name +='_num_channel' + str(num_channel)
        net_name += '_y1_dim' + str(y1_dim)
        net_name += '_y2_dim' + str(y2_dim)
    net.name = net_name
    return net

def ladder_conv_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, constrained=True, **kwargs):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert image0_shape[1] == 32
    assert image0_shape[2] == 32
    assert len(vel_shape) == 1
    image0_num_channel = image0_shape[0]
    num_channel = kwargs.get('num_channel') or 16
    image1_num_channel = num_channel
    image2_num_channel = num_channel
    image1_shape = (image1_num_channel, 16, 16)
    image2_shape = (image2_num_channel, 8, 8)
    y0_dim = image0_shape[1] * image0_shape[2] # 1024
    y1_dim = kwargs.get('y1_dim') or 128
    y2_dim = kwargs.get('y2_dim') or 32
    u_dim = vel_shape[0]

    conv1_kwargs = dict(param=[dict(name='conv1', lr_mult=1, decay_mult=1), dict(name='conv1_bias', lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image1_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    conv2_kwargs = dict(param=[dict(name='conv2', lr_mult=1, decay_mult=1), dict(name='conv2_bias', lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image2_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    deconv0_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                          convolution_param=dict(num_output=image0_num_channel,
                                                 kernel_size=6,
                                                 stride=2,
                                                 pad=2,
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)))
    deconv1_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                          convolution_param=dict(num_output=image1_num_channel,
                                                 kernel_size=6,
                                                 stride=2,
                                                 pad=2,
                                                 weight_filler=dict(type='gaussian', std=0.01),
                                                 bias_filler=dict(type='constant', value=0)))
    conv0_merge_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                              convolution_param=dict(num_output=image0_num_channel,
                                                     kernel_size=1,
                                                     stride=1,
                                                     pad=0,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant', value=0)))
    conv1_merge_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                              convolution_param=dict(num_output=image1_num_channel,
                                                     kernel_size=1,
                                                     stride=1,
                                                     pad=0,
                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                     bias_filler=dict(type='constant', value=0)))
    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    image0 = n.image_curr
    u = n.vel

    n.image1 = L.Convolution(image0, **conv1_kwargs)
    n.image1 = L.ReLU(n.image1, in_place=True)
    n.image2 = L.Convolution(n.image1, **conv2_kwargs)
    n.image2 = L.ReLU(n.image2, in_place=True)

    n.y2 = L.InnerProduct(n.image2, num_output=y2_dim, weight_filler=dict(type='xavier'))
    n.y1 = L.InnerProduct(n.image1, num_output=y1_dim, weight_filler=dict(type='xavier'))
    n.y0 = L.Flatten(image0)
    n.y01 = L.Concat(n.y0, n.y1, axis=1)
    n.y = L.Concat(n.y01, n.y2, axis=1)

    n.y2_diff_pred = Bilinear(n, n.y2, u, y2_dim, u_dim, name='bilinear2', **fc_kwargs)
    n.y1_diff_pred = Bilinear(n, n.y1, u, y1_dim, u_dim, name='bilinear1', **fc_kwargs)
    n.y0_diff_pred = Bilinear(n, n.y0, u, y0_dim, u_dim, name='bilinear0', **fc_kwargs)
    n.y01_diff_pred = L.Concat(n.y0_diff_pred, n.y1_diff_pred, axis=1)
    n.y_diff_pred = L.Concat(n.y01_diff_pred, n.y2_diff_pred, axis=1)

    n.y2_next_pred = L.Eltwise(n.y2, n.y2_diff_pred, operation=P.Eltwise.SUM)
    n.image2_next_pred_flat = L.InnerProduct(n.y2_next_pred, num_output=np.prod(image2_shape), weight_filler=dict(type='xavier'))
    n.image2_next_pred = L.Reshape(n.image2_next_pred_flat, shape=dict(dim=[batch_size]+list(image2_shape)))

    n.y1_next_pred = L.Eltwise(n.y1, n.y1_diff_pred, operation=P.Eltwise.SUM)
    n.image1_next_pred1_flat = L.InnerProduct(n.y1_next_pred, num_output=np.prod(image1_shape), weight_filler=dict(type='xavier'))
    n.image1_next_pred1 = L.Reshape(n.image1_next_pred1_flat, shape=dict(dim=[batch_size]+list(image1_shape)))
    n.image1_next_pred2 = L.Deconvolution(n.image2_next_pred, **deconv1_kwargs)
    n.image1_next_pred12 = L.Concat(n.image1_next_pred1, n.image1_next_pred2, concat_param=dict(axis=1))
    n.image1_next_pred = L.Convolution(n.image1_next_pred12, **conv1_merge_kwargs)
    n.image1_next_pred = L.ReLU(n.image1_next_pred, in_place=True)

    n.y0_next_pred = L.Eltwise(n.y0, n.y0_diff_pred, operation=P.Eltwise.SUM)
    n.image0_next_pred0 = L.Reshape(n.y0_next_pred, shape=dict(dim=[batch_size]+list(image0_shape)))
    n.image0_next_pred1 = L.Deconvolution(n.image1_next_pred, **deconv0_kwargs)
    n.image0_next_pred01 = L.Concat(n.image0_next_pred0, n.image0_next_pred1, concat_param=dict(axis=1))

    if constrained:
        n.image_next_pred_unconstrained = L.Convolution(n.image0_next_pred01, **conv0_merge_kwargs)
        image0_next_pred = n.image_next_pred = L.TanH(n.image_next_pred_unconstrained)
    else:
        image0_next_pred = n.image_next_pred = L.Convolution(n.image0_next_pred01, **conv0_merge_kwargs)

    image0_next = n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.image1_next = L.Convolution(image0_next, **conv1_kwargs)
    n.image1_next = L.ReLU(n.image1_next, in_place=True)
    n.image2_next = L.Convolution(n.image1_next, **conv2_kwargs)
    n.image2_next = L.ReLU(n.image2_next, in_place=True)

    n.image0_next_loss = L.EuclideanLoss(image0_next, image0_next_pred)
    n.image1_next_loss = L.EuclideanLoss(n.image1_next, n.image1_next_pred)
    n.image2_next_loss = L.EuclideanLoss(n.image2_next, n.image2_next_pred)

    net = n.to_proto()
    if net_name is None:
        net_name = 'LadderConvActionCondEncoderNet'
        if constrained:
            net_name += '_constrained'
        net_name +='_num_channel' + str(num_channel)
        net_name += '_y1_dim' + str(y1_dim)
        net_name += '_y2_dim' + str(y2_dim)
    net.name = net_name
    return net

def ConvolutionPooling(n, image, conv_kwargs, pool_kwargs, name=''):
    conv_1_kwargs = copy.deepcopy(conv_kwargs)
    for param in conv_1_kwargs['param']:
        if 'name' in param:
            param['name'] += '_1'
    conv_2_kwargs = copy.deepcopy(conv_kwargs)
    for param in conv_2_kwargs['param']:
        if 'name' in param:
            param['name'] += '_2'
    conv_1 = n.tops['conv'+name+'_1'] = L.Convolution(image, **conv_1_kwargs)
    n.tops['relu'+name+'_1'] = L.ReLU(conv_1, in_place=True)
    conv_2 = n.tops['conv'+name+'_2'] = L.Convolution(conv_1, **conv_2_kwargs)
    n.tops['relu'+name+'_2'] = L.ReLU(conv_2, in_place=True)
    return L.Pooling(conv_2, name='pool'+name, **pool_kwargs)

def DeconvolutionUpsample(n, image, deconv_kwargs, upsample_kwargs, name=''):
    for param in deconv_kwargs['param']:
        assert 'name' not in param
    deconv_1 = n.tops['deconv'+name+'_1'] = L.Deconvolution(image, **deconv_kwargs)
    n.tops['derelu'+name+'_1'] = L.ReLU(deconv_1, in_place=True)
    deconv_2 = n.tops['deconv'+name+'_2'] = L.Deconvolution(deconv_1, **deconv_kwargs)
    n.tops['derelu'+name+'_2'] = L.ReLU(deconv_2, in_place=True)
    return L.Deconvolution(deconv_2, **upsample_kwargs)

def conv_kwargs_(num_output, kernel_size, stride, pad,
                 lr_mult=1, lr_mult_bias=1,
                 name=None, name_bias=None,
                 weight_filler_type='gaussian'):
    param = [dict(lr_mult=lr_mult, decay_mult=1), dict(lr_mult=lr_mult_bias, decay_mult=1)]
    if name is not None:
        param[0]['name'] = name
    if name_bias is not None:
        param[1]['name'] = name_bias
    if weight_filler_type == 'gaussian':
        weight_filler = dict(type='gaussian', std=0.01)
    elif weight_filler_type == 'constant':
        weight_filler = dict(type='constant', value=0)
    elif weight_filler_type == 'bilinear':
        weight_filler = dict(type='bilinear')
    else:
        raise
    convolution_param = dict(num_output=num_output,
                             kernel_size=kernel_size,
                             stride=stride,
                             pad=pad,
                             weight_filler=weight_filler,
                             bias_filler=dict(type='constant', value=0))
    if weight_filler_type == 'bilinear':
        convolution_param['group'] = num_output
        convolution_param['bias_term'] = False
        param.pop()
    conv_kwargs = dict(param=param, convolution_param=convolution_param)
    return conv_kwargs

def ImageBilinear(n, image, u, image_shape, u_dim, fc_kwargs, name='', init=False):
    y_c = 1
    y_shape = [y_c] + list(image_shape[1:])
    y_dim = np.prod(y_shape)
    y = n.tops['y'+name] = L.Convolution(image, **conv_kwargs_(y_c, 1, 1, 0))
    y_flat = n.tops['y'+name+'_flat'] = L.Reshape(y, shape=dict(dim=[0, -1]))
    y_diff_pred_flat = n.tops['y'+name+'_diff_pred_flat'] = Bilinear(n, y_flat, u, y_dim, u_dim, name='bilinear'+name, **fc_kwargs)
    y_diff_pred = n.tops['y'+name+'_diff_pred'] = L.Reshape(y_diff_pred_flat, shape=dict(dim=[0]+list(y_shape)))
    y_next_pred = n.tops['y'+name+'_next_pred'] = L.Eltwise(y, y_diff_pred, operation=P.Eltwise.SUM)
    if init:
        return L.Convolution(y_next_pred, **conv_kwargs_(image_shape[0], 1, 1, 0, weight_filler_type='constant'))
    else:
        return L.Convolution(y_next_pred, **conv_kwargs_(image_shape[0], 1, 1, 0))

def fcn_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, finest_level=3, **kwargs):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert len(vel_shape) == 1
    image1_shape = (64,  image0_shape[1]//2, image0_shape[2]//2)
    image2_shape = (128, image1_shape[1]//2, image1_shape[2]//2)
    image3_shape = (256, image2_shape[1]//2, image2_shape[2]//2)
    u_dim = vel_shape[0]

    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    image0_c = image0_shape[0]
    image1_c = image1_shape[0]
    image2_c = image2_shape[0]
    image3_c = image3_shape[0]
    conv1_kwargs = conv_kwargs_(image1_c, 3, 1, 1, name='conv1', name_bias='conv1_bias')
    conv2_kwargs = conv_kwargs_(image2_c, 3, 1, 1, name='conv2', name_bias='conv2_bias')
    conv3_kwargs = conv_kwargs_(image3_c, 3, 1, 1, name='conv3', name_bias='conv3_bias')
    pool_kwargs = dict(pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
    deconv3_kwargs = conv_kwargs_(image2_c, 3, 1, 1)
    deconv2_kwargs = conv_kwargs_(image1_c, 3, 1, 1)
    deconv1_kwargs = conv_kwargs_(image0_c, 3, 1, 1)
    upsample3_kwargs = conv_kwargs_(image2_c, 2, 2, 0)
    upsample2_kwargs = conv_kwargs_(image1_c, 2, 2, 0)
    upsample1_kwargs = conv_kwargs_(image0_c, 2, 2, 0)

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname, shuffle=True)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    image0 = n.image_curr
    u = n.vel

    image1 = n.pool1 = ConvolutionPooling(n, image0, conv1_kwargs, pool_kwargs, name='1')
    image2 = n.pool2 = ConvolutionPooling(n, image1, conv2_kwargs, pool_kwargs, name='2')
    image3 = n.pool3 = ConvolutionPooling(n, image2, conv3_kwargs, pool_kwargs, name='3')

    if finest_level < 0 or finest_level > 3:
        raise
    y_flat_list = []
    y_diff_pred_flat_list = []
    # image3_next_pred
    n.image3_next_pred = ImageBilinear(n, image3, u, image3_shape, u_dim, fc_kwargs, name='3')
    y_flat_list.append(n.y3_flat)
    y_diff_pred_flat_list.append(n.y3_diff_pred_flat)
    # image2_next_pred
    n.image2_next_pred_3 = DeconvolutionUpsample(n, n.image3_next_pred, deconv3_kwargs, upsample3_kwargs, name='3')
    if finest_level < 3:
        n.image2_next_pred_2 = ImageBilinear(n, image2, u, image2_shape, u_dim, fc_kwargs, name='2', init=True)
        image2_next_pred = n.image2_next_pred = n.image2_next_pred = L.Eltwise(n.image2_next_pred_3, n.image2_next_pred_2, operation=P.Eltwise.SUM)
        y_flat_list.append(n.y2_flat)
        y_diff_pred_flat_list.append(n.y2_diff_pred_flat)
    else:
        image2_next_pred = n.image2_next_pred_3
    # image1_next_pred
    n.image1_next_pred_2 = DeconvolutionUpsample(n, image2_next_pred, deconv2_kwargs, upsample2_kwargs, name='2')
    if finest_level < 2:
        n.image1_next_pred_1 = ImageBilinear(n, image1, u, image1_shape, u_dim, fc_kwargs, name='1', init=True)
        image1_next_pred = n.image1_next_pred = L.Eltwise(n.image1_next_pred_2, n.image1_next_pred_1, operation=P.Eltwise.SUM)
        y_flat_list.append(n.y1_flat)
        y_diff_pred_flat_list.append(n.y1_diff_pred_flat)
    else:
        image1_next_pred = n.image1_next_pred_2
    # image0_next_pred
    n.image0_next_pred_1 = DeconvolutionUpsample(n, image1_next_pred, deconv1_kwargs, upsample1_kwargs, name='1')
    if finest_level < 1:
        n.image0_next_pred_0 = ImageBilinear(n, image0, u, image0_shape, u_dim, fc_kwargs, name='0', init=True)
        image0_next_pred = n.image0_next_pred = L.Eltwise(n.image0_next_pred_1, n.image0_next_pred_0, operation=P.Eltwise.SUM)
        y_flat_list.append(n.y0_flat)
        y_diff_pred_flat_list.append(n.y0_diff_pred_flat)
    else:
        image0_next_pred = n.image0_next_pred_1
    n.y = L.Concat(*y_flat_list, concat_param=dict(axis=1))
    n.y_diff_pred = L.Concat(*y_diff_pred_flat_list, concat_param=dict(axis=1))

    n.image_next_pred = L.TanH(image0_next_pred)

    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)
#     n.image1_next = ConvolutionPooling(n, image0_next, conv1_kwargs, pool_kwargs, name='conv1_next')
#     n.image2_next = ConvolutionPooling(n, n.image1_next, conv2_kwargs, pool_kwargs, name='conv2_next')
#     n.image3_next = ConvolutionPooling(n, n.image2_next, conv3_kwargs, pool_kwargs, name='conv3_next')

    n.image0_next_loss = L.EuclideanLoss(n.image_next, n.image_next_pred)
#     n.image3_next_loss = L.EuclideanLoss(n.image3_next, n.image3_next_pred)

    net = n.to_proto()
    if net_name is None:
        net_name = 'FcnActionCondEncoderNet'
        net_name +='_finest_level' + str(finest_level)
    net.name = net_name
    return net
