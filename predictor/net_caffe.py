from __future__ import division

import numpy as np
import copy
from collections import OrderedDict
import cv2
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
    return net, None

def Bilinear(n, y, u, y_dim, u_dim, name='bilinear', **fc_kwargs):
    re_y = n.tops[name+'_re_y'] = L.Reshape(y, shape=dict(dim=[0, -1, 1]))
    tile_re_y = n.tops[name+'_tile_re_y'] = L.Tile(re_y, axis=2, tiles=u_dim)
    re_u = n.tops[name+'_re_u'] = L.Reshape(u, shape=dict(dim=[0, 1, -1]))
    tile_re_u = n.tops[name+'_tile_re_u'] = L.Tile(re_u, axis=1, tiles=y_dim)
    outer_yu = n.tops[name+'_outer_yu'] = L.Eltwise(tile_re_y, tile_re_u, operation=P.Eltwise.PROD)
    fc_outer_yu = n.tops[name+'_fc_outer_yu'] = L.InnerProduct(outer_yu, num_output=y_dim, **fc_kwargs)
    fc_u = n.tops[name+'_fc_u'] = L.InnerProduct(u, num_output=y_dim, **fc_kwargs)
    return L.Eltwise(fc_outer_yu, fc_u, operation=P.Eltwise.SUM)

def bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearNet', phase=None, **kwargs):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    y_dim = np.prod(image_shape)
    u_dim, = vel_shape

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
    return net, None

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
    return net, None

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
    return net, None

def small_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, constrained=True, **kwargs):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert len(vel_shape) == 1
    image0_num_channel = image0_shape[0]
    num_channel = kwargs.get('num_channel') or 16
    image1_num_channel = num_channel
    image2_num_channel = num_channel
    image1_shape = (image1_num_channel, image0_shape[1]//2, image0_shape[2]//2)
    image2_shape = (image2_num_channel, image1_shape[1]//2, image1_shape[2]//2)
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
    n.image2_next_pred = L.Reshape(n.image2_next_pred_flat, shape=dict(dim=[0]+list(image2_shape)))

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
    return net, None

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

    weight_fillers = OrderedDict()
    ds_kernel = cv2.getGaussianKernel(ksize=5, sigma=-1)
    ds_weight_filler = ds_kernel.dot(ds_kernel.T)

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)

    n.image_curr_ds = L.Convolution(n.image_curr, name='blur_conv1', **blur_conv_kwargs)
    n.image_diff_ds = L.Convolution(n.image_diff, name='blur_conv2', **blur_conv_kwargs)
    weight_fillers['image_curr_ds'] = [ds_weight_filler]
    weight_fillers['image_diff_ds'] = [ds_weight_filler]

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
    return net, weight_fillers

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
    return net, None

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
    return net, None

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
    param = [dict(lr_mult=lr_mult, decay_mult=lr_mult), dict(lr_mult=lr_mult_bias, decay_mult=lr_mult_bias)]
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

def ImageBilinear(n, image, u, image_shape, u_dim, bilinear_kwargs,
                  y_channel_dim=1, axis=1, name=''):
    assert len(bilinear_kwargs['param']) == 3
    assert axis == 1 or axis == 2
    fc_outer_yu_kwargs = dict(param=[bilinear_kwargs['param'][0], dict(lr_mult=0, decay_mult=0)],
                              weight_filler=bilinear_kwargs['bilinear_filler'],
                              bias_filler=dict(type='constant', value=0))
    fc_u_kwargs = dict(param=[bilinear_kwargs['param'][1], bilinear_kwargs['param'][2]],
                       weight_filler=bilinear_kwargs['linear_filler'],
                       bias_filler=bilinear_kwargs['bias_filler'])
    y_channel_dim = y_channel_dim or image_shape[0]
    # image -> y
    if y_channel_dim != image_shape[0]:
        y_conv1_kwargs = conv_kwargs_(y_channel_dim, 1, 1, 0)
        y = n.tops['y'+name] = L.Convolution(image, **y_conv1_kwargs)
        y_shape = (y_channel_dim,) + image_shape[1:]
    else:
        y = image
        y_shape = image_shape
    n.tops['y'+name+'_flat'] = L.Flatten(y)
    y_dim = np.prod(y_shape)
    # y, u -> outer_yu
    re_y_shape = (0,)*axis + (1, -1) # e.g. (N, C, 1, I)
    re_y = n.tops['bilinear'+name+'_re_y'] = L.Reshape(y, shape=dict(dim=list(re_y_shape)))
    tile_re_y = n.tops['bilinear'+name+'_tile_re_y'] = L.Tile(re_y, axis=axis, tiles=u_dim)
    re_u_shape = (0,) + (1,)*(axis-1) + (-1, 1) # e.g. (N, 1, J, 1)
    re_u = n.tops['bilinear'+name+'_re_u'] = L.Reshape(u, shape=dict(dim=list(re_u_shape)))
    if axis != 1:
        tile_re_u1 = n.tops['bilinear'+name+'_tile_re_u1'] = L.Tile(re_u, axis=axis+1, tiles=np.prod(y_shape[axis-1:]))
        tile_re_u = n.tops['bilinear'+name+'_tile_re_u'] = L.Tile(tile_re_u1, axis=1, tiles=np.prod(y_shape[0]))
    else:
        tile_re_u = n.tops['bilinear'+name+'_tile_re_u'] = L.Tile(re_u, axis=axis+1, tiles=np.prod(y_shape[axis-1:]))
    outer_yu = n.tops['bilinear'+name+'_outer_yu'] = L.Eltwise(tile_re_y, tile_re_u, operation=P.Eltwise.PROD) # e.g. (N, C, J, I)
    # outer_yu, u -> y_next_pred
    # bilinear term
    bilinear_yu = n.tops['bilinear'+name+'_bilinear_yu'] = L.InnerProduct(outer_yu, num_output=np.prod(y_shape[axis-1:]), axis=axis, **fc_outer_yu_kwargs) # e.g. (N, C, I)
    # linear and bias terms
    fc_u = n.tops['bilinear'+name+'_linear_u'] = L.InnerProduct(u, num_output=np.prod(y_shape[axis-1:]), **fc_u_kwargs) # e.g. (N, I)
    if axis != 1:
        re_fc_u_shape = (0,) + (1,)*(axis-1) + (np.prod(y_shape[axis-1:]),) # e.g. (N, 1, I)
        re_fc_u = n.tops['bilinear'+name+'_re_fc_u'] = L.Reshape(fc_u, shape=dict(dim=list(re_fc_u_shape)))
        linear_u = n.tops['bilinear'+name+'_tile_re_fc_u'] = L.Tile(re_fc_u, axis=1, tiles=y_channel_dim) # e.g. (N, C, I)
    else:
        linear_u = fc_u
    bilinear_yu_linear_u = n.tops['bilinear'+name+'_bilinear_yu_linear_u'] = L.Eltwise(bilinear_yu, linear_u, operation=P.Eltwise.SUM) # e.g. (N, C, I)
    n.tops['y'+name+'_diff_pred_flat'] = L.Flatten(bilinear_yu_linear_u)
    y_diff_pred = n.tops['y'+name+'_diff_pred'] = L.Reshape(bilinear_yu_linear_u, shape=dict(dim=list((0,) + y_shape)))
    y_next_pred = n.tops['y'+name+'_next_pred'] = L.Eltwise(y, y_diff_pred, operation=P.Eltwise.SUM)
    if y_channel_dim != image_shape[0]:
        y_diff_conv2_kwargs = conv_kwargs_(image_shape[0], 1, 1, 0)
        image_next_pred = L.Convolution(y_next_pred, **y_diff_conv2_kwargs)
    else:
        image_next_pred = L.Concat(y_next_pred, concat_param=dict(axis=1)) # proxy for identity layer
    return image_next_pred

def ImageBilinearChannelwise(n, x, u, x_shape, u_dim, bilinear_kwargs, axis=1, name='', share_weights=True):
    assert len(bilinear_kwargs['param']) == 4
    assert axis == 1 or axis == 2
    fc_outer_yu_kwargs = dict(param=[bilinear_kwargs['param'][0]],
                              bias_term=False,
                              weight_filler=bilinear_kwargs['bilinear_filler'])
    fc_y_kwargs = dict(param=bilinear_kwargs['param'][1],
                       bias_term=False,
                       weight_filler=bilinear_kwargs['linear_y_filler'])
    fc_u_kwargs = dict(param=bilinear_kwargs['param'][2:],
                       weight_filler=bilinear_kwargs['linear_u_filler'],
                       bias_filler=bilinear_kwargs['bias_filler'])
    # y, u -> outer_yu
    re_y_shape = (0,)*axis + (1, -1) # e.g. (N, 1, CI)  or (N, C, 1, I)
    re_y = n.tops['bilinear'+name+'_re_y'] = L.Reshape(x, shape=dict(dim=list(re_y_shape)))
    tile_re_y = n.tops['bilinear'+name+'_tile_re_y'] = L.Tile(re_y, axis=axis, tiles=u_dim) # (N, J, CI)  or (N, C, J, I)
    re_u_shape = (0,) + (1,)*(axis-1) + (-1, 1) # e.g. (N, J, 1) or (N, 1, J, 1)
    re_u = n.tops['bilinear'+name+'_re_u'] = L.Reshape(u, shape=dict(dim=list(re_u_shape)))
    if axis == 1:
        tile_re_u = n.tops['bilinear'+name+'_tile_re_u'] = L.Tile(re_u, axis=axis+1, tiles=np.prod(x_shape[axis-1:])) # (N, J, CI)
    else:
        tile_re_u1 = n.tops['bilinear'+name+'_tile_re_u1'] = L.Tile(re_u, axis=axis+1, tiles=np.prod(x_shape[axis-1:])) # (N, 1, J, I)
        tile_re_u = n.tops['bilinear'+name+'_tile_re_u'] = L.Tile(tile_re_u1, axis=1, tiles=np.prod(x_shape[0]))  # (N, C, J, I)
    outer_yu = n.tops['bilinear'+name+'_outer_yu'] = L.Eltwise(tile_re_y, tile_re_u, operation=P.Eltwise.PROD) # e.g. (N, J, CI) or (N, C, J, I)
    # outer_yu, u -> y_next_pred
    if share_weights:
        # bilinear term
        bilinear_yu = n.tops['bilinear'+name+'_bilinear_yu'] = L.InnerProduct(outer_yu, num_output=np.prod(x_shape[axis-1:]), axis=axis, **fc_outer_yu_kwargs) # e.g. (N, CI) or (N, C, I)
        # linear
        linear_y = n.tops['bilinear'+name+'_linear_y'] = L.InnerProduct(x, num_output=np.prod(x_shape[axis-1:]), axis=axis, **fc_y_kwargs) # e.g. (N, CI) or (N, I)
        # linear and bias terms
        fc_u = n.tops['bilinear'+name+'_linear_u'] = L.InnerProduct(u, num_output=np.prod(x_shape[axis-1:]), **fc_u_kwargs) # e.g. (N, CI) or (N, I)
        if axis == 1:
            linear_u = fc_u
        else:
            re_fc_u_shape = (0,) + (1,)*(axis-1) + (np.prod(x_shape[axis-1:]),) # e.g. (N, 1, I)
            re_fc_u = n.tops['bilinear'+name+'_re_fc_u'] = L.Reshape(fc_u, shape=dict(dim=list(re_fc_u_shape)))
            linear_u = n.tops['bilinear'+name+'_tile_re_fc_u'] = L.Tile(re_fc_u, axis=1, tiles=x_shape[0]) # e.g. (N, C, I)
        bilinear_yu_linear_u = n.tops['bilinear'+name+'_bilinear_yu_linear_u'] = L.Eltwise(bilinear_yu, linear_y, linear_u, operation=P.Eltwise.SUM) # e.g. (N, C, I)
    else:
        assert axis == 2
        # bilinear term
        outer_yu_channels = L.Slice(outer_yu, ntop=x_shape[0], slice_param=dict(axis=1, slice_point=range(1, x_shape[0])))
        bilinear_yu_channels = []
        for channel, outer_yu_channel in enumerate(outer_yu_channels):
            n.tops['bilinear'+name+'_outer_yu_%d'%channel] = outer_yu_channel
            n.tops['bilinear'+name+'_bilinear_yu_%d'%channel] = \
            bilinear_yu_channel = L.InnerProduct(outer_yu_channel, num_output=np.prod(x_shape[axis-1:]), axis=axis, **fc_outer_yu_kwargs) # e.g. (N, 1, I)
            bilinear_yu_channels.append(bilinear_yu_channel)
        bilinear_yu = n.tops['bilinear'+name+'_bilinear_yu'] = L.Concat(*bilinear_yu_channels, axis=1) # e.g. (N, C, I)
        # linear
        y_channels = L.Slice(x, ntop=x_shape[0], slice_param=dict(axis=1, slice_point=range(1, x_shape[0])))
        linear_y_channels = []
        for channel, y_channel in enumerate(y_channels):
            n.tops['bilinear'+name+'_y_%d'%channel] = y_channel
            n.tops['bilinear'+name+'_linear_y_%d'%channel] = \
            linear_y_channel = L.InnerProduct(y_channel, num_output=np.prod(x_shape[axis-1:]), axis=axis, **fc_y_kwargs) # e.g. (N, 1, I)
            linear_y_channels.append(linear_y_channel)
        linear_y = n.tops['bilinear'+name+'_linear_y'] = L.Concat(*linear_y_channels, axis=1) # e.g. (N, C, I)
        # linear and bias terms
        re_fc_u_shape = (0,) + (1,)*(axis-1) + (np.prod(x_shape[axis-1:]),) # e.g. (N, 1, I)
        linear_u_channels = []
        for channel in range(x_shape[0]):
            fc_u_channel = n.tops['bilinear'+name+'_linear_u_%d'%channel] = L.InnerProduct(u, num_output=np.prod(x_shape[axis-1:]), **fc_u_kwargs) # e.g. (N, I)
            linear_u_channel = n.tops['bilinear'+name+'_re_fc_u_%d'%channel] = L.Reshape(fc_u_channel, shape=dict(dim=list(re_fc_u_shape))) # e.g. (N, 1, I)
            linear_u_channels.append(linear_u_channel)
        linear_u = n.tops['bilinear'+name+'_linear_u'] = L.Concat(*linear_u_channels, axis=1) # e.g. (N, C, I)

        bilinear_yu_linear_u = n.tops['bilinear'+name+'_bilinear_yu_linear_u'] = L.Eltwise(bilinear_yu, linear_y, linear_u, operation=P.Eltwise.SUM) # e.g. (N, C, I)
    x_diff_pred = L.Reshape(bilinear_yu_linear_u, shape=dict(dim=list((0,) + x_shape)))
    return x_diff_pred

def fcn_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, levels=None, x1_c_dim=16, num_downsample=0, share_bilinear_weights=True, ladder_loss=True, batch_normalization=False, concat=False, **kwargs):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    u_dim = u_shape[0]
    levels = levels or [3]
    levels = sorted(set(levels))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname, shuffle=True)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    x, u = n.image_curr, n.vel

    weight_fillers = OrderedDict()
    if num_downsample:
        ds_kernel = cv2.getGaussianKernel(ksize=2, sigma=-1)
        ds_weight_filler = ds_kernel.dot(ds_kernel.T)

    # preprocess
    x0 = x
    x0_shape = x_shape
    for i_ds in range(num_downsample):
        n.tops['x0_ds%d'%(i_ds+1)] = \
        x0 = L.Convolution(x0,
                           param=[dict(lr_mult=0, decay_mult=0)],
                           convolution_param=dict(num_output=x0_shape[0], kernel_size=2, stride=2, pad=0,
                                                  group=x0_shape[0], bias_term=False))
        weight_fillers['x0_ds%d'%(i_ds+1)] = [ds_weight_filler]
        x0_shape = (x0_shape[0], x0_shape[1]//2, x0_shape[2]//2)
    if num_downsample > 0:
        n.x0 = n.tops.pop('x0_ds%d'%num_downsample)
        weight_fillers['x0'] = weight_fillers.pop('x0_ds%d'%num_downsample)

    # encoding
    xlevels = OrderedDict()
    xlevels_shape = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            xlevel = x0
            xlevel_shape = x0_shape
        else:
            if level == 1:
                xlevelm1_c_dim = x0_shape[0]
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevelm1_c_dim
            n.tops['x%d_1'%level] = \
            xlevel_1 = L.Convolution(xlevels[level-1],
                                     param=[dict(name='x%d_1.w'%level, lr_mult=1, decay_mult=1),
                                            dict(name='x%d_1.b'%level, lr_mult=1, decay_mult=1)],
                                     convolution_param=dict(num_output=xlevel_c_dim, kernel_size=3, stride=1, pad=1,
                                                            weight_filler=dict(type='gaussian', std=0.01),
                                                            bias_filler=dict(type='constant', value=0)))
            if batch_normalization:
                n.tops['bnx%d_1'%level] = \
                xlevel_1 = L.BatchNorm(xlevel_1, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
            n.tops['rx%d_1'%level] = L.ReLU(xlevel_1, in_place=True)
            n.tops['x%d_2'%level] = \
            xlevel_2 = L.Convolution(xlevel_1,
                                     param=[dict(name='x%d_2.w'%level, lr_mult=1, decay_mult=1),
                                            dict(name='x%d_2.b'%level, lr_mult=1, decay_mult=1)],
                                     convolution_param=dict(num_output=xlevel_c_dim, kernel_size=3, stride=1, pad=1,
                                                            weight_filler=dict(type='gaussian', std=0.01),
                                                            bias_filler=dict(type='constant', value=0)))
            if batch_normalization:
                n.tops['bnx%d_2'%level] = \
                xlevel_2 = L.BatchNorm(xlevel_2, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
            n.tops['rx%d_2'%level] = L.ReLU(xlevel_2, in_place=True)
            n.tops['x%d'%level] = \
            xlevel = L.Pooling(xlevel_2, pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
            xlevel_shape = (xlevel_c_dim, xlevels_shape[level-1][1]//2, xlevels_shape[level-1][2]//2)
        xlevels[level] = xlevel
        xlevels_shape[level] = xlevel_shape

    # bilinear
    xlevels_next_pred_s0 = OrderedDict() # 0th summand
    ylevels = OrderedDict()
    ylevels_diff_pred = OrderedDict()
    for level in levels:
        xlevel, xlevel_shape = xlevels[level], xlevels_shape[level]
        n.tops['x%d_diff_pred_s0'%level] = \
        xlevel_diff_pred_s0 = ImageBilinearChannelwise(n, xlevel, u, xlevel_shape, u_dim,
                                                       dict(param=[dict(lr_mult=1, decay_mult=1),
                                                                   dict(lr_mult=1, decay_mult=1),
                                                                   dict(lr_mult=1, decay_mult=1),
                                                                   dict(lr_mult=1, decay_mult=1)],
                                                            bilinear_filler=dict(type='gaussian', std=0.001),
                                                            linear_y_filler=dict(type='gaussian', std=0.001),
                                                            linear_u_filler=dict(type='gaussian', std=0.001),
                                                            bias_filler=dict(type='constant', value=0)),
                                                       axis=2,
                                                       name=str(level),
                                                       share_weights=share_bilinear_weights)
        ylevels[level] = n.tops['y%d'%level] = L.Flatten(xlevel)
        ylevels_diff_pred[level] = n.tops['y%d_diff_pred'%level] = L.Flatten(xlevel_diff_pred_s0)
        xlevels_next_pred_s0[level] = n.tops['x%d_next_pred_s0'%level] = L.Eltwise(xlevel, xlevel_diff_pred_s0, operation=P.Eltwise.SUM)

    # features
    n.y = L.Concat(*ylevels.values(), concat_param=dict(axis=1))
    n.y_diff_pred = L.Concat(*ylevels_diff_pred.values(), concat_param=dict(axis=1))

    # decoding
    xlevels_next_pred = OrderedDict()
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            xlevel_next_pred = xlevels_next_pred_s0[level]
        else:
            if level == 0:
                xlevel_c_dim = x1_c_dim
                xlevelm1_c_dim = x0_shape[0]
            elif level < levels[-1]-1:
                xlevel_c_dim = xlevelm1_c_dim
                xlevelm1_c_dim = xlevel_c_dim // 2
            n.tops['x%d_next_pred_2'%(level+1)] = \
            xlevel_next_pred_2 = L.Deconvolution(xlevels_next_pred[level+1],
                                                 param=[dict(lr_mult=1, decay_mult=1)],
                                                 convolution_param=dict(num_output=xlevel_c_dim, kernel_size=2, stride=2, pad=0,
                                                                        group=xlevel_c_dim, bias_term=False,
                                                                        weight_filler=dict(type='bilinear')))
            # TODO: nonlinearity needed?
            n.tops['x%d_next_pred_1'%(level+1)] = \
            xlevel_next_pred_1 = L.Deconvolution(xlevel_next_pred_2,
                                                 param=[dict(lr_mult=1, decay_mult=1),
                                                        dict(lr_mult=1, decay_mult=1)],
                                                 convolution_param=dict(num_output=xlevel_c_dim, kernel_size=3, stride=1, pad=1,
                                                                        weight_filler=dict(type='gaussian', std=0.01),
                                                                        bias_filler=dict(type='constant', value=0)))
            if batch_normalization:
                n.tops['bnx%d_next_pred_1'%(level+1)] = \
                xlevel_next_pred_1 = L.BatchNorm(xlevel_next_pred_1, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
            n.tops['rx%d_next_pred_1'%(level+1)] = L.ReLU(xlevel_next_pred_1, in_place=True)
            if concat:
                if level in xlevels_next_pred_s0:
                    n.tops['cx%d_next_pred_1'%(level+1)] = \
                    xlevel_next_pred_1 = L.Concat(xlevels_next_pred_s0[level], xlevel_next_pred_1)
                n.tops['x%d_next_pred'%level] = \
                xlevel_next_pred= L.Deconvolution(xlevel_next_pred_1,
                                                  param=[dict(lr_mult=1, decay_mult=1),
                                                         dict(lr_mult=1, decay_mult=1)],
                                                  convolution_param=dict(num_output=xlevelm1_c_dim, kernel_size=3, stride=1, pad=1,
                                                                         weight_filler=dict(type='gaussian', std=0.01),
                                                                         bias_filler=dict(type='constant', value=0)))
                if level != 0: # or level in xlevels_next_pred_s0:
                    if batch_normalization:
                        n.tops['bnx%d_next_pred'%level] = \
                        xlevel_next_pred = L.BatchNorm(xlevel_next_pred, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
                    n.tops['rx%d_next_pred'%level] = L.ReLU(xlevel_next_pred, in_place=True)
            else:
                n.tops['x%d_next_pred_s1'%level] = \
                xlevel_next_pred_s1 = L.Deconvolution(xlevel_next_pred_1,
                                                      param=[dict(lr_mult=1, decay_mult=1),
                                                             dict(lr_mult=1, decay_mult=1)],
                                                      convolution_param=dict(num_output=xlevelm1_c_dim, kernel_size=3, stride=1, pad=1,
                                                                             weight_filler=dict(type='gaussian', std=0.01),
                                                                             bias_filler=dict(type='constant', value=0)))
                if level != 0: # or level in xlevels_next_pred_s0:
                    if batch_normalization:
                        n.tops['bnx%d_next_pred_s1'%level] = \
                        xlevel_next_pred_s1 = L.BatchNorm(xlevel_next_pred_s1, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
                    n.tops['rx%d_next_pred_s1'%level] = L.ReLU(xlevel_next_pred_s1, in_place=True)
                if level in xlevels_next_pred_s0:
                    # sum using fixed coeffs
                    # n.tops['x%d_next_pred'%level] = \
                    # xlevel_next_pred = L.Eltwise(xlevels_next_pred_s0[level], xlevel_next_pred_s1, operation=P.Eltwise.SUM, coeff=[.5, .5])
                    # workaround to sum using learnable coeffs
                    xlevel_shape = xlevels_shape[level]
                    re_xlevel_next_pred_s0 = n.tops['re_x%d_next_pred_s0'%level] = L.Reshape(xlevels_next_pred_s0[level], shape=dict(dim=list((0, 1, -1))))
                    re_xlevel_next_pred_s1 = n.tops['re_x%d_next_pred_s1'%level] = L.Reshape(xlevel_next_pred_s1, shape=dict(dim=list((0, 1, -1))))
                    re_xlevel_next_pred_s01 = n.tops['re_x%d_next_pred_s01'%level] = L.Concat(re_xlevel_next_pred_s0, re_xlevel_next_pred_s1)
                    n.tops['re_x%d_next_pred'%level] = \
                    re_xlevel_next_pred = L.Convolution(re_xlevel_next_pred_s01,
                                                        param=[dict(lr_mult=1, decay_mult=1)],
                                                        convolution_param=dict(num_output=1, kernel_size=1, stride=1, pad=0,
                                                                               bias_term=False,
                                                                               weight_filler=dict(type='constant', value=0.5),
                                                                               engine=P.Convolution.CAFFE))
                    xlevel_next_pred = n.tops['re_x%d_next_pred'%level] = L.Reshape(re_xlevel_next_pred, shape=dict(dim=list((0,) + xlevel_shape)))
                else:
                    n.tops['x%d_next_pred'%level] = n.tops.pop('x%d_next_pred_s1'%level)
                    xlevel_next_pred = xlevel_next_pred_s1
        xlevels_next_pred[level] = xlevel_next_pred

    x_next_pred = n.image_next_pred = L.TanH(xlevels_next_pred[0])
    x0_next_pred = x_next_pred
    if num_downsample > 0:
        n.x0_next_pred = n.tops.pop('image_next_pred') # for consistent name (i.e. all image or all x0)

    x_next = n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    # preprocess
    x0_next = x_next
    for i_ds in range(num_downsample):
        n.tops['x0_next_ds%d'%(i_ds+1)] = \
        x0_next = L.Convolution(x0_next,
                                param=[dict(lr_mult=0, decay_mult=0)],
                                convolution_param=dict(num_output=x0_shape[0], kernel_size=2, stride=2, pad=0,
                                                       group=x0_shape[0], bias_term=False))
        weight_fillers['x0_next_ds%d'%(i_ds+1)] = [ds_weight_filler]
    if num_downsample > 0:
        n.x0_next = n.tops.pop('x0_next_ds%d'%num_downsample)
        weight_fillers['x0_next'] = weight_fillers.pop('x0_next_ds%d'%num_downsample)

    n.x0_next_loss = L.EuclideanLoss(x0_next, x0_next_pred)

    if ladder_loss:
        # encoding of next image
        xlevels_next = OrderedDict()
        for level in range(levels[-1]+1):
            if level == 0:
                xlevel_next = x0_next
            else:
                if level == 1:
                    xlevelm1_c_dim = x0_shape[0]
                    xlevel_c_dim = x1_c_dim
                else:
                    xlevelm1_c_dim = xlevel_c_dim
                    xlevel_c_dim = 2 * xlevelm1_c_dim
                n.tops['x%d_next_1'%level] = \
                xlevel_next_1 = L.Convolution(xlevels_next[level-1],
                                              param=[dict(name='x%d_1.w'%level, lr_mult=1, decay_mult=1),
                                                     dict(name='x%d_1.b'%level, lr_mult=1, decay_mult=1)],
                                              convolution_param=dict(num_output=xlevel_c_dim, kernel_size=3, stride=1, pad=1,
                                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                                     bias_filler=dict(type='constant', value=0)))
                if batch_normalization:
                    n.tops['bnx%d_next_1'%level] = \
                    xlevel_next_1 = L.BatchNorm(xlevel_next_1, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
                n.tops['rx%d_next_1'%level] = L.ReLU(xlevel_next_1, in_place=True)
                n.tops['x%d_next_2'%level] = \
                xlevel_next_2 = L.Convolution(xlevel_next_1,
                                              param=[dict(name='x%d_2.w'%level, lr_mult=1, decay_mult=1),
                                                     dict(name='x%d_2.b'%level, lr_mult=1, decay_mult=1)],
                                              convolution_param=dict(num_output=xlevel_c_dim, kernel_size=3, stride=1, pad=1,
                                                                     weight_filler=dict(type='gaussian', std=0.01),
                                                                     bias_filler=dict(type='constant', value=0)))
                if batch_normalization:
                    n.tops['bnx%d_next_2'%level] = \
                    xlevel_next_2 = L.BatchNorm(xlevel_next_2, param=[dict(lr_mult=0)]*3, batch_norm_param=dict(use_global_stats=(phase == caffe.TRAIN)))
                n.tops['rx%d_next_2'%level] = L.ReLU(xlevel_next_2, in_place=True)
                n.tops['x%d_next'%level] = \
                xlevel_next = L.Pooling(xlevel_next_2, pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0)
            xlevels_next[level] = xlevel_next

        for level in levels:
            if level == 0:
                continue
            n.tops['x%d_next_loss'%level] = L.EuclideanLoss(xlevels_next[level], xlevels_next_pred[level], loss_weight=1.)

    net = n.to_proto()
    if net_name is None:
        net_name = 'FcnActionCondEncoderNet'
        net_name +='_levels' + ''.join([str(level) for level in levels])
        net_name += '_x1cdim' + str(x1_c_dim)
        net_name += '_numds' + str(num_downsample)
        net_name += '_share' + str(int(share_bilinear_weights))
        net_name += '_ladder' + str(int(ladder_loss))
        net_name += '_bn' + str(int(batch_normalization))
        if concat:
            net_name += '_concat' + str(int(concat))
    net.name = net_name
    return net, weight_fillers

def paper_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name=None, phase=None, **kwargs):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    u_dim = u_shape[0]
    y_dim = 2048

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname, shuffle=True)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)

    # encoding
    n.conv1 = L.Convolution(n.image_curr, name='conv1', num_output=64, kernel_size=8, pad_h=0, pad_w=1,stride=2)
    n.relu1 = L.ReLU(n.conv1, name='relu1', in_place=True)
    n.conv2 = L.Convolution(n.conv1, name='conv2', num_output=128, kernel_size=6, pad=1, stride=2)
    n.relu2 = L.ReLU(n.conv2, name='relu2', in_place=True)
    n.conv3 = L.Convolution(n.conv2, name='conv3', num_output=128, kernel_size=6, pad=1, stride=2)
    n.relu3 = L.ReLU(n.conv3, name='relu3', in_place=True)
    n.conv4 = L.Convolution(n.conv3, name='conv4', num_output=128, kernel_size=4, pad=0, stride=2)
    n.relu4 = L.ReLU(n.conv4, name='relu4', in_place=True)

    n.y0 = L.InnerProduct(n.conv4, num_output=y_dim, weight_filler=dict(type='xavier'))
    n.y = L.InnerProduct(n.y0, num_output=y_dim, weight_filler=dict(type='xavier'))

    # bilinear
    n.y_diff_pred = ImageBilinearChannelwise(n, n.y, n.vel, (y_dim,), u_dim,
                                             dict(param=[dict(lr_mult=1, decay_mult=1),
                                                         dict(lr_mult=0, decay_mult=0),
                                                         dict(lr_mult=1, decay_mult=1)],
                                                  bilinear_filler=dict(type='gaussian', std=0.001),
                                                  linear_filler=dict(type='constant', value=0),
                                                  bias_filler=dict(type='constant', value=0)),
                                             axis=1)
    n.y_next_pred = L.Eltwise(n.y, n.y_diff_pred, operation=P.Eltwise.SUM)

    # decoding
    n.y_next_pred0 = L.InnerProduct(n.y_next_pred, num_output=y_dim, weight_filler=dict(type='xavier'))
    n.y_next_pred1 = L.InnerProduct(n.y_next_pred0, num_output=y_dim, weight_filler=dict(type='xavier'))
    n.relu5 = L.ReLU(n.y_next_pred1, name='relu5', in_place=True)

    n.deconv4 = L.Deconvolution(n.y_next_pred1,
                                param=[dict(lr_mult=1, decay_mult=1),
                                       dict(lr_mult=1, decay_mult=1)],
                                convolution_param=dict(num_output=128, kernel_size=4, pad=0, stride=2,
                                                       weight_filler=dict(type='gaussian', std=0.01),
                                                       bias_filler=dict(type='constant', value=0)))
    n.derelu4 = L.ReLU(n.deconv4, name='derelu4', in_place=True)
    n.deconv3 = L.Deconvolution(n.deconv4,
                                param=[dict(lr_mult=1, decay_mult=1),
                                       dict(lr_mult=1, decay_mult=1)],
                                convolution_param=dict(num_output=128, kernel_size=6, pad=1, stride=2,
                                                       weight_filler=dict(type='gaussian', std=0.01),
                                                       bias_filler=dict(type='constant', value=0)))
    n.derelu3 = L.ReLU(n.deconv3, name='derelu3', in_place=True)
    n.deconv2 = L.Deconvolution(n.deconv3,
                                param=[dict(lr_mult=1, decay_mult=1),
                                       dict(lr_mult=1, decay_mult=1)],
                                convolution_param=dict(num_output=128, kernel_size=6, pad=1, stride=2,
                                                       weight_filler=dict(type='gaussian', std=0.01),
                                                       bias_filler=dict(type='constant', value=0)))
    n.derelu2 = L.ReLU(n.deconv2, name='derelu2', in_place=True)
    n.image_next_pred = L.Deconvolution(n.deconv2,
                                param=[dict(lr_mult=1, decay_mult=1),
                                       dict(lr_mult=1, decay_mult=1)],
                                convolution_param=dict(num_output=3, kernel_size=8, pad_h=0, pad_w=1, stride=2,
                                                       weight_filler=dict(type='gaussian', std=0.01),
                                                       bias_filler=dict(type='constant', value=0)))

    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.x_next_loss = L.EuclideanLoss(n.image_next, n.image_next_pred)

    net = n.to_proto()
    if net_name is None:
        net_name = 'ActionCondEncoderNet2'
    net.name = net_name
    return net, None
