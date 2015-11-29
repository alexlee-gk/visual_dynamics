from __future__ import division

import numpy as np
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

def traverse_layers_to_keep(layer, layers, layers_to_keep):
    if layer in layers_to_keep: # has already been traversed
        return
    layers_to_keep.append(layer)
    for layer_name in layer.bottom:
        if layer_name in layers:
            traverse_layers_to_keep(layers[layer_name], layers, layers_to_keep)

def deploy_net(net, inputs, input_shapes, outputs, batch_size=1, force_backward=True):
    # remove all layers that are not descendants of output layers
    layers = dict([(layer.name, layer) for layer in net.layer])
    layers_to_keep = []
    for output in outputs:
        traverse_layers_to_keep(layers[output], layers, layers_to_keep)
    layer_names_to_remove = set(layers.keys()) - set([l.name for l in layers_to_keep])
    for layer_name in layer_names_to_remove:
        net.layer.remove(layers[layer_name])

    net.input.extend(inputs)
    net.input_shape.extend([pb2.BlobShape(dim=(batch_size,)+shape) for shape in input_shapes])
    net.force_backward = force_backward
    return net

def train_val_net(net):
    # remove all layers that are not descendants of loss layers
    layers = dict([(layer.name, layer) for layer in net.layer])
    layers_to_keep = [layers['data']]
    loss_layers = [layer for layer in layers.values() if layer.name.endswith('loss')]
    for loss_layer in loss_layers:
        traverse_layers_to_keep(loss_layer, layers, layers_to_keep)
    layer_names_to_remove = set(layers.keys()) - set([l.name for l in layers_to_keep])
    for layer_name in layer_names_to_remove:
        net.layer.remove(layers[layer_name])
    return net

def approx_bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='ApproxBilinearNet'):
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
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
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
    re_y = n.tops[name+'_re_y'] = L.Reshape(y, shape=dict(dim=[-1, y_dim, 1]))
    tile_re_y = n.tops[name+'_tile_re_y'] = L.Tile(re_y, axis=2, tiles=u_dim)
    flatten_tile_re_y = n.tops[name+'_flatten_tile_re_y'] = L.Flatten(tile_re_y)
    tile_u = n.tops[name+'_tile_u'] = L.Tile(u, axis=1, tiles=y_dim)
    outer_yu = n.tops[name+'_outer_yu'] = L.Eltwise(flatten_tile_re_y, tile_u, operation=P.Eltwise.PROD)
    fc_outer_yu = n.tops[name+'_fc_outer_yu'] = L.InnerProduct(outer_yu, num_output=y_dim, **fc_kwargs)
    fc_u = n.tops[name+'_fc_u'] = L.InnerProduct(u, num_output=y_dim, **fc_kwargs)
    return L.Eltwise(fc_outer_yu, fc_u, operation=P.Eltwise.SUM)

def bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearNet'):
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
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    u = n.vel
    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = Bilinear(n, n.y, u, y_dim, u_dim, **fc_kwargs)
    n.y_diff = L.Flatten(n.image_diff)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='ActionCondEncoderNet'):
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
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)

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

def small_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='SmallActionCondEncoderNet'):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    y_dim = 128
    u_dim = vel_shape[0]
    conv_num_output = 16
    conv2_wh = 8

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
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)

    n.conv1 = L.Convolution(n.image_curr, **conv_kwargs)
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

    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.loss = L.EuclideanLoss(n.image_next_pred, n.image_next, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def downsampled_small_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='DownsampledSmallActionCondEncoderNet'):
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
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)

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

def hierarchichal_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='HierarchichalActionCondEncoderNet'):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert image0_shape[1] == 32
    assert image0_shape[2] == 32
    assert len(vel_shape) == 1
    image0_num_channel = image0_shape[0]
    image1_num_channel = 16
    image2_num_channel = 16
    image1_shape = (image1_num_channel, 16, 16)
    image2_shape = (image2_num_channel, 8, 8)
    y0_dim = image0_shape[1] * image0_shape[2] # 1024
    y1_dim = 128
    y2_dim = 32
    u_dim = vel_shape[0]

    conv0_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image0_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    conv1_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image1_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    conv2_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                        convolution_param=dict(num_output=image2_num_channel,
                                               kernel_size=6,
                                               stride=2,
                                               pad=2,
                                               weight_filler=dict(type='gaussian', std=0.01),
                                               bias_filler=dict(type='constant', value=0)))
    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)

    n.image1 = L.Convolution(n.image_curr, **conv1_kwargs)
    n.image1 = L.ReLU(n.image1, in_place=True)
    n.image2 = L.Convolution(n.image1, **conv2_kwargs)
    n.image2 = L.ReLU(n.image2, in_place=True)

    u = n.vel

    n.y0 = L.Flatten(n.image_curr)
    n.y0_diff_pred = Bilinear(n, n.y0, u, y0_dim, u_dim, name='bilinear0', **fc_kwargs)
    n.y0_next_pred = L.Eltwise(n.y0, n.y0_diff_pred, operation=P.Eltwise.SUM)
    n.image0_next_pred0 = L.Reshape(n.y0_next_pred, shape=dict(dim=[batch_size]+list(image0_shape)))

    n.y1 = L.InnerProduct(n.image1, num_output=y1_dim, weight_filler=dict(type='xavier'))
    n.y1_diff_pred = Bilinear(n, n.y1, u, y1_dim, u_dim, name='bilinear1', **fc_kwargs)
    n.y1_next_pred = L.Eltwise(n.y1, n.y1_diff_pred, operation=P.Eltwise.SUM)
    n.image1_next_pred1_flat = L.InnerProduct(n.y1_next_pred, num_output=np.prod(image1_shape), weight_filler=dict(type='xavier'))
    n.image1_next_pred1 = L.Reshape(n.image1_next_pred1_flat, shape=dict(dim=[batch_size]+list(image1_shape)))

    n.image0_next_pred1 = L.Deconvolution(n.image1_next_pred1, **conv0_kwargs)
    n.image0_next_pred1 = L.ReLU(n.image0_next_pred1, in_place=True)

    n.y2 = L.InnerProduct(n.image2, num_output=y2_dim, weight_filler=dict(type='xavier'))
    n.y2_diff_pred = Bilinear(n, n.y2, u, y2_dim, u_dim, name='bilinear2', **fc_kwargs)
    n.y2_next_pred = L.Eltwise(n.y2, n.y2_diff_pred, operation=P.Eltwise.SUM)
    n.image2_next_pred2_flat = L.InnerProduct(n.y2_next_pred, num_output=np.prod(image2_shape), weight_filler=dict(type='xavier'))
    n.image2_next_pred2 = L.Reshape(n.image2_next_pred2_flat, shape=dict(dim=[batch_size]+list(image2_shape)))

    n.image1_next_pred2 = L.Deconvolution(n.image2_next_pred2, **conv1_kwargs)
    n.image1_next_pred2 = L.ReLU(n.image1_next_pred2, in_place=True)
    n.image0_next_pred2 = L.Deconvolution(n.image1_next_pred2, **conv0_kwargs)
    n.image0_next_pred2 = L.ReLU(n.image0_next_pred2, in_place=True)

    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)

    n.image0_next_pred0_loss = L.EuclideanLoss(n.image0_next_pred0, n.image_next)
    n.image0_next_pred1_loss = L.EuclideanLoss(n.image0_next_pred1, n.image_next)
    n.image0_next_pred2_loss = L.EuclideanLoss(n.image0_next_pred2, n.image_next)

    n.image1_next_pred_loss = L.EuclideanLoss(n.image1_next_pred1, n.image1_next_pred2)

    n.y01 = L.Concat(n.y0, n.y1, axis=1)
    n.y = L.Concat(n.y01, n.y2, axis=1)
    n.y01_diff_pred = L.Concat(n.y0_diff_pred, n.y1_diff_pred, axis=1)
    n.y_diff_pred = L.Concat(n.y01_diff_pred, n.y2_diff_pred, axis=1)

    net = n.to_proto()
    net.name = net_name
    return net

def ladder_action_cond_encoder_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='LadderActionCondEncoderNet'):
    assert len(input_shapes) == 2
    image0_shape, vel_shape = input_shapes
    assert len(image0_shape) == 3
    assert image0_shape[1] == 32
    assert image0_shape[2] == 32
    assert len(vel_shape) == 1
    image0_num_channel = image0_shape[0]
    image1_num_channel = 16
    image2_num_channel = 16
    image1_shape = (image1_num_channel, 16, 16)
    image2_shape = (image2_num_channel, 8, 8)
    y0_dim = image0_shape[1] * image0_shape[2] # 1024
    y1_dim = 128
    y2_dim = 32
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
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    image0 = n.image_curr
    u = n.vel

    n.image1 = L.Convolution(image0, **conv1_kwargs)
    n.image1 = L.ReLU(n.image1, in_place=True)
    n.image2 = L.Convolution(n.image1, **conv2_kwargs)
    n.image2 = L.ReLU(n.image2, in_place=True)

    n.y2 = L.InnerProduct(n.image2, num_output=y2_dim, weight_filler=dict(type='xavier'))
    n.y2_diff_pred = Bilinear(n, n.y2, u, y2_dim, u_dim, name='bilinear2', **fc_kwargs)
    n.y2_next_pred = L.Eltwise(n.y2, n.y2_diff_pred, operation=P.Eltwise.SUM)
    n.image2_next_pred_flat = L.InnerProduct(n.y2_next_pred, num_output=np.prod(image2_shape), weight_filler=dict(type='xavier'))
    n.image2_next_pred = L.Reshape(n.image2_next_pred_flat, shape=dict(dim=[batch_size]+list(image2_shape)))

    n.y1 = L.InnerProduct(n.image1, num_output=y1_dim, weight_filler=dict(type='xavier'))
    n.y1_diff_pred = Bilinear(n, n.y1, u, y1_dim, u_dim, name='bilinear1', **fc_kwargs)
    n.y1_next_pred = L.Eltwise(n.y1, n.y1_diff_pred, operation=P.Eltwise.SUM)
    n.image1_next_pred1_flat = L.InnerProduct(n.y1_next_pred, num_output=np.prod(image1_shape), weight_filler=dict(type='xavier'))
    n.image1_next_pred1 = L.Reshape(n.image1_next_pred1_flat, shape=dict(dim=[batch_size]+list(image1_shape)))
    n.image1_next_pred2 = L.Deconvolution(n.image2_next_pred, **deconv1_kwargs)
    n.image1_next_pred2 = L.ReLU(n.image1_next_pred2, in_place=True)
    n.image1_next_pred = L.Eltwise(n.image1_next_pred1, n.image1_next_pred2, operation=P.Eltwise.SUM)

    n.y0 = L.Flatten(image0)
    n.y0_diff_pred = Bilinear(n, n.y0, u, y0_dim, u_dim, name='bilinear0', **fc_kwargs)
    n.y0_next_pred = L.Eltwise(n.y0, n.y0_diff_pred, operation=P.Eltwise.SUM)
    n.image0_next_pred0 = L.Reshape(n.y0_next_pred, shape=dict(dim=[batch_size]+list(image0_shape)))
    n.image0_next_pred1 = L.Deconvolution(n.image1_next_pred, **deconv0_kwargs)
    n.image0_next_pred1 = L.ReLU(n.image0_next_pred1, in_place=True)
    n.image0_next_pred = L.Eltwise(n.image0_next_pred0, n.image0_next_pred1, operation=P.Eltwise.SUM)

    n.image_next = L.Eltwise(n.image_curr, n.image_diff, operation=P.Eltwise.SUM)
    image0_next = n.image_next

    n.image1_next = L.Convolution(image0_next, **conv1_kwargs)
    n.image1_next = L.ReLU(n.image1_next, in_place=True)
    n.image2_next = L.Convolution(n.image1_next, **conv2_kwargs)
    n.image2_next = L.ReLU(n.image2_next, in_place=True)

    n.image0_next_loss = L.EuclideanLoss(image0_next, n.image0_next_pred)
    n.image1_next_loss = L.EuclideanLoss(n.image1_next, n.image1_next_pred)
    n.image2_next_loss = L.EuclideanLoss(n.image2_next, n.image2_next_pred)

    n.y01 = L.Concat(n.y0, n.y1, axis=1)
    n.y = L.Concat(n.y01, n.y2, axis=1)
    n.y01_diff_pred = L.Concat(n.y0_diff_pred, n.y1_diff_pred, axis=1)
    n.y_diff_pred = L.Concat(n.y01_diff_pred, n.y2_diff_pred, axis=1)

    net = n.to_proto()
    net.name = net_name
    return net
