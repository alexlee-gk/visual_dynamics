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

def deploy_net(net, inputs, input_shapes, outputs, batch_size=1, force_backward=True):
    # remove data layers and the ones that depend on them (except for inputs) or the output
    layers_to_remove = [layer for layer in net.layer if not layer.bottom]
    for output in outputs:
        layers_to_remove.extend([layer for layer in net.layer if output in layer.bottom])
    for layer_to_remove in layers_to_remove:
        if layer_to_remove not in net.layer:
            continue
        tops_to_remove = set(layer_to_remove.top) - set(inputs)
        net.layer.remove(layer_to_remove)
        for top_to_remove in tops_to_remove:
            layers_to_remove.extend([layer for layer in net.layer if top_to_remove in layer.bottom])

    net.input.extend(inputs)
    net.input_shape.extend([pb2.BlobShape(dim=(batch_size,)+shape) for shape in input_shapes])
    net.force_backward = force_backward
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

def bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearNet'):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    _, height, width = image_shape
    y_dim = height * width
    u_dim = vel_shape[0]

    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     num_output=y_dim,
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    u = n.vel
    n.y = L.Flatten(n.image_curr, name='flatten1')
    n.y_diff = L.Flatten(n.image_diff, name='flatten2')
    n.re_y = L.Reshape(n.y, name='reshape', shape=dict(dim=[-1, y_dim, 1]))
    n.tile_re_y = L.Tile(n.re_y, name='tile1', axis=2, tiles=u_dim)
    n.flatten_tile_re_y = L.Flatten(n.tile_re_y, name='flatten3')
    n.tile_u = L.Tile(u, name='tile2', axis=1, tiles=y_dim)
    n.outer_yu = L.Eltwise(n.flatten_tile_re_y, n.tile_u, name='prod', operation=P.Eltwise.PROD)
    n.fc_outer_yu = L.InnerProduct(n.outer_yu, name='fc1', **fc_kwargs)
    n.fc_u = L.InnerProduct(u, name='fc2', **fc_kwargs)
    n.y_diff_pred = L.Eltwise(n.fc_outer_yu, n.fc_u, name='sum', operation=P.Eltwise.SUM)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net

def conv_bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearNet'):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    _, height, width = image_shape
    y_dim = height * width
    u_dim = vel_shape[0]

    fc_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     num_output=y_dim+18,
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))
    conv_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                       kernel_size=5,
                       num_output=1,
                       weight_filler=dict(type='xavier'),
                       bias_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    u = n.vel

    n.y = L.Flatten(n.image_curr, name='flatten1')
    n.conv_image_curr = L.Convolution(n.image_curr, name='conv', **conv_kwargs)
    n.conv_y = L.Flatten(n.conv_image_curr)
    n.concat_y = L.Concat(n.y, n.conv_y)
    n.re_y = L.Reshape(n.concat_y, name='reshape', shape=dict(dim=[-1, y_dim+18, 1]))
    n.tile_re_y = L.Tile(n.re_y, name='tile1', axis=2, tiles=u_dim)
    n.flatten_tile_re_y = L.Flatten(n.tile_re_y, name='flatten3')
    n.tile_u = L.Tile(u, name='tile2', axis=1, tiles=y_dim+18)
    n.outer_yu = L.Eltwise(n.flatten_tile_re_y, n.tile_u, name='prod', operation=P.Eltwise.PROD)
    n.fc_outer_yu = L.InnerProduct(n.outer_yu, name='fc1', **fc_kwargs)
    n.fc_u = L.InnerProduct(u, name='fc2', **fc_kwargs)
    n.y_diff_pred = L.Eltwise(n.fc_outer_yu, n.fc_u, name='sum', operation=P.Eltwise.SUM)

    n.y_diff = L.Flatten(n.image_diff, name='flatten2')
    n.conv_image_diff = L.Convolution(n.image_diff, name='conv', **conv_kwargs)
    n.conv_y_diff = L.Flatten(n.conv_image_diff)
    n.concat_y_diff = L.Concat(n.y_diff, n.conv_y_diff)

    n.loss = L.EuclideanLoss(n.y_diff_pred, n.concat_y_diff, name='loss')

    net = n.to_proto()
    net.name = net_name
    return net
