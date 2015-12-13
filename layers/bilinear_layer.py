import argparse
import numpy as np
import time
import caffe
from caffe import layers as L
from caffe import params as P
from predictor import NetFeaturePredictor


class BilinearLayer(caffe.Layer):

    def setup(self, bottom, top):
        self.param_propagate_down = True
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute outer product.")
        y, u = bottom[0].data, bottom[1].data
        assert y.ndim == 2
        assert u.ndim == 2
        self.blobs.add_blob(y.shape[1], y.shape[1], u.shape[1])
        self.blobs.add_blob(y.shape[1], u.shape[1])
        self.blobs.add_blob(y.shape[1])
        self.blobs[0].data[...] *= 0
        self.blobs[1].data[...] *= 0
        self.blobs[2].data[...] *= 0

    def reshape(self, bottom, top):
        y, u = bottom[0].data, bottom[1].data
        assert y.ndim == 2
        assert u.ndim == 2
        top[0].reshape(*y.shape)

    def forward(self, bottom, top):
        y, u = bottom[0].data, bottom[1].data
        K = I = y.shape[1]
        J = u.shape[1]
        N = y.shape[0]

        self.outer_yu = np.empty((N, I*J)) # (y[:, :, None] * u[:, None, :]).reshape((N, I*J))
        for n in range(N):
            self.outer_yu[n, :] = y[n, :][:, None].dot(u[n, :][:, None].T).reshape((1, I*J))

        self.blobs[0].reshape(K, I*J)
        y_dot = self.outer_yu.dot(self.blobs[0].data.T) + u.dot(self.blobs[1].data.T) + self.blobs[2].data
        self.blobs[0].reshape(K, I, J)
        top[0].data[...] = y_dot

    def backward(self, top, propagate_down, bottom):
        y, u = bottom[0].data, bottom[1].data
        K = I = y.shape[1]
        J = u.shape[1]
        N = y.shape[0]

        if self.param_propagate_down:
            # Gradient with respect to bilinear weight
            self.blobs[0].reshape(K, I*J)
            self.blobs[0].diff[...] += top[0].diff.T.dot(self.outer_yu)
            self.blobs[0].reshape(K, I, J)

            # Gradient with respect to linear weight
            self.blobs[1].diff[...] += top[0].diff.T.dot(u)

            # Gradient with respect to bias
            self.blobs[2].diff[...] += top[0].diff.sum(axis=0) # TODO bias multiplier

        if propagate_down[0] or propagate_down[1]:
            outer_yu_diff = top[0].diff.dot(self.blobs[0].data.reshape((K, I*J))).reshape((N, I, J))

        if propagate_down[0]:
            # Gradient with respect to y
            bottom[0].diff[...] = np.einsum("nij,nj->ni", outer_yu_diff, u)

        if propagate_down[1]:
            # Gradient with respect to u
            bottom[1].diff[...] = np.einsum("nij,ni->nj", outer_yu_diff, y) + top[0].diff.dot(self.blobs[1].data)


def Bilinear(n, y, u, y_dim, u_dim, fc_outer_kwargs, fc_u_kwargs, name='bilinear'):
    re_y = n.tops[name+'_re_y'] = L.Reshape(y, shape=dict(dim=[0, -1, 1]))
    tile_re_y = n.tops[name+'_tile_re_y'] = L.Tile(re_y, axis=2, tiles=u_dim)
    re_u = n.tops[name+'_re_u'] = L.Reshape(u, shape=dict(dim=[0, 1, -1]))
    tile_re_u = n.tops[name+'_tile_re_u'] = L.Tile(re_u, axis=1, tiles=y_dim)
    outer_yu = n.tops[name+'_outer_yu'] = L.Eltwise(tile_re_y, tile_re_u, operation=P.Eltwise.PROD)
    fc_outer_yu = n.tops[name+'_fc_outer_yu'] = L.InnerProduct(outer_yu, num_output=y_dim, **fc_outer_kwargs)
    fc_u = n.tops[name+'_fc_u'] = L.InnerProduct(u, num_output=y_dim, **fc_u_kwargs)
    return L.Eltwise(fc_outer_yu, fc_u, operation=P.Eltwise.SUM)

def bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='BilinearNet', phase=None, **kwargs):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1
    y_dim = np.prod(image_shape)
    u_dim = vel_shape[0]

    fc_outer_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='constant', value=0))
    fc_u_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                     weight_filler=dict(type='gaussian', std=0.001),
                     bias_filler=dict(type='gaussian', std=0.001))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = Bilinear(n, n.y, u, y_dim, u_dim, fc_outer_kwargs, fc_u_kwargs)
    n.y_diff = L.Flatten(n.image_diff)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    n.image_diff_pred = L.Reshape(n.y_diff_pred, shape=dict(dim=[batch_size] + list(image_shape)))
    n.image_next_pred = L.Eltwise(n.image_curr, n.image_diff_pred, operation=P.Eltwise.SUM)

    net = n.to_proto()
    net.name = net_name
    return net

def python_bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='PythonBilinearNet', phase=None, **kwargs):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = L.Python(n.y, u, module='layers.bilinear_layer', layer='BilinearLayer')
    n.y_diff = L.Flatten(n.image_diff)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    n.image_diff_pred = L.Reshape(n.y_diff_pred, shape=dict(dim=[batch_size] + list(image_shape)))
    n.image_next_pred = L.Eltwise(n.image_curr, n.image_diff_pred, operation=P.Eltwise.SUM)

    net = n.to_proto()
    net.name = net_name
    return net

def cpu_bilinear_net(input_shapes, hdf5_txt_fname='', batch_size=1, net_name='CpuBilinearNet', phase=None, **kwargs):
    assert len(input_shapes) == 2
    image_shape, vel_shape = input_shapes
    assert len(image_shape) == 3
    assert len(vel_shape) == 1

    bilinear_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=1)],
                           bilinear_filler=dict(type='gaussian', std=0.001),
                           linear_filler=dict(type='gaussian', std=0.001))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = L.Bilinear(n.y, u, **bilinear_kwargs)
    n.y_diff = L.Flatten(n.image_diff)
    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    n.image_diff_pred = L.Reshape(n.y_diff_pred, shape=dict(dim=[batch_size] + list(image_shape)))
    n.image_next_pred = L.Eltwise(n.image_curr, n.image_diff_pred, operation=P.Eltwise.SUM)

    net = n.to_proto()
    net.name = net_name
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('val_hdf5_fname', nargs='?', type=str, default=None)

    args = parser.parse_args()

    np.random.seed(0)

    inputs = ['image_curr', 'vel']
    input_shapes = NetFeaturePredictor.infer_input_shapes(inputs, None, args.train_hdf5_fname)
    outputs = ['y_diff_pred', 'y', 'image_next_pred']
    net = NetFeaturePredictor(bilinear_net, inputs, input_shapes, outputs)
    net_p = NetFeaturePredictor(python_bilinear_net, inputs, input_shapes, outputs)
    net_c = NetFeaturePredictor(cpu_bilinear_net, inputs, input_shapes, outputs)
    image_shape, vel_shape = input_shapes
    bilinear_param_shape = (np.prod(image_shape), np.prod(image_shape), np.prod(vel_shape))
    bilinear_param = net.params['bilinear_fc_outer_yu'][0]
    linear_param = net.params['bilinear_fc_u'][0]
    bias_param = net.params['bilinear_fc_u'][1]
    assert np.any(bilinear_param)
    assert np.any(linear_param)
    def make_diff_1(net):
        for blob in net.blobs.values():
            blob.diff[...] *= 0
            blob.diff[...] += 1
        for param in net.params.values():
            for blob in param:
                blob.diff[...] *= 0
                blob.diff[...] += 1
    make_diff_1(net)

    # generate inputs
    image_curr = np.random.random(input_shapes[0])*2 - 1
    vel = np.random.random(input_shapes[1])*2 - 1
    # compute output
    forward_kwargs = dict(zip(inputs, [image_curr[None, ...], vel[None, :]]))
    y_diff_pred = net.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    # compute diffs
    backward_kwargs = dict(y_diff_pred=y_diff_pred)
    diffs = net.backward_all(**backward_kwargs)
    image_curr_diff, vel_diff = diffs['image_curr'], diffs['vel']

    ### python bilinear net
    net_p.params['y_diff_pred'][0].data[...] = bilinear_param.data.reshape(bilinear_param_shape)
    net_p.params['y_diff_pred'][1].data[...] = linear_param.data
    net_p.params['y_diff_pred'][2].data[...] = bias_param.data
    make_diff_1(net_p)
    y_diff_pred_p = net_p.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    print "forward python", np.linalg.norm(y_diff_pred - y_diff_pred_p)
    diffs_p = net_p.backward_all(**backward_kwargs)
    image_curr_diff_p, vel_diff_p = diffs_p['image_curr'], diffs_p['vel']
    print "backward python", np.linalg.norm(image_curr_diff - image_curr_diff_p), np.linalg.norm(vel_diff - vel_diff_p)
    print "backward python", np.linalg.norm(net_p.params['y_diff_pred'][0].diff - bilinear_param.diff.reshape(bilinear_param_shape))
    print "backward python", np.linalg.norm(net_p.params['y_diff_pred'][1].diff - linear_param.diff)

    ### cpu bilinear net
    caffe.set_mode_cpu()
    net_c.params['y_diff_pred'][0].data[...] = bilinear_param.data.reshape(bilinear_param_shape)
    net_c.params['y_diff_pred'][1].data[...] = linear_param.data
    net_c.params['y_diff_pred'][2].data[...] = bias_param.data
    make_diff_1(net_c)
    y_diff_pred_c = net_c.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    print "forward cpu", np.linalg.norm(y_diff_pred - y_diff_pred_c)
    diffs_c = net_c.backward_all(**backward_kwargs)
    image_curr_diff_c, vel_diff_c = diffs_c['image_curr'], diffs_c['vel']
    print "backward cpu", np.linalg.norm(image_curr_diff - image_curr_diff_c), np.linalg.norm(vel_diff - vel_diff_c)
    print "backward cpu", np.linalg.norm(net_c.params['y_diff_pred'][0].diff - bilinear_param.diff.reshape(bilinear_param_shape))
    print "backward cpu", np.linalg.norm(net_c.params['y_diff_pred'][1].diff - linear_param.diff)

    ### gpu bilinear net
    caffe.set_mode_gpu()
    net_c.params['y_diff_pred'][0].data[...] = bilinear_param.data.reshape(bilinear_param_shape)
    net_c.params['y_diff_pred'][1].data[...] = linear_param.data
    net_c.params['y_diff_pred'][2].data[...] = bias_param.data
    make_diff_1(net_c)
    y_diff_pred_c = net_c.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    print "forward gpu", np.linalg.norm(y_diff_pred - y_diff_pred_c)
    diffs_c = net_c.backward_all(**backward_kwargs)
    image_curr_diff_c, vel_diff_c = diffs_c['image_curr'], diffs_c['vel']
    print "backward gpu", np.linalg.norm(image_curr_diff - image_curr_diff_c), np.linalg.norm(vel_diff - vel_diff_c)
    print "backward gpu", np.linalg.norm(net_c.params['y_diff_pred'][0].diff - bilinear_param.diff.reshape(bilinear_param_shape))
    print "backward gpu", np.linalg.norm(net_c.params['y_diff_pred'][1].diff - linear_param.diff)


    def time_forward(net, n_iter=100):
        start_time = time.time()
        for _ in range(n_iter):
            net.forward_all(blobs=['y_diff_pred'], **forward_kwargs)
        return (time.time() - start_time) / n_iter
    def time_backward(net, n_iter=100):
        start_time = time.time()
        for _ in range(n_iter):
            net.backward_all(**backward_kwargs)
        return (time.time() - start_time) / n_iter

    print "forward duration", time_forward(net)
    print "backward duration", time_backward(net)

    print "forward python duration", time_forward(net_p)
    print "backward python duration", time_backward(net_p)

    caffe.set_mode_cpu()
    print "forward cpu duration", time_forward(net_c)
    print "backward cpu duration", time_backward(net_c)

    caffe.set_mode_gpu()
    print "forward gpu duration", time_forward(net_c)
    print "backward gpu duration", time_backward(net_c)

if __name__ == "__main__":
    main()
