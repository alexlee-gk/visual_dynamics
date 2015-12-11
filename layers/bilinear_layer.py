import argparse
import numpy as np
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
        self.blobs[0].data[...] *= 0
        self.blobs[1].data[...] *= 0

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
            self.outer_yu[n, :] = y[n, :][:, None].dot(u[n, :][:, None].T).reshape((N, I*J))

        self.blobs[0].reshape(K, I*J)
        y_dot = self.outer_yu.dot(self.blobs[0].data.T) + u.dot(self.blobs[1].data.T)
        self.blobs[0].reshape(K, I, J)
        top[0].data[...] = y_dot

    def backward(self, top, propagate_down, bottom):
        y, u = bottom[0].data, bottom[1].data
        K = I = y.shape[1]
        J = u.shape[1]
        N = y.shape[0]

        if self.param_propagate_down:
            # Gradient with respect to first weights
            self.blobs[0].reshape(K, I*J)
            self.blobs[0].diff[...] = top[0].diff.T.dot(self.outer_yu)
            self.blobs[0].reshape(K, I, J)

            # Gradient with respect to second weights
            self.blobs[1].diff[...] = top[0].diff.T.dot(u)

        if propagate_down[0]:
            # Gradient with respect to y
            # bottom[0].diff[...] = np.einsum("nk,kij,nj->ni", top[0].diff, self.blobs[0].data, u)
            for n in range(N):
                for k in range(K):
                    bottom[0].diff[n, :] += top[0].diff[n, k] * self.blobs[0].data[k, :, :].dot(u[n, :])

        if propagate_down[1]:
            # Gradient with respect to u
            # bottom[1].diff[...] = np.einsum("nk,kij,ni->nj", top[0].diff, self.blobs[0].data, y) + top[0].diff.dot(self.blobs[1].data)
            outer_dy = np.empty((N, K*I)) # (top[0].diff[:, :, None] * y[:, None, :]).reshape((N, K*I))
            for n in range(N):
                outer_dy[n, :] = top[0].diff[n, :][:, None].dot(y[n, :][:, None].T).reshape((N, K*I))
            self.blobs[0].reshape(K*I, J)
            bottom[1].diff[...] = outer_dy.dot(self.blobs[0].data) + top[0].diff.dot(self.blobs[1].data)
            self.blobs[0].reshape(K, I, J)


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

    bilinear_kwargs = dict(param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=0, decay_mult=0)],
                           bilinear_filler=dict(type='gaussian', std=0.001),
                           linear_filler=dict(type='constant', value=0))

    n = caffe.NetSpec()
    data_kwargs = dict(name='data', ntop=3, batch_size=batch_size, source=hdf5_txt_fname)
    if phase is not None:
        data_kwargs.update(dict(include=dict(phase=phase)))
    n.image_curr, n.image_diff, n.vel = L.HDF5Data(**data_kwargs)
    u = n.vel
    n.image_diff_pred = L.Bilinear(n.image_curr, u, **bilinear_kwargs)

    n.y = L.Flatten(n.image_curr)
    n.y_diff_pred = L.Flatten(n.image_diff_pred)
    n.y_diff = L.Flatten(n.image_diff)

    n.loss = L.EuclideanLoss(n.y_diff_pred, n.y_diff, name='loss')

    n.image_next_pred = L.Eltwise(n.image_curr, n.image_diff_pred, operation=P.Eltwise.SUM)

    net = n.to_proto()
    net.name = net_name
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('val_hdf5_fname', nargs='?', type=str, default=None)

    args = parser.parse_args()

    inputs = ['image_curr', 'vel']
    input_shapes = NetFeaturePredictor.infer_input_shapes(inputs, None, args.train_hdf5_fname)
    outputs = ['y_diff_pred', 'y', 'image_next_pred']
    net = NetFeaturePredictor(bilinear_net, inputs, input_shapes, outputs)
    net_p = NetFeaturePredictor(python_bilinear_net, inputs, input_shapes, outputs)
    net_c = NetFeaturePredictor(cpu_bilinear_net, inputs, input_shapes, outputs)
    net_p.params['y_diff_pred'][0].data[...] = net.params['bilinear_fc_outer_yu'][0].data.reshape(net_p.params['y_diff_pred'][0].data.shape)
    net_p.params['y_diff_pred'][1].data[...] = net.params['bilinear_fc_u'][0].data
    net_c.params['image_diff_pred'][0].data[...] = net.params['bilinear_fc_outer_yu'][0].data.reshape(net_c.params['image_diff_pred'][0].data.shape)
    net_c.params['image_diff_pred'][1].data[...] = net.params['bilinear_fc_u'][0].data
    assert not np.any(net.params['bilinear_fc_outer_yu'][1].data)
    assert not np.any(net.params['bilinear_fc_u'][1].data)

    np.random.seed(0)
    image_curr = np.random.random(input_shapes[0])*2 - 1
    vel = np.random.random(input_shapes[1])*2 - 1
    forward_kwargs = dict(zip(inputs, [image_curr[None, ...], vel[None, :]]))

    y_diff_pred = net.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    y_diff_pred_p = net_p.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    y_diff_pred_c = net_c.forward_all(blobs=['y_diff_pred'], **forward_kwargs)['y_diff_pred']
    print "forward python", np.linalg.norm(y_diff_pred - y_diff_pred_p)
    print "forward cpu", np.linalg.norm(y_diff_pred - y_diff_pred_c)

    backward_kwargs = dict(y_diff_pred=y_diff_pred)
    diffs = net.backward_all(start='y_diff_pred', **backward_kwargs)
    image_curr_diff, vel_diff = diffs['image_curr'], diffs['vel']
    diffs_p = net_p.backward_all(start='y_diff_pred', **backward_kwargs)
    image_curr_diff_p, vel_diff_p = diffs_p['image_curr'], diffs_p['vel']
    diffs_c = net_c.backward_all(start='y_diff_pred', **backward_kwargs)
    image_curr_diff_c, vel_diff_c = diffs_c['image_curr'], diffs_c['vel']
    print "backward python", np.linalg.norm(image_curr_diff - image_curr_diff_p), np.linalg.norm(vel_diff - vel_diff_p)
    print "backward python", np.linalg.norm(net_p.params['y_diff_pred'][0].diff - net.params['bilinear_fc_outer_yu'][0].diff.reshape(net_p.params['y_diff_pred'][0].diff.shape))
    print "backward python", np.linalg.norm(net_p.params['y_diff_pred'][1].diff - net.params['bilinear_fc_u'][0].diff)
    print "backward cpu", np.linalg.norm(image_curr_diff - image_curr_diff_c), np.linalg.norm(vel_diff - vel_diff_c)
    print "backward cpu", np.linalg.norm(net_c.params['image_diff_pred'][0].diff - net.params['bilinear_fc_outer_yu'][0].diff.reshape(net_c.params['image_diff_pred'][0].diff.shape))
    print "backward cpu", np.linalg.norm(net_c.params['image_diff_pred'][1].diff - net.params['bilinear_fc_u'][0].diff)

if __name__ == "__main__":
    main()
