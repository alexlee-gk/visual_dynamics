from __future__ import division

import os
import numpy as np
import argparse
import h5py
import caffe
from caffe.proto import caffe_pb2 as pb2
import net
import bilinear

class FeaturePredictor(object):
    """
    Predicts change in features (y_dot) given the current input (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape, y_shape):
        self.x_shape = x_shape
        self.u_shape = u_shape
        self.y_shape = y_shape

    def train(self, data):
        raise NotImplementedError

    def predict(self, X, U):
        raise NotImplementedError

    def jacobian_control(self, X, U):
        raise NotImplementedError

    def feature_from_input(self, X):
        """
        By default, the feture is just the input flattened
        """
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        if X.shape == self.x_shape:
            Y = X.flatten()
        else:
            Y = X.reshape((X.shape[0], -1))
        return Y


class NetPredictor(caffe.Net):
    """
    Predicts output given the current inputs
        inputs -> prediction
    """
    def __init__(self, model_file, pretrained_file=None):
        if pretrained_file is None:
            caffe.Net.__init__(self, model_file, caffe.TEST)
        else:
            caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        assert len(self.outputs) == 1

    def predict(self, *inputs):
        out = self.forward_all(**dict(zip(self.inputs, inputs)))
        prediction_name = self.outputs[0]
        predictions = out[prediction_name]
        return predictions

    def jacobian(self, wrt_input_name, *inputs):
        assert wrt_input_name in self.inputs
        assert self.blobs[wrt_input_name].data.ndim == 2
        batch = len(self.blobs[self.inputs[0]].data.shape) == len(inputs[0].shape)
        if batch:
            batch_size = len(inputs[0])
            for input_ in inputs[1:]:
                assert batch_size == len(input_)
        else:
            batch_size = 1
            inputs = list(inputs)
            for i, input_ in enumerate(inputs):
                inputs[i] = np.expand_dims(input_, axis=0)
            inputs = tuple(inputs)
        prediction_name = self.outputs[0]
        prediction_dim = self.blobs[prediction_name].shape[1]
        jacs = []
        for e in np.eye(prediction_dim):
            _, diffs = self.forward_backward_all(**dict(zip(self.inputs + [prediction_name],
                                                            inputs + (np.tile(e, (batch_size, 1)),))))
            jacs.append(diffs[wrt_input_name])
        jacs = np.asarray(jacs)
        jacs = jacs.swapaxes(0, 1)
        if batch:
            return jacs
        else:
            return np.squeeze(jacs, axis=0)


class BilinearFeaturePredictor(bilinear.BilinearFunction, FeaturePredictor):
    """
    Predicts change in features (y_dot) given the current input image (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape, y_shape):
        FeaturePredictor.__init__(self, x_shape, u_shape, y_shape)
        self.x_shape = x_shape
        self.u_shape = u_shape
        self.y_shape = y_shape
        y_dim = np.prod(x_shape)
        assert len(u_shape) == 1 or (len(u_shape) == 2 and u_shape[1] == 1)
        u_dim = u_shape[0]
        assert len(y_shape) == 1 or (len(y_shape) == 2 and y_shape[1] == 1)
        assert y_dim == y_shape[0]
        bilinear.BilinearFunction.__init__(self, y_dim, u_dim)

    def train(self, X, U, Y_dot):
        Y = self.feature_from_input(X)
        self.fit(Y, U, Y_dot)

    def predict(self, X, U):
        Y = self.feature_from_input(X)
        return self.eval(Y, U)

    def jacobian_control(self, X, U):
        Y = self.feature_from_input(X)
        # in the bilinear model, the jacobian doesn't depend on u
        return self.jac_u(Y)


class NetFeaturePredictor(NetPredictor, FeaturePredictor):
    """
    Predicts change in features (y_dot) given the current input image (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, net_func, inputs, input_shapes, output, pretrained_file=None):
        self.net_func = net_func
        self.net_name = self.__class__.__name__.replace('FeaturePredictor', '')

        self.deploy_net = net_func(input_shapes, net_name=self.net_name)
        self.deploy_net = net.deploy_net(self.deploy_net, inputs, input_shapes, [output])

        deploy_fname = self.get_model_fname('deploy')
        with open(deploy_fname, 'w') as f:
            f.write(str(self.deploy_net))

        NetPredictor.__init__(self, deploy_fname, pretrained_file=pretrained_file)

        x_shape, u_shape = input_shapes
        _, y_dim = self.blobs[output].shape
        y_shape = (y_dim,)
        FeaturePredictor.__init__(self, x_shape, u_shape, y_shape)

    def train(self, train_hdf5_fname, val_hdf5_fname=None, solver_param=None, batch_size=100):
        if val_hdf5_fname is None:
            val_hdf5_fname = train_hdf5_fname.replace('train', 'val')

        input_shapes = (self.x_shape, self.u_shape)
        train_val_nets = []
        for hdf5_fname in [train_hdf5_fname, val_hdf5_fname]:
            head, tail = os.path.split(hdf5_fname)
            root, _ = os.path.splitext(tail)
            hdf5_txt_fname = os.path.join(head, '.' + root + '.txt')
            if not os.path.isfile(hdf5_txt_fname):
                with open(hdf5_txt_fname, 'w') as f:
                    f.write(hdf5_fname + '\n')
            train_val_net = self.net_func(input_shapes, hdf5_txt_fname, batch_size, self.net_name)
            train_val_nets.append(train_val_net)
        self.train_net, self.val_net = train_val_nets

        train_fname = self.get_model_fname('train')
        with open(train_fname, 'w') as f:
            f.write(str(self.train_net))
        val_fname = self.get_model_fname('val')
        with open(val_fname, 'w') as f:
            f.write(str(self.val_net))

        if solver_param is None:
            solver_param = pb2.SolverParameter()
        self.add_default_parameters(solver_param)

        solver_fname = self.get_model_fname('solver')
        with open(solver_fname, 'w') as f:
            f.write(str(solver_param))

        solver = caffe.get_solver(solver_fname)
        for param_name, param in self.params.items():
            for blob, solver_blob in zip(param, solver.net.params[param_name]):
                solver_blob.data[...] = blob.data
        solver.solve()

    def jacobian_control(self, X, U):
        return self.jacobian(self.inputs[1], X, U)

    def add_default_parameters(self, solver_param):
        if not solver_param.train_net:
            train_fname = self.get_model_fname('train')
            solver_param.train_net = train_fname
        if not solver_param.test_net:
            val_fname = self.get_model_fname('val')
            solver_param.test_net.append(val_fname)
        if not solver_param.solver_type:   solver_param.solver_type = pb2.SolverParameter.SGD
        if not solver_param.test_iter:     solver_param.test_iter.append(1000)
        if not solver_param.test_interval: solver_param.test_interval = 2500
        if not solver_param.base_lr:       solver_param.base_lr = 0.05
        if not solver_param.lr_policy:     solver_param.lr_policy = "step"
        if not solver_param.gamma:         solver_param.gamma = 0.9
        if not solver_param.stepsize:      solver_param.stepsize = 1000
        if not solver_param.display:       solver_param.display = 20
        if not solver_param.max_iter:      solver_param.max_iter = 10000
        if not solver_param.max_iter:      solver_param.max_iter = 0.9
        if not solver_param.weight_decay:  solver_param.weight_decay = 0.0005
        if not solver_param.snapshot:      solver_param.snapshot = 1000
        if not solver_param.snapshot_prefix:
            snapshot_prefix = self.get_snapshot_fname()
            solver_param.snapshot_prefix = snapshot_prefix
        # don't change solver_param.solver_mode

    def get_model_dir(self):
        model_dir = os.path.join('models', self.net_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def get_model_fname(self, phase):
        model_dir = self.get_model_dir()
        fname = os.path.join(model_dir, phase + '.prototxt')
        return fname

    def get_snapshot_fname(self):
        snapshot_dir = os.path.join('models', self.net_name, 'snapshot')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        snapshot_prefix = os.path.join(snapshot_dir, self.net_name)
        return snapshot_prefix

    def infer_input_shapes(self, inputs, default_input_shapes, hdf5_fname_hint):
        input_shapes = []
        if hdf5_fname_hint is None:
            input_shapes = list(default_input_shapes)
        else:
            with h5py.File(hdf5_fname_hint, 'r+') as f:
                for input_, default_input_shape in zip(inputs, default_input_shapes):
                    input_shape = f[input_].shape[1:]
                    assert len(input_shape) == len(default_input_shape)
                    input_shapes.append(input_shape)
        return input_shapes


class BilinearNetFeaturePredictor(NetFeaturePredictor):
    def __init__(self, hdf5_fname_hint=None, pretrained_file=None):
        inputs = ['image_curr', 'vel']
        default_input_shapes = [(1,7,10), (2,)]
        input_shapes = self.infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint)
        output = 'y_diff_pred'
        super(BilinearNetFeaturePredictor, self).__init__(net.bilinear_net, inputs, input_shapes, output, pretrained_file=pretrained_file)


class ApproxBilinearNetFeaturePredictor(NetFeaturePredictor):
    def __init__(self, hdf5_fname_hint=None, pretrained_file=None):
        inputs = ['image_curr', 'vel']
        default_input_shapes = [(1,7,10), (2,)]
        input_shapes = self.infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint)
        output = 'y_diff_pred'
        super(ApproxBilinearNetFeaturePredictor, self).__init__(net.approx_bilinear_net, inputs, input_shapes, output, pretrained_file=pretrained_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('val_hdf5_fname', nargs='?', type=str, default=None)

    args = parser.parse_args()

    predictor_bn = BilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname)
    predictor_abn = ApproxBilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname)
    predictor_b = BilinearFeaturePredictor(predictor_bn.x_shape,
                                           predictor_bn.u_shape,
                                           predictor_bn.y_shape)

    # train
    train_file = h5py.File(args.train_hdf5_fname, 'r+')
    X = train_file['image_curr'][:]
    U = train_file['vel'][:]
    X_dot = train_file['image_diff'][:]
    Y_dot = predictor_b.feature_from_input(X_dot)
    N = len(X)
    predictor_bn.train(args.train_hdf5_fname, args.val_hdf5_fname)
    predictor_abn.train(args.train_hdf5_fname, args.val_hdf5_fname)
    predictor_b.train(X, U, Y_dot)
    print "bn train error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "abn train error", (np.linalg.norm(Y_dot - predictor_abn.predict(X, U))**2) / (2*N)
    print "b train error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)

    # validation
    if args.val_hdf5_fname is None:
        val_file = h5py.File(args.train_hdf5_fname.replace('train', 'val'), 'r+')
    else:
        val_file = h5py.File(args.val_hdf5_fname, 'r+')
    X = val_file['image_curr'][:]
    U = val_file['vel'][:]
    X_dot = val_file['image_diff'][:]
    Y_dot = predictor_b.feature_from_input(X_dot)
    N = len(X)
    print "bn validation error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "abn validation error", (np.linalg.norm(Y_dot - predictor_abn.predict(X, U))**2) / (2*N)
    print "b validation error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)

    # set parameters of bn to the ones of b and check that their methods return the same outputs
    predictor_bn.params['fc1'][0].data[...] = predictor_b.A.reshape((predictor_b.A.shape[0], -1))
    predictor_bn.params['fc2'][0].data[...] = predictor_b.B
    Y_dot_bn = predictor_bn.predict(X, U)
    Y_dot_b = predictor_b.predict(X, U)
    print "all close Y_dot_bn, Y_dot_b", np.allclose(Y_dot_bn, Y_dot_b, atol=1e-4)
    x = X[0, ...]
    u = U[0, ...]
    jac_bn = predictor_bn.jacobian_control(x, u)
    jac_b = predictor_b.jacobian_control(x, u)
    print "all close jac_bn, jac_b", np.allclose(jac_bn, jac_b)

if __name__ == "__main__":
    main()
