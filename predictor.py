from __future__ import division

import os
import numpy as np
import argparse
import h5py
import cv2
import caffe
from caffe.proto import caffe_pb2 as pb2
import net
import bilinear

class FeaturePredictor(object):
    """
    Predicts change in features (y_dot) given the current input (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape, y_shape=None):
        self.x_shape = x_shape
        self.u_shape = u_shape
        if y_shape is None:
            y_shape = self.feature_from_input(np.empty(x_shape)).shape
        self.y_shape = y_shape

    def train(self, data):
        raise NotImplementedError

    def predict(self, X, U):
        raise NotImplementedError

    def jacobian_control(self, X, U):
        raise NotImplementedError

    def feature_from_input(self, X):
        """
        By default, the feature is just the input flattened
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
    def __init__(self, model_file, pretrained_file=None, prediction_name=None):
        if pretrained_file is None:
            caffe.Net.__init__(self, model_file, caffe.TEST)
        else:
            caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        self.prediction_name = prediction_name or self.outputs[0]
        self.prediction_dim = self.blobs[self.prediction_name].shape[1]

    def predict(self, *inputs):
        batch = len(self.blob(self.inputs[0]).data.shape) == len(inputs[0].shape)
        if batch:
            batch_size = len(inputs[0])
            for input_ in inputs[1:]:
                if input_ is None:
                    continue
                assert batch_size == len(input_)
        else:
            batch_size = 1
            inputs = list(inputs)
            for i, input_ in enumerate(inputs):
                if input_ is None:
                    continue
                inputs[i] = input_[None, :]
            inputs = tuple(inputs)
        out = self.forward_all(**dict(zip(self.inputs, inputs)))
        predictions = out[self.prediction_name]
        if batch:
            return predictions
        else:
            return np.squeeze(predictions, axis=0)

    def jacobian(self, wrt_input_name, *inputs):
        assert wrt_input_name in self.inputs
        batch = len(self.blob(self.inputs[0]).data.shape) == len(inputs[0].shape)
        wrt_input_shape = self.blob(wrt_input_name).data.shape
        if batch:
            batch_size = len(inputs[0])
            for input_ in inputs[1:]:
                if input_ is None:
                    continue
                assert batch_size == len(input_)
        else:
            batch_size = 1
            inputs = list(inputs)
            for i, input_ in enumerate(inputs):
                if input_ is None:
                    continue
                inputs[i] = input_[None, :]
            inputs = tuple(inputs)
        _, wrt_input_dim = wrt_input_shape
        inputs = list(inputs)
        # use inputs with zeros for the inputs that are not specified
        for i, (input_name, input_) in enumerate(zip(self.inputs, inputs)):
            if input_ is None:
                inputs[i] = np.zeros(self.blob(input_name).shape)
        other_outputs = [] # all outputs but the prediction one
        for output_name in self.outputs:
            if output_name == self.prediction_name:
                continue
            other_outputs.append((output_name, np.zeros(self.blob(output_name).shape)))
        jacs = np.empty((batch_size, self.prediction_dim, wrt_input_dim))
        for k, single_inputs in enumerate(zip(*inputs)):
            self.forward(**dict(zip(self.inputs, [input_[None, :] for input_ in single_inputs])))
            for i, e in enumerate(np.eye(self.prediction_dim)):
                diff = self.backward(**dict([(self.prediction_name, e[None, :])] + other_outputs))
                jacs[k, i:i+1, :] = diff[wrt_input_name].copy()
        if batch:
            return jacs
        else:
            return np.squeeze(jacs, axis=0)

    def blob(self, blob_name):
        return self._blobs[list(self._blob_names).index(blob_name)]


class BilinearFeaturePredictor(bilinear.BilinearFunction, FeaturePredictor):
    """
    Predicts change in features (y_dot) given the current input image (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape, y_shape=None):
        FeaturePredictor.__init__(self, x_shape, u_shape, y_shape)
        y_dim = np.prod(self.x_shape)
        assert len(self.u_shape) == 1 or (len(self.u_shape) == 2 and self.u_shape[1] == 1)
        u_dim = self.u_shape[0]
        assert len(self.y_shape) == 1 or (len(self.y_shape) == 2 and self.y_shape[1] == 1)
        assert y_dim == self.y_shape[0]
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
    def __init__(self, net_func, inputs, input_shapes, outputs, pretrained_file=None, postfix=''):
        """
        Assumes that outputs[0] is the prediction_name
        """
        self.net_func = net_func
        self.postfix = postfix

        self.deploy_net_param = net_func(input_shapes)
        self.deploy_net_param = net.deploy_net(self.deploy_net_param, inputs, input_shapes, outputs)
        self.net_name = str(self.deploy_net_param.name)

        deploy_fname = self.get_model_fname('deploy')
        with open(deploy_fname, 'w') as f:
            f.write(str(self.deploy_net_param))

        NetPredictor.__init__(self, deploy_fname, pretrained_file=pretrained_file, prediction_name=outputs[0])

        self.add_blur_weights(self.params) # TODO: better way to do this?

        x_shape, u_shape = input_shapes
        _, y_dim = self.blobs[self.prediction_name].shape
        y_shape = (y_dim,)
        FeaturePredictor.__init__(self, x_shape, u_shape, y_shape)

    def train(self, train_hdf5_fname, val_hdf5_fname=None, solverstate_fname=None, solver_param=None, batch_size=100):
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
            train_val_net = net.train_val_net(train_val_net)
            train_val_nets.append(train_val_net)
        self.train_net_param, self.val_net_param = train_val_nets

        train_fname = self.get_model_fname('train')
        with open(train_fname, 'w') as f:
            f.write(str(self.train_net_param))
        val_fname = self.get_model_fname('val')
        with open(val_fname, 'w') as f:
            f.write(str(self.val_net_param))

        if solver_param is None:
            solver_param = pb2.SolverParameter()
        self.add_default_parameters(solver_param)

        solver_fname = self.get_model_fname('solver')
        with open(solver_fname, 'w') as f:
            f.write(str(solver_param))

        solver = caffe.get_solver(solver_fname)
        self.add_blur_weights(solver.net.params)
        for param_name, param in self.params.items():
            for blob, solver_blob in zip(param, solver.net.params[param_name]):
                solver_blob.data[...] = blob.data
        if solverstate_fname is not None:
            solver.restore(solverstate_fname)
        solver.solve()
        for param_name, param in self.params.items():
            for blob, solver_blob in zip(param, solver.net.params[param_name]):
                blob.data[...] = solver_blob.data

    def jacobian_control(self, X, U):
        return self.jacobian(self.inputs[1], X, U)

    def feature_from_input(self, X, input_name='image_curr', output_name='y'):
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        batch = X.shape != self.x_shape
        if not batch:
            X = X[None, :]
        kwargs = dict()
        for input_ in self.inputs:
            if input_ == input_name:
                kwargs[input_] = X
            else:
                kwargs[input_] = np.zeros_like(self.blobs[input_].data)
        out = self.forward(blobs=[output_name], **kwargs)
        Y = out[output_name].copy()
        if not batch:
            Y = np.squeeze(Y, axis=0)
        return Y

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

    @staticmethod
    def add_blur_weights(params):
        for param_name, param in params.items():
            if 'blur' in param_name:
                assert len(param) == 1
                blob = param[0]
                assert blob.data.shape[-1] == blob.data.shape[-2]
                kernel_size = blob.data.shape[-1]
                kernel = cv2.getGaussianKernel(kernel_size, -1)
                blob.data[...] = kernel.dot(kernel.T)

    def get_model_dir(self):
        model_dir = os.path.join('models', self.net_name + self.postfix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def get_model_fname(self, phase):
        model_dir = self.get_model_dir()
        fname = os.path.join(model_dir, phase + '.prototxt')
        return fname

    def get_snapshot_fname(self):
        snapshot_dir = os.path.join(self.get_model_dir(), 'snapshot')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        snapshot_prefix = os.path.join(snapshot_dir, '')
        return snapshot_prefix

    @staticmethod
    def infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint):
        input_shapes = []
        if hdf5_fname_hint is None:
            input_shapes = list(default_input_shapes)
        else:
            with h5py.File(hdf5_fname_hint, 'r+') as f:
                for input_ in inputs:
                    input_shape = f[input_].shape[1:]
                    input_shapes.append(input_shape)
        return input_shapes


class BilinearNetFeaturePredictor(NetFeaturePredictor):
    def __init__(self, hdf5_fname_hint=None, pretrained_file=None, postfix=''):
        inputs = ['image_curr', 'vel']
        default_input_shapes = [(1,7,10), (2,)]
        input_shapes = self.infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint)
        outputs = ['y_diff_pred', 'y']
        super(BilinearNetFeaturePredictor, self).__init__(net.bilinear_net,
                                                          inputs, input_shapes, outputs,
                                                          pretrained_file=pretrained_file,
                                                          postfix=postfix)

    def jacobian_control(self, X, U):
        if X.shape == self.x_shape:
            y = self.feature_from_input(X)
            y_dim, = y.shape
            A = self.params.values()[0][0].data.reshape((y_dim, y_dim, -1))
            B = self.params.values()[1][0].data
            jac = np.einsum("kij,i->kj", A, y) + B
            return jac
        else:
            return np.asarray([self.jacobian_control(x, None) for x in X])


class ApproxBilinearNetFeaturePredictor(NetFeaturePredictor):
    def __init__(self, hdf5_fname_hint=None, pretrained_file=None, postfix=''):
        inputs = ['image_curr', 'vel']
        default_input_shapes = [(1,7,10), (2,)]
        input_shapes = self.infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint)
        outputs = ['y_diff_pred', 'y']
        super(ApproxBilinearNetFeaturePredictor, self).__init__(net.approx_bilinear_net,
                                                                inputs, input_shapes, outputs,
                                                                pretrained_file=pretrained_file,
                                                                postfix=postfix)

class ActionCondEncoderNetFeaturePredictor(NetFeaturePredictor):
    def __init__(self, hdf5_fname_hint=None, pretrained_file=None, postfix=''):
        inputs = ['image_curr', 'vel']
        default_input_shapes = [(1,7,10), (2,)]
        input_shapes = self.infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint)
        outputs = ['y_diff_pred', 'y']
        super(ActionCondEncoderNetFeaturePredictor, self).__init__(net.action_cond_encoder_net,
                                                                   inputs, input_shapes, outputs,
                                                                   pretrained_file=pretrained_file,
                                                                   postfix=postfix)


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
    # set parameters of bn to the ones of b and check that their methods return the same outputs
    print "bn validation error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "abn validation error", (np.linalg.norm(Y_dot - predictor_abn.predict(X, U))**2) / (2*N)
    print "b validation error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)

    predictor_bn.params['fc1'][0].data[...] = predictor_b.A.reshape((predictor_b.A.shape[0], -1))
    predictor_bn.params['fc2'][0].data[...] = predictor_b.B
    Y_dot_bn = predictor_bn.predict(X, U)
    Y_dot_b = predictor_b.predict(X, U)
    print "all close Y_dot_bn, Y_dot_b", np.allclose(Y_dot_bn, Y_dot_b, atol=1e-4)
    jac_bn = predictor_bn.jacobian_control(X, U)
    jac_b = predictor_b.jacobian_control(X, U)
    print "all close jac_bn, jac_b", np.allclose(jac_bn, jac_b)

if __name__ == "__main__":
    main()
