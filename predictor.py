from __future__ import division

import os
import numpy as np
import argparse
import h5py
import cv2
import lasagne
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
        self.prediction_dim = self.blob(self.prediction_name).shape[1]

    def predict(self, *inputs, **kwargs):
        batch = self.blob(self.inputs[0]).data.ndim == inputs[0].ndim
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
        prediction_name = kwargs.get('prediction_name') or self.prediction_name
        if self.batch_size != 1 and batch_size == 1:
            for input_ in self.inputs:
                blob = self.blob(input_)
                blob.reshape(1, *blob.shape[1:])
            self.reshape()
        outs = self.forward_all(blobs=[prediction_name], end=prediction_name, **dict(zip(self.inputs, inputs)))
        if self.batch_size != 1 and batch_size == 1:
            for input_ in self.inputs:
                blob = self.blob(input_)
                blob.reshape(self.batch_size, *blob.shape[1:])
            self.reshape()
        predictions = outs[prediction_name]
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
        # use outputs with zeros for the outpus that doesn't affect the backward computation
        output_diffs = dict()
        for output_name in self.outputs:
            if output_name == self.prediction_name:
                output_diffs[output_name] = np.eye(self.prediction_dim)
            else:
                output_diffs[output_name] = np.zeros((self.prediction_dim,) + self.blob(output_name).diff.shape[1:])
        jacs = np.empty((batch_size, self.prediction_dim, wrt_input_dim))
        for k, input_ in enumerate(zip(*inputs)):
            input_blobs = dict(zip(self.inputs, [np.repeat(in_[None, :], self.batch_size, axis=0) for in_ in input_]))
            self.forward_all(blobs=[self.prediction_name], end=self.prediction_name, **input_blobs)
            diffs = self.backward_all(diffs=[self.prediction_name], start=self.prediction_name, **output_diffs)
            jacs[k, :, :] = diffs[wrt_input_name]
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
    def __init__(self, net_func, inputs, input_shapes, outputs, pretrained_file=None, postfix='', batch_size=32):
        """
        Assumes that outputs[0] is the prediction_name
        batch_size_1: if True, another net of batch_size of 1 is created, and this net is used for computing forward in the predict method
        """
        self.net_func = net_func
        self.postfix = postfix
        self.batch_size = batch_size

        self.deploy_net_param = net_func(input_shapes, batch_size=batch_size)
        self.deploy_net_param = net.deploy_net(self.deploy_net_param, inputs, input_shapes, outputs, batch_size=batch_size)
        self.net_name = str(self.deploy_net_param.name)
        deploy_fname = self.get_model_fname('deploy')
        with open(deploy_fname, 'w') as f:
            f.write(str(self.deploy_net_param))

        if pretrained_file is not None and not pretrained_file.endswith('.caffemodel'):
            pretrained_file = self.get_snapshot_prefix() + '_iter_' + pretrained_file + '.caffemodel'
        NetPredictor.__init__(self, deploy_fname, pretrained_file=pretrained_file, prediction_name=outputs[0])

        self.add_blur_weights(self.params) # TODO: better way to do this?

        x_shape, u_shape = input_shapes
        _, y_dim = self.blob(self.prediction_name).shape
        y_shape = (y_dim,)
        FeaturePredictor.__init__(self, x_shape, u_shape, y_shape)

        self.train_net = None
        self.val_net = None

    def train(self, train_hdf5_fname, val_hdf5_fname=None, solverstate_fname=None, solver_param=None, batch_size=32):
        hdf5_txt_fnames = []
        for hdf5_fname in [train_hdf5_fname, val_hdf5_fname]:
            if hdf5_fname is not None:
                head, tail = os.path.split(hdf5_fname)
                root, _ = os.path.splitext(tail)
                hdf5_txt_fname = os.path.join(head, '.' + root + '.txt')
                if not os.path.isfile(hdf5_txt_fname):
                    with open(hdf5_txt_fname, 'w') as f:
                        f.write(hdf5_fname + '\n')
                hdf5_txt_fnames.append(hdf5_txt_fname)
            else:
                hdf5_txt_fnames.append(None)
        train_hdf5_txt_fname, val_hdf5_txt_fname = hdf5_txt_fnames

        input_shapes = (self.x_shape, self.u_shape)
        train_net_param = self.net_func(input_shapes, train_hdf5_txt_fname, batch_size, self.net_name, phase=caffe.TRAIN)
        if val_hdf5_fname is not None:
            val_net_param = self.net_func(input_shapes, val_hdf5_txt_fname, batch_size, self.net_name, phase=caffe.TEST)

        self.train_val_net_param = train_net_param
        if val_hdf5_fname is not None:
            layers = [layer for layer in self.train_val_net_param.layer]
            # remove layers except for data layers
            for layer in layers:
                if 'Data' not in layer.type:
                    self.train_val_net_param.layer.remove(layer)
            # add data layers from validation net
            self.train_val_net_param.layer.extend([layer for layer in val_net_param.layer if 'Data' in layer.type])
            # add back the layers that are not data layers
            self.train_val_net_param.layer.extend([layer for layer in layers if 'Data' not in layer.type])
        self.train_val_net_param = net.train_val_net(self.train_val_net_param)
        train_val_fname = self.get_model_fname('train_val')
        with open(train_val_fname, 'w') as f:
            f.write(str(self.train_val_net_param))

        if solver_param is None:
            solver_param = pb2.SolverParameter()
        self.add_default_parameters(solver_param, val_net=val_hdf5_fname is not None)

        solver_fname = self.get_model_fname('solver')
        with open(solver_fname, 'w') as f:
            f.write(str(solver_param))

        solver = caffe.get_solver(solver_fname)
        self.add_blur_weights(solver.net.params)
        for param_name, param in self.params.items():
            for blob, solver_blob in zip(param, solver.net.params[param_name]):
                solver_blob.data[...] = blob.data
        if solverstate_fname is not None:
            if not solverstate_fname.endswith('.solverstate'):
                solverstate_fname = self.get_snapshot_prefix() + '_iter_' + solverstate_fname + '.solverstate'
            solver.restore(solverstate_fname)
        solver.solve()
        for param_name, param in self.params.items():
            for blob, solver_blob in zip(param, solver.net.params[param_name]):
                blob.data[...] = solver_blob.data

        self.train_net = solver.net
        if val_hdf5_fname is not None:
            self.val_net = solver.test_nets[0]

    def jacobian_control(self, X, U):
        return self.jacobian(self.inputs[1], X, U)

    def feature_from_input(self, X, input_name='image_curr', output_name='y'):
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        batch = X.shape != self.x_shape
        if not batch:
            X = X[None, :]
        batch_size = len(X)
        input_blobs = dict()
        for input_ in self.inputs:
            if input_ == input_name:
                input_blobs[input_] = X
            else:
                input_blobs[input_] = np.zeros((batch_size,) + self.blob(input_).data.shape[1:])
        outs = self.forward_all(blobs=[output_name], end=output_name, **input_blobs)
        Y = outs[output_name]
        if not batch:
            Y = np.squeeze(Y, axis=0)
        return Y

    def add_default_parameters(self, solver_param, val_net=True):
        if not solver_param.train_net:
            train_val_fname = self.get_model_fname('train_val')
            solver_param.train_net = train_val_fname
        if val_net:
            if not solver_param.test_net:
                train_val_fname = self.get_model_fname('train_val')
                solver_param.test_net.append(train_val_fname)
            if not solver_param.test_iter:
                solver_param.test_iter.append(10)
        else:
            del solver_param.test_net[:]
            del solver_param.test_iter[:]
        if not solver_param.solver_type:   solver_param.solver_type = pb2.SolverParameter.SGD
        if not solver_param.test_interval: solver_param.test_interval = 1000
        if not solver_param.base_lr:       solver_param.base_lr = 0.05
        if not solver_param.lr_policy:     solver_param.lr_policy = "step"
        if not solver_param.gamma:         solver_param.gamma = 0.9
        if not solver_param.stepsize:      solver_param.stepsize = 1000
        if not solver_param.display:       solver_param.display = 20
        if not solver_param.max_iter:      solver_param.max_iter = 10000
        if not solver_param.momentum:      solver_param.momentum = 0.9
        if not solver_param.momentum2:      solver_param.momentum2 = 0.999
        if not solver_param.weight_decay:  solver_param.weight_decay = 0.0005
        if not solver_param.snapshot:      solver_param.snapshot = 1000
        if not solver_param.snapshot_prefix:
            snapshot_prefix = self.get_snapshot_prefix()
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

    def get_snapshot_prefix(self):
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
        outputs = ['y_diff_pred', 'y', 'image_next_pred']
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

class BilinearConstrainedNetFeaturePredictor(BilinearNetFeaturePredictor):
    def __init__(self, hdf5_fname_hint=None, pretrained_file=None, postfix=''):
        inputs = ['image_curr', 'vel']
        default_input_shapes = [(1,7,10), (2,)]
        input_shapes = self.infer_input_shapes(inputs, default_input_shapes, hdf5_fname_hint)
        outputs = ['y_diff_pred', 'y', 'image_next_pred']
        NetFeaturePredictor.__init__(self,
                                     net.bilinear_constrained_net,
                                     inputs, input_shapes, outputs,
                                     pretrained_file=pretrained_file,
                                     postfix=postfix)


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

    if args.val_hdf5_fname is None:
        val_file = h5py.File(args.train_hdf5_fname.replace('train', 'val'), 'r+')
    else:
        val_file = h5py.File(args.val_hdf5_fname, 'r+')

    inputs = ['image_curr', 'vel']
    input_shapes = NetFeaturePredictor.infer_input_shapes(inputs, None, args.train_hdf5_fname)
    x_shape, u_shape = input_shapes

    predictor_bn = BilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname)
    predictor_abn = ApproxBilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname)
    predictor_b = BilinearFeaturePredictor(x_shape, u_shape, None)
    import predictor_theano
    predictor_tbn = predictor_theano.TheanoNetFeaturePredictor(*predictor_theano.build_bilinear_net(input_shapes))

    # train
    train_file = h5py.File(args.train_hdf5_fname, 'r+')
    X = train_file['image_curr'][:]
    U = train_file['vel'][:]
    X_dot = train_file['image_diff'][:]
    Y_dot = predictor_b.feature_from_input(X_dot)
    N = len(X)
    predictor_bn.train(args.train_hdf5_fname, args.val_hdf5_fname)
    predictor_abn.train(args.train_hdf5_fname, args.val_hdf5_fname)
    predictor_tbn.train(args.train_hdf5_fname, args.val_hdf5_fname)
    predictor_b.train(X, U, Y_dot)
    print "bn train error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "abn train error", (np.linalg.norm(Y_dot - predictor_abn.predict(X, U))**2) / (2*N)
    print "tbn train error", (np.linalg.norm(Y_dot - predictor_tbn.predict(X, U))**2) / (2*N)
    print "b train error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)

    # validation
    X = val_file['image_curr'][:]
    U = val_file['vel'][:]
    X_dot = val_file['image_diff'][:]
    Y_dot = predictor_b.feature_from_input(X_dot)
    N = len(X)
    # set parameters of bn to the ones of b and check that their methods return the same outputs
    print "bn validation error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "abn validation error", (np.linalg.norm(Y_dot - predictor_abn.predict(X, U))**2) / (2*N)
    print "tbn validation error", (np.linalg.norm(Y_dot - predictor_tbn.predict(X, U))**2) / (2*N)
    print "b validation error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)
    predictor_bn.params['bilinear_fc_outer_yu'][0].data[...] = predictor_b.A.reshape((predictor_b.A.shape[0], -1))
    predictor_bn.params['bilinear_fc_u'][0].data[...] = predictor_b.B
    predictor_tbn_params = {param.name: param for param in lasagne.layers.get_all_params(predictor_tbn.l_x_next_pred)}
    predictor_tbn_params['M'].set_value(predictor_b.A)
    predictor_tbn_params['N'].set_value(predictor_b.B)

    Y_dot_bn = predictor_bn.predict(X, U)
    Y_dot_tbn = predictor_tbn.predict(X, U)
    Y_dot_b = predictor_b.predict(X, U)
    print "all close Y_dot_bn, Y_dot_b", np.allclose(Y_dot_bn, Y_dot_b, atol=1e-4)
    print "all close Y_dot_tbn, Y_dot_b", np.allclose(Y_dot_tbn, Y_dot_b)
    jac_bn = predictor_bn.jacobian_control(X, U)
    jac_tbn = predictor_tbn.jacobian_control(X, U)
    jac_b = predictor_b.jacobian_control(X, U)
    print "all close jac_bn, jac_b", np.allclose(jac_bn, jac_b)
    print "all close jac_tbn, jac_b", np.allclose(jac_tbn, jac_b)

if __name__ == "__main__":
    main()
