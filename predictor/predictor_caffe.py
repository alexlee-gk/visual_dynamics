from __future__ import division

import os
import re
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2 as pb2
import net_caffe
import predictor


class CaffeNetPredictor(caffe.Net):
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


class CaffeNetFeaturePredictor(CaffeNetPredictor, predictor.FeaturePredictor):
    """
    Predicts change in features (y_dot) given the current input image (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, net_func, input_shapes, input_names=None, output_names=None, pretrained_file=None, postfix='', batch_size=32):
        """
        Assumes that outputs[0] is the prediction_name
        batch_size_1: if True, another net_caffe of batch_size of 1 is created, and this net_caffe is used for computing forward in the predict method
        """
        predictor.FeaturePredictor.__init__(self, *input_shapes, input_names=input_names, output_names=output_names)
        self.net_func = net_func
        self.postfix = postfix
        self.batch_size = batch_size

        self.deploy_net_param = net_func(input_shapes, batch_size=batch_size)
        self.deploy_net_param = net_caffe.deploy_net(self.deploy_net_param, self.input_names, input_shapes, self.output_names, batch_size=batch_size)
        self.net_name = str(self.deploy_net_param.name)
        deploy_fname = self.get_model_fname('deploy')
        with open(deploy_fname, 'w') as f:
            f.write(str(self.deploy_net_param))

        if pretrained_file is not None and not pretrained_file.endswith('.caffemodel'):
            pretrained_file = self.get_snapshot_prefix() + '_iter_' + pretrained_file + '.caffemodel'
        CaffeNetPredictor.__init__(self, deploy_fname, pretrained_file=pretrained_file, prediction_name=self.output_names[0])
        self.output_names = [name for name in self.output_names if name in self.blobs]

        self.add_blur_weights(self.params) # TODO: better way to do this?

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
            # add data layers from validation net_caffe
            self.train_val_net_param.layer.extend([layer for layer in val_net_param.layer if 'Data' in layer.type])
            # add back the layers that are not data layers
            self.train_val_net_param.layer.extend([layer for layer in layers if 'Data' not in layer.type])
        self.train_val_net_param = net_caffe.train_val_net(self.train_val_net_param)
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

    def predict(self, *inputs, **kwargs):
        if 'prediction_name' in kwargs and kwargs['prediction_name'] not in self.blobs:
            kwargs['prediction_name'] = kwargs['prediction_name'].replace('image', 'x0')
        return super(CaffeNetFeaturePredictor, self).predict(*inputs, **kwargs)

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

    def preprocess_input(self, X):
        if 'x0' in self.blobs:
            return self.feature_from_input(X, output_name='x0')
        else:
            return X

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
        model_dir = predictor.FeaturePredictor.get_model_dir(self)
        model_dir = os.path.join(model_dir, 'caffe', self.net_name + self.postfix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def get_model_fname(self, phase):
        model_dir = self.get_model_dir()
        fname = os.path.join(model_dir, phase + '.prototxt')
        return fname


class BilinearNetFeaturePredictor(CaffeNetFeaturePredictor):
    def __init__(self, input_shapes, **kwargs):
        super(BilinearNetFeaturePredictor, self).__init__(net_caffe.bilinear_net, **kwargs)

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

class FcnActionCondEncoderNetFeaturePredictor(CaffeNetFeaturePredictor):
    def jacobian_control(self, X, U):
        if X.shape == self.x_shape:
            levels = []
            for key in self.blobs.keys():
                match = re.match('bilinear(\d+)_re_y$', key)
                if match:
                    assert len(match.groups()) == 1
                    levels.append(int(match.group(1)))
            levels = sorted(levels)

            ylevels = []
            jaclevels = []
            for level in levels:
                output_name = 'x%d'%level
                if output_name == 'x0' and output_name not in self.blobs:
                    xlevel = X
                else:
                    xlevel = self.feature_from_input(X, output_name=output_name)
                ylevels.append(xlevel.flatten())
                xlevel_c_dim = xlevel.shape[0]
                y_dim = np.prod(xlevel.shape[1:])
                u_dim, = self.u_shape
                A = self.params['bilinear%d_bilinear_yu'%level][0].data.reshape((y_dim, u_dim, y_dim))
                B = self.params['bilinear%d_linear_u'%level][0].data
                c = self.params['bilinear%d_linear_u'%level][1].data
                jaclevel = (np.einsum("kji,ci->ckj", A, xlevel.reshape((xlevel_c_dim, y_dim))) + B + c[:, None]).reshape(xlevel_c_dim * y_dim, u_dim)
                jaclevels.append(jaclevel)
            y = np.concatenate(ylevels)
            assert np.allclose(y, self.feature_from_input(X)) # make sure the order of features in the hierarchy are consistent
            jac = np.concatenate(jaclevels)
            return jac
        else:
            return np.asarray([self.jacobian_control(x, None) for x in X])
