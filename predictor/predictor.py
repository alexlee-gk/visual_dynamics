from __future__ import division

import numpy as np
import h5py
import os
import bilinear

class FeaturePredictor(object):
    """
    Predicts change in features (y_dot) given the current input (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape, input_names=None, output_names=None, net_name=None, postfix=None):
        self.input_names = input_names or ['image_curr', 'vel']
        self.output_names = output_names or ['y_diff_pred', 'y', 'y_next_pred', 'image_next_pred', 'x0_next_pred'] # ok if some of these don't exist
        self.x_shape = x_shape
        self.u_shape = u_shape
        self.net_name = net_name
        self.postfix = postfix

    def train(self, data):
        raise NotImplementedError

    def predict(self, X, U):
        raise NotImplementedError

    def jacobian_control(self, X, U):
        raise NotImplementedError
    
    def feature_from_input(self, X):
        raise NotImplementedError

    def preprocess_input(self, X):
        return X
    
    def copy_from(self, params_fname):
        raise NotImplementedError

    def get_model_dir(self):
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def get_snapshot_prefix(self):
        snapshot_dir = os.path.join(self.get_model_dir(), 'snapshot')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        snapshot_prefix = os.path.join(snapshot_dir, '')
        return snapshot_prefix
    
    @staticmethod
    def infer_input_shapes(hdf5_fname_hint, inputs=None):
        inputs = inputs or ['image_curr', 'vel']
        input_shapes = []
        with h5py.File(hdf5_fname_hint, 'r+') as f:
            for input_ in inputs:
                input_shape = f[input_].shape[1:]
                input_shapes.append(input_shape)
        return input_shapes


class BilinearFeaturePredictor(bilinear.BilinearFunction, FeaturePredictor):
    """
    Predicts change in features (y_dot) given the current input image (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape):
        FeaturePredictor.__init__(self, x_shape, u_shape)
        y_dim = np.prod(self.x_shape)
        assert len(self.u_shape) == 1 or (len(self.u_shape) == 2 and self.u_shape[1] == 1)
        u_dim = self.u_shape[0]
        bilinear.BilinearFunction.__init__(self, y_dim, u_dim)

    def train(self, X, U, Y_dot):
        Y = self.flatten(X)
        self.fit(Y, U, Y_dot)

    def predict(self, X, U):
        Y = self.flatten(X)
        return self.eval(Y, U)

    def jacobian_control(self, X, U):
        Y = self.flatten(X)
        # in the bilinear model, the jacobian doesn't depend on u
        return self.jac_u(Y)
    
    def flatten(self, X):
        is_batched = X.shape[1:] == self.x_shape
        if is_batched:
            Y = X.reshape((X.shape[0], -1))
        else:
            Y = X.flatten()
        return Y
        

def main():
    import argparse
    from caffe.proto import caffe_pb2 as pb2
    import theano
    import cgt
    import predictor_caffe
    import predictor_theano
    import net_theano
    import net_cgt
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('val_hdf5_fname', nargs='?', type=str, default=None)

    args = parser.parse_args()

    if args.val_hdf5_fname is None:
        val_file = h5py.File(args.train_hdf5_fname.replace('train', 'val'), 'r+')
    else:
        val_file = h5py.File(args.val_hdf5_fname, 'r+')

    input_shapes = FeaturePredictor.infer_input_shapes(args.train_hdf5_fname)
    
    predictor_bn = predictor_caffe.BilinearNetFeaturePredictor(input_shapes)
    predictor_b = BilinearFeaturePredictor(*input_shapes)
    predictor_tbn = predictor_theano.TheanoNetFeaturePredictor(*net_theano.build_bilinear_net(input_shapes))
    predictor_cbn = predictor_theano.CgtNetFeaturePredictor(*net_cgt.build_bilinear_net(input_shapes))

    # train
    train_file = h5py.File(args.train_hdf5_fname, 'r+')
    X = train_file['image_curr'][:]
    U = train_file['vel'][:]
    X_dot = train_file['image_diff'][:]
    Y_dot = predictor_b.flatten(X_dot)
    N = len(X)
    predictor_bn.train(args.train_hdf5_fname, args.val_hdf5_fname, solver_param=pb2.SolverParameter(max_iter=100))
    predictor_tbn.train(args.train_hdf5_fname, args.val_hdf5_fname, max_iter=100)
    predictor_cbn.train(args.train_hdf5_fname, args.val_hdf5_fname, max_iter=100)
    predictor_b.train(X, U, Y_dot)
    print "bn train error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "tbn train error", (np.linalg.norm(Y_dot - predictor_tbn.predict(X, U))**2) / (2*N)
    print "cbn train error", (np.linalg.norm(Y_dot - predictor_cbn.predict(X, U))**2) / (2*N)
    print "b train error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)

    # validation
    X = val_file['image_curr'][:]
    U = val_file['vel'][:]
    X_dot = val_file['image_diff'][:]
    Y_dot = predictor_b.flatten(X_dot)
    N = len(X)
    # set parameters of bn to the ones of b and check that their methods return the same outputs
    print "bn validation error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N)
    print "tbn validation error", (np.linalg.norm(Y_dot - predictor_tbn.predict(X, U))**2) / (2*N)
    print "cbn validation error", (np.linalg.norm(Y_dot - predictor_cbn.predict(X, U))**2) / (2*N)
    print "b validation error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N)
    predictor_bn.params['bilinear_fc_outer_yu'][0].data[...] = predictor_b.A.reshape((predictor_b.A.shape[0], -1))
    predictor_bn.params['bilinear_fc_u'][0].data[...] = predictor_b.B
    predictor_tbn_params = {param.name: param for param in predictor_tbn.get_all_params()}
    predictor_tbn_params['M'].set_value(predictor_b.A.astype(theano.config.floatX))
    predictor_tbn_params['N'].set_value(predictor_b.B.astype(theano.config.floatX))
    predictor_cbn_params = {param.name: param for param in predictor_cbn.get_all_params()}
    predictor_cbn_params['bilinear.M'].op.set_value(predictor_b.A.astype(cgt.floatX))
    predictor_cbn_params['bilinear.N'].op.set_value(predictor_b.B.astype(cgt.floatX))

    Y_dot_bn = predictor_bn.predict(X, U)
    Y_dot_tbn = predictor_tbn.predict(X, U)
    Y_dot_b = predictor_b.predict(X, U)
    print "all close Y_dot_bn, Y_dot_b", np.allclose(Y_dot_bn, Y_dot_b, atol=1e-4)
    print "all close Y_dot_tbn, Y_dot_b", np.allclose(Y_dot_tbn, Y_dot_b, atol=1e-4)
    jac_bn = predictor_bn.jacobian_control(X, U)
    jac_tbn = predictor_tbn.jacobian_control(X, U)
    jac_b = predictor_b.jacobian_control(X, U)
    print "all close jac_bn, jac_b", np.allclose(jac_bn, jac_b, atol=1e-3)
    print "all close jac_tbn, jac_b", np.allclose(jac_tbn, jac_b, atol=1e-3)

if __name__ == "__main__":
    main()
