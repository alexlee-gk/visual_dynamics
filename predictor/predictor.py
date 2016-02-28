import collections
import os
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import bilinear
from utils import util


class FeaturePredictor(object):
    """
    Predicts change in features (y_dot) given the current input (x) and control (u):
        x, u -> y_dot
    """
    def __init__(self, x_shape, u_shape, input_names=None, output_names=None, net_name=None, postfix=None, backend=None):
        self.input_names = input_names or ['image_curr', 'vel']
        self.output_names = output_names or ['y_diff_pred', 'y', 'y_next_pred', 'image_next_pred', 'x0_next_pred'] # ok if some of these don't exist
        self.x_shape = x_shape
        self.u_shape = u_shape
        self.net_name = net_name
        self.postfix = postfix
        self.backend = backend

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

    def response_maps_from_input(self, X):
        return collections.OrderedDict()

    def predict_response_maps_from_input(self, X, U):
        return collections.OrderedDict()

    def copy_from(self, params_fname):
        raise NotImplementedError

    def visualize_response_maps(self, x, u, x_next=None, w=None):
        xlevels = self.response_maps_from_input(x)
        xlevels_next_pred = self.predict_response_maps_from_input(x, u)
        xlevels_all = [[('x', x), *xlevels.items()],
                       [(None, None), *xlevels_next_pred.items()]]
        if x_next is not None:
            xlevels_next = self.response_maps_from_input(x_next)
            xlevels_all.append([('x_next', x_next), *[(name+'_next', xlevel) for (name, xlevel) in xlevels_next.items()]])

        if w is None:
            is_w_ones = True
        else:
            is_w_ones = np.all(w == 1)
            w = w.copy()
        plt.ion()
        num_rows = len(xlevels_all) * (1 if is_w_ones else 2)
        num_cols = max(len(xlevels) for xlevels in xlevels_all)
        fig, axarr = plt.subplots(num_rows, num_cols,
                                  num=1, figsize=(num_cols*6, num_rows*6))
        if axarr.ndim == 1:
            axarr = axarr.reshape((1, -1))
        # calculate min and max of responses at each level in order to consistently normalize the data
        xlevels_min = np.inf * np.ones(num_cols)
        xlevels_max = -np.inf * np.ones(num_cols)
        for xlevels in xlevels_all:
            for i, (name, xlevel) in enumerate(xlevels):
                if xlevel is None:
                    continue
                xlevels_min[i] = min(xlevels_min[i], xlevel.min())
                xlevels_max[i] = max(xlevels_max[i], xlevel.max())
        # plot images and response maps
        row = 0
        for xlevels in xlevels_all:
            for i, (name, xlevel) in enumerate(xlevels):
                if xlevel is None:
                    fig.delaxes(axarr[row, i])
                    continue
                if xlevel.shape[0] == 3:
                    axarr[row, i].imshow(cv2.cvtColor(util.image_from_obs(xlevel), cv2.COLOR_BGR2RGB))
                else:
                    axarr[row, i].imshow(util.vis_square(xlevel,
                                                         data_min=xlevels_min[i],
                                                         data_max=xlevels_max[i]))
                axarr[row, i].set_title(name)
                if i != 0 and not is_w_ones:
                    axarr[row+1, i].imshow(util.vis_square(xlevel * w[:xlevel.size].reshape(xlevel.shape),
                                                           data_min=xlevels_min[i],
                                                           data_max=xlevels_max[i]))
                    w = w[xlevel.size:]
            row += (1 if is_w_ones else 2)
        plt.draw()

    def restore_losses(self, curr_iter=0, num_test_nets=1):
        loss_txt_fname = self.get_loss_fname()
        headers = ['iter', 'train loss'] + ['test %d loss'%i_test for i_test in range(num_test_nets)]
        if os.path.isfile(loss_txt_fname):
            iter_loss_items = np.loadtxt(loss_txt_fname, dtype={'names': headers, 'formats': [np.int] + [np.float]*(1+num_test_nets)}, unpack=True)
            iters, losses, *val_losses = [item.reshape(-1) for item in iter_loss_items]
            iters = [iter_ for iter_ in iters if iter_ < curr_iter]
            losses = losses[:len(iters)].tolist()
            val_losses = [test_losses[:len(iters)].tolist() for test_losses in val_losses]
        else:
            iters = []
            losses = []
            val_losses = [[] for _ in range(num_test_nets)]
        return iters, losses, val_losses

    def save_losses(self, iters, losses, val_losses):
        loss_txt_fname = self.get_loss_fname()
        headers = ['iter', 'train loss'] + ['test %d loss'%i_test for i_test in range(len(val_losses))]
        np.savetxt(loss_txt_fname, np.asarray([iters, losses] + val_losses).T, fmt=['%d'] + ['%.2f']*(1+len(val_losses)), delimiter='\t', header='\t'.join(headers))
        loss_fig_fname = self.get_loss_fname(ext='.pdf')
        plt.ion()
        fig = plt.figure(2)
        plt.cla()
        fig.canvas.set_window_title(self.net_name + '_' + self.postfix)
        plt.plot(iters, losses, label='train')
        for i_test, test_losses in enumerate(val_losses):
            plt.plot(iters, test_losses, label='test %d'%i_test)
        plt.ylabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.draw()
        plt.savefig(loss_fig_fname)

    def get_model_dir(self):
        if self.backend is not None:
            model_dir = os.path.join('models', self.backend, self.net_name + '_' + self.postfix)
        else:
            model_dir = os.path.join('models', self.net_name + '_' + self.postfix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def get_snapshot_prefix(self):
        snapshot_dir = os.path.join(self.get_model_dir(), 'snapshot')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        snapshot_prefix = os.path.join(snapshot_dir, '')
        return snapshot_prefix

    def get_loss_fname(self, ext='.txt'):
        return os.path.join(self.get_model_dir(), 'loss' + ext)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('val_hdf5_fname', nargs='?', type=str, default=None)
    parser.add_argument('--backends', nargs='+', type=str, choices=['caffe', 'theano', 'cgt'], default=['theano'], help='backends to compare against')
    args = parser.parse_args()

    if 'caffe' in args.backends:
        from caffe.proto import caffe_pb2 as pb2
        from . import predictor_caffe
    if 'theano' in args.backends:
        import theano
        from . import predictor_theano
        from . import net_theano
        theano.config.floatX = 'float64' # override floatX to float64 for comparison purposes
    if 'cgt' in args.backends:
        import cgt
        from . import predictor_theano
        from . import net_cgt
        cgt.floatX = 'float64'

    if args.val_hdf5_fname is None:
        val_file = h5py.File(args.train_hdf5_fname.replace('train', 'val'), 'r+')
    else:
        val_file = h5py.File(args.val_hdf5_fname, 'r+')

    input_shapes = FeaturePredictor.infer_input_shapes(args.train_hdf5_fname)

    if 'caffe' in args.backends:
        predictor_bn = predictor_caffe.BilinearNetFeaturePredictor(input_shapes)
    if 'theano' in args.backends:
        predictor_tbn = predictor_theano.TheanoNetFeaturePredictor(*net_theano.build_bilinear_net(input_shapes))
    if 'cgt' in args.backends:
        predictor_cbn = predictor_theano.CgtNetFeaturePredictor(*net_cgt.build_bilinear_net(input_shapes))
    predictor_b = BilinearFeaturePredictor(*input_shapes)

    # train
    train_file = h5py.File(args.train_hdf5_fname, 'r+')
    X = train_file['image_curr'][:]
    U = train_file['vel'][:]
    X_dot = train_file['image_diff'][:]
    Y_dot = predictor_b.flatten(X_dot)
    N = len(X)
    if 'caffe' in args.backends:
        predictor_bn.train(args.train_hdf5_fname, args.val_hdf5_fname, solver_param=pb2.SolverParameter(max_iter=100))
        print("bn train error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N))
    if 'theano' in args.backends:
        predictor_tbn.train(args.train_hdf5_fname, args.val_hdf5_fname, max_iter=100)
        print("tbn train error", (np.linalg.norm(Y_dot - predictor_tbn.predict(X, U))**2) / (2*N))
    if 'cgt' in args.backends:
        predictor_cbn.train(args.train_hdf5_fname, args.val_hdf5_fname, max_iter=100)
        print("cbn train error", (np.linalg.norm(Y_dot - predictor_cbn.predict(X, U))**2) / (2*N))
    predictor_b.train(X, U, Y_dot)
    print("b train error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N))

    # validation
    X = val_file['image_curr'][:]
    U = val_file['vel'][:]
    X_dot = val_file['image_diff'][:]
    Y_dot = predictor_b.flatten(X_dot)
    N = len(X)
    # set parameters of bn to the ones of b and check that their methods return the same outputs
    if 'caffe' in args.backends:
        print("bn validation error", (np.linalg.norm(Y_dot - predictor_bn.predict(X, U))**2) / (2*N))
    if 'theano' in args.backends:
        print("tbn validation error", (np.linalg.norm(Y_dot - predictor_tbn.predict(X, U))**2) / (2*N))
    if 'cgt' in args.backends:
        print("cbn validation error", (np.linalg.norm(Y_dot - predictor_cbn.predict(X, U))**2) / (2*N))
    print("b validation error", (np.linalg.norm(Y_dot - predictor_b.predict(X, U))**2) / (2*N))

    # set parameters to those of predictor_b
    if 'caffe' in args.backends:
        predictor_bn.params['bilinear_fc_outer_yu'][0].data[...] = predictor_b.Q.reshape((predictor_b.Q.shape[0], -1))
        predictor_bn.params['bilinear_fc_u'][0].data[...] = predictor_b.R
        # TODO params for the other terms
    if 'theano' in args.backends:
        predictor_tbn_params = {param.name: param for param in predictor_tbn.get_all_params()}
        predictor_tbn_params['Q'].set_value(predictor_b.Q.astype(theano.config.floatX))
        predictor_tbn_params['R'].set_value(predictor_b.R.astype(theano.config.floatX))
        predictor_tbn_params['S'].set_value(predictor_b.S.astype(theano.config.floatX))
        predictor_tbn_params['b'].set_value(predictor_b.b.astype(theano.config.floatX))
    if 'cgt' in args.backends:
        predictor_cbn_params = {param.name: param for param in predictor_cbn.get_all_params()}
        predictor_cbn_params['bilinear.M'].op.set_value(predictor_b.Q.astype(cgt.floatX))
        predictor_cbn_params['bilinear.N'].op.set_value(predictor_b.R.astype(cgt.floatX))
        # TODO params for the other terms

    # check predictions are the same
    Y_dot_b = predictor_b.predict(X, U)
    if 'caffe' in args.backends:
        Y_dot_bn = predictor_bn.predict(X, U)
        print("Y_dot_bn, Y_dot_b")
        print("\tall close", np.allclose(Y_dot_bn, Y_dot_b))
        print("\tnorm", np.linalg.norm(Y_dot_bn - Y_dot_b))
    if 'theano' in args.backends:
        Y_dot_tbn = predictor_tbn.predict(X, U)
        print("Y_dot_tbn, Y_dot_b")
        print("\tall close", np.allclose(Y_dot_tbn, Y_dot_b))
        print("\tnorm", np.linalg.norm(Y_dot_tbn - Y_dot_b))
    if 'cgt' in args.backends:
        Y_dot_cbn = predictor_cbn.predict(X, U)
        print("Y_dot_cbn, Y_dot_b")
        print("\tall close", np.allclose(Y_dot_cbn, Y_dot_b))
        print("\tnorm", np.linalg.norm(Y_dot_cbn - Y_dot_b))

    # check jacobians are the same
    jac_b = predictor_b.jacobian_control(X, U)
    if 'caffe' in args.backends:
        jac_bn, _ = predictor_bn.jacobian_control(X, U)
        print("jac_bn, jac_b")
        print("\tall close", np.allclose(jac_bn, jac_b))
        print("\tnorm", np.linalg.norm(jac_bn - jac_b))
    if 'theano' in args.backends:
        jac_tbn = predictor_tbn.jacobian_control(X, U)
        print("jac_tbn, jac_b")
        print("\tall close", np.allclose(jac_tbn, jac_b))
        print("\tnorm", np.linalg.norm(jac_tbn - jac_b))
    if 'cgt' in args.backends:
        jac_cbn = predictor_cbn.jacobian_control(X, U)
        print("jac_cbn, jac_b")
        print("\tall close", np.allclose(jac_cbn, jac_b))
        print("\tnorm", np.linalg.norm(jac_cbn - jac_b))


    if 'theano' in args.backends:
        # predictors with axis=2
        input_shapes_c = [(1,) + input_shapes[0][1:], input_shapes[1]] # shape per channel
        y_dim = np.prod(input_shapes[0])
        y_dim_c = np.prod(input_shapes_c[0])
        u_dim, = input_shapes[1]

        predictor_b_c = BilinearFeaturePredictor(*input_shapes_c)
        predictor_tbn_ax2 = predictor_theano.TheanoNetFeaturePredictor(*net_theano.build_bilinear_net(input_shapes, axis=2))

        # set parameters to the first ones of predictor_b
        predictor_b_c.Q = predictor_b.Q[:y_dim_c, :y_dim_c, :]
        predictor_b_c.R = predictor_b.R[:y_dim_c, :]
        predictor_b_c.S = predictor_b.S[:y_dim_c, :y_dim_c]
        predictor_b_c.b = predictor_b.b[:y_dim_c]
        predictor_tbn_ax2_params = {param.name: param for param in predictor_tbn_ax2.get_all_params()}
        predictor_tbn_ax2_params['Q'].set_value(predictor_b_c.Q.astype(theano.config.floatX))
        predictor_tbn_ax2_params['R'].set_value(predictor_b_c.R.astype(theano.config.floatX))
        predictor_tbn_ax2_params['S'].set_value(predictor_b_c.S.astype(theano.config.floatX))
        predictor_tbn_ax2_params['b'].set_value(predictor_b_c.b.astype(theano.config.floatX))

        # check predictions are the same
        Y_dot_b_ax2 = []
        for c in range(X.shape[1]):
            X_c = X[:, c:c+1, :, :]
            Y_dot_b_c = predictor_b_c.predict(X_c, U)
            Y_dot_b_ax2.append(Y_dot_b_c)
        Y_dot_b_ax2 = np.asarray(Y_dot_b_ax2).swapaxes(0, 1).reshape((-1, y_dim))
        Y_dot_tbn_ax2 = predictor_tbn_ax2.predict(X, U)
        print("Y_dot_tbn_ax2, Y_dot_b_ax2")
        print("\tall close", np.allclose(Y_dot_tbn_ax2, Y_dot_b_ax2))
        print("\tnorm", np.linalg.norm(Y_dot_tbn_ax2 - Y_dot_b_ax2))
        # check jacobians are the same
        jac_b_ax2 = []
        for c in range(X.shape[1]):
            X_c = X[:, c:c+1, :, :]
            jac_b_c = predictor_b_c.jacobian_control(X_c, U)
            jac_b_ax2.append(jac_b_c)
        jac_b_ax2 = np.asarray(jac_b_ax2).swapaxes(0, 1).reshape((-1, y_dim, u_dim))
        jac_tbn_ax2 = predictor_tbn_ax2.jacobian_control(X, U)
        print("jac_tbn_ax2, jac_b_ax2")
        print("\tall close", np.allclose(jac_tbn_ax2, jac_b_ax2))
        print("\tnorm", np.linalg.norm(jac_tbn_ax2 - jac_b_ax2))


if __name__ == "__main__":
    main()
