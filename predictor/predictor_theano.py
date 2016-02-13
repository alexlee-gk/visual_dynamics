import numpy as np
import h5py
import pickle
from collections import OrderedDict
import time
import re
import theano
import theano.tensor as T
import lasagne
from . import predictor


def iterate_minibatches(*data, **kwargs):
    batch_size = kwargs.get('batch_size') or 1
    shuffle = kwargs.get('shuffle') or False
    N = len(data[0])
    for datum in data[1:]:
        assert len(datum) == N
    if shuffle:
        indices = np.arange(N)
        np.random.shuffle(indices)
    for start_idx in range(0, N - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield tuple(datum[excerpt] for datum in data)


def iterate_minibatches_indefinitely(hdf5_fname, *data_names, **kwargs):
    batch_size = kwargs.get('batch_size') or 1
    shuffle = kwargs.get('shuffle') or False
    with h5py.File(hdf5_fname, 'r+') as f:
        N = len(f[data_names[0]])
        for data_name in data_names[1:]:
            assert len(f[data_name]) == N
        indices = []
        while True:
            if len(indices) < batch_size:
                new_indices = np.arange(N)
                if shuffle:
                    np.random.shuffle(new_indices)
                indices.extend(new_indices)
            excerpt = np.asarray(indices[0:batch_size])
            unsorted = np.any(excerpt[1:] < excerpt[:-1])
            if unsorted:
                batch_data = tuple(np.asarray([f[data_name][ind][()] for ind in excerpt], dtype=theano.config.floatX) for data_name in data_names)
            else:
                batch_data = tuple(np.asarray(f[data_name][excerpt.tolist()], dtype=theano.config.floatX) for data_name in data_names)
            del indices[0:batch_size]
            yield batch_data


class TheanoNetFeaturePredictor(predictor.FeaturePredictor):
    def __init__(self, net_name, input_vars, pred_layers, loss, loss_deterministic=None, pretrained_file=None, postfix=''):
        """
        input_vars: dict of input variables
        pred_layers: dict of layers of the network, which should contain at least all the root layers of the network.
            Also, allows to specify layers that have multiple names (i.e. different names can map to the same layer).
        """
        self.X_var, self.U_var, self.X_diff_var = [input_vars[var_name] for var_name in ['X', 'U', 'X_diff']]
        self.pred_layers = OrderedDict(pred_layers)
        for pred_layer in list(pred_layers.values()):
            self.pred_layers.update((layer.name, layer) for layer in lasagne.layers.get_all_layers(pred_layer) if layer.name is not None)
        layer_name_aliases = [('x0', 'x'), ('x', 'x0'), ('x0_next', 'x_next'), ('x_next', 'x0_next'), ('x0', 'image_curr'), ('x0_next', 'image_next'), ('x0_next_pred', 'image_next_pred')]
        for name, name_alias  in layer_name_aliases:
            if name in self.pred_layers:
                self.pred_layers[name_alias] = self.pred_layers[name]
        x_shape, u_shape = (self.pred_layers['x'].shape[1:], self.pred_layers['u'].shape[1:])
        predictor.FeaturePredictor.__init__(self, x_shape, u_shape, net_name=net_name, postfix=postfix, backend='theano')
        self.loss = loss
        self.loss_deterministic = loss_deterministic or loss
        self.pred_fns = {}
        self.jacobian_var = self.jacobian_fn = None
        if pretrained_file is not None and not pretrained_file.endswith('.pkl'):
            pretrained_file = self.get_snapshot_prefix() + '_iter_' + pretrained_file + '.pkl'
        if pretrained_file is not None:
            self.copy_from(pretrained_file)

    def train(self, train_hdf5_fname, val_hdf5_fname=None, solverstate_fname=None,
              batch_size=32,
              test_iter = 10,
              solver_type = 'SGD',
              test_interval = 1000,
              base_lr = 0.05,
              gamma = 0.9,
              stepsize = 1000,
              display = 20,
              max_iter=10000,
              momentum = 0.9,
              momentum2 = 0.999,
              weight_decay=0.0005,
              snapshot=1000,
              snapshot_prefix=None):
        # training data
        minibatches = iterate_minibatches_indefinitely(train_hdf5_fname, 'image_curr', 'vel', 'image_diff',
                                                       batch_size=batch_size, shuffle=True)

        # training loss
        param_l2_penalty = lasagne.regularization.regularize_network_params(self.pred_layers['x0_next_pred'], lasagne.regularization.l2)
        loss = self.loss + weight_decay * param_l2_penalty / 2.

        # training function
        params = self.get_all_params(trainable=True)
        learning_rate_var = T.scalar(name='learning_rate')
        if solver_type == 'SGD':
            if momentum:
                updates = lasagne.updates.momentum(loss, params, learning_rate_var, momentum)
            else:
                updates = lasagne.updates.sgd(loss, params, learning_rate_var)
        elif solver_type == 'ADAM':
            updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate_var, beta1=momentum, beta2=momentum2)
        else:
            raise

        start_time = time.time()
        print("Compiling training function...")
        train_fn = theano.function([self.X_var, self.U_var, self.X_diff_var, learning_rate_var], loss, updates=updates)
        print("... finished in %.2f s"%(time.time() - start_time))

        validate = test_interval and val_hdf5_fname is not None
        if validate:
            # validation loss
            test_loss = self.loss_deterministic + weight_decay * param_l2_penalty / 2.

            # validation function
            start_time = time.time()
            print("Compiling validation function...")
            val_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], test_loss)
            print("... finished in %.2f s"%(time.time() - start_time))

        if solverstate_fname is not None and not solverstate_fname.endswith('.pkl'):
            solverstate_fname = self.get_snapshot_prefix() + '_iter_' + solverstate_fname + '.pkl'
        iter_ = self.restore(solverstate_fname)
        print("Starting training...")
        while iter_ < max_iter:
            current_step = iter_ // stepsize
            learning_rate = base_lr * gamma ** current_step

            if validate and iter_ % test_interval == 0:
                self.test_all(val_fn, val_hdf5_fname, batch_size, test_iter)

            X, U, X_next = next(minibatches)
            loss = train_fn(X, U, X_next, learning_rate)

            if display and iter_ % display == 0:
                print(("Iteration {} of {}, lr = {}".format(iter_, max_iter, learning_rate)))
                print(("    training loss = {:.6f}".format(float(loss))))
            iter_ += 1
            if snapshot and iter_ % snapshot == 0 and iter_ > 0:
                self.snapshot(iter_, snapshot_prefix)

        if snapshot and not (snapshot and iter_ % snapshot == 0 and iter_ > 0):
            self.snapshot(iter_, snapshot_prefix)
        if display and iter_ % display == 0:
            print(("Iteration {} of {}, lr = {}".format(iter_, max_iter, learning_rate)))
            print(("    training loss = {:.6f}".format(float(loss))))
        if validate and iter_ % test_interval == 0:
            self.test_all(val_fn, val_hdf5_fname, batch_size, test_iter)

    def restore(self, solverstate_fname):
        """
        Restores if solverstate_fname is not None and returns the iteration at which the network is restored
        """
        if solverstate_fname is not None:
            self.copy_from(solverstate_fname)
            match = re.match('.*_iter_(\d+).pkl$', solverstate_fname)
            if match:
                assert len(match.groups()) == 1
                iter_ = int(match.group(1))
        else:
            iter_ = 0
        return iter_

    def _predict(self, *inputs, prediction_name=None):
        prediction_name = prediction_name or 'x0_next_pred'
        if prediction_name in self.pred_fns:
            pred_fn = self.pred_fns[prediction_name]
        else:
            pred_var = lasagne.layers.get_output(self.pred_layers[prediction_name], deterministic=True)
            input_vars = [self.X_var]
            if self.U_var in theano.gof.graph.inputs([pred_var]):
                input_vars.append(self.U_var)
            start_time = time.time()
            print("Compiling prediction function...")
            pred_fn = theano.function(input_vars, pred_var)
            print("... finished in %.2f s"%(time.time() - start_time))
            self.pred_fns[prediction_name] = pred_fn
        return pred_fn(*inputs)

    def predict(self, *inputs, prediction_name=None):
        X = inputs[0]
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        is_batched = X.shape == self.x_shape
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if is_batched:
            inputs = [input_[None, ...] for input_ in inputs]
        pred = self._predict(*inputs, prediction_name=prediction_name)
        if is_batched:
            pred = np.squeeze(pred, 0)
        return pred

    def _jacobian_control(self, *inputs):
        if self.jacobian_fn is None:
            prediction_name = 'y_diff_pred'
            Y_diff_pred_var = lasagne.layers.get_output(self.pred_layers[prediction_name], deterministic=True)
            self.jacobian_var, updates = theano.scan(lambda i, Y_diff_pred_var, U_var: theano.gradient.jacobian(Y_diff_pred_var[i], U_var)[:, i, :], sequences=T.arange(Y_diff_pred_var.shape[0]), non_sequences=[Y_diff_pred_var, self.U_var])
            input_vars = [self.X_var]
            if self.U_var in theano.gof.graph.inputs([self.jacobian_var]):
                input_vars.append(self.U_var)
            start_time = time.time()
            print("Compiling jacobian function...")
            self.jacobian_fn = theano.function(input_vars, self.jacobian_var, updates=updates)
            print("... finished in %.2f s"%(time.time() - start_time))
        return self.jacobian_fn(*inputs)

    def jacobian_control(self, *inputs):
        X = inputs[0]
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        is_batched = X.shape == self.x_shape
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if is_batched:
            inputs = [input_[None, ...] for input_ in inputs]
        jac = self._jacobian_control(*inputs)
        if is_batched:
            jac = np.squeeze(jac, 0)
        return jac

    def feature_from_input(self, X):
        return self.predict(X, prediction_name='Y')

    def features_from_input(self, X):
        levels = []
        for key in self.pred_layers.keys():
            match = re.match('x(\d+)$', key)
            if match:
                assert len(match.groups()) == 1
                levels.append(int(match.group(1)))
        levels = sorted(levels)
        xlevels = OrderedDict()
        for level in levels:
            output_name = 'x%d'%level
            xlevels[output_name] = self.predict(X, prediction_name=output_name)
        return xlevels

    def get_all_params(self, **tags):
        return lasagne.layers.get_all_params(self.pred_layers['x0_next_pred'], **tags)

    def get_all_param_values(self, **tags):
        return lasagne.layers.get_all_param_values(self.pred_layers['x0_next_pred'], **tags)

    def set_all_param_values(self, all_param_values, **tags):
        lasagne.layers.set_all_param_values(self.pred_layers['x0_next_pred'], all_param_values, **tags)

    def test_all(self, val_fn, val_hdf5_fname, batch_size, test_iter):
        loss = 0
        minibatches = iterate_minibatches_indefinitely(val_hdf5_fname, 'image_curr', 'vel', 'image_diff',
                                                       batch_size=batch_size, shuffle=False)
        for _ in range(test_iter):
            X, U, X_next = next(minibatches)
            loss += val_fn(X, U, X_next)
        print(("    validation loss = {:.6f}".format(loss / test_iter)))

    def snapshot(self, iter_, snapshot_prefix):
        snapshot_prefix = snapshot_prefix or self.get_snapshot_prefix()
        snapshot_fname = snapshot_prefix + '_iter_%d.pkl'%iter_
        snapshot_file = open(snapshot_fname, 'wb')
        all_param_values = self.get_all_param_values()
        print("Snapshotting to pickle file", snapshot_fname)
        pickle.dump(all_param_values, snapshot_file, protocol=pickle.HIGHEST_PROTOCOL)
        snapshot_file.close()

    def copy_from(self, snapshot_fname):
        print("Copying weights from pickle file", snapshot_fname)
        all_param_values = pickle.load(open(snapshot_fname, 'rb'))
        for i, param_value in enumerate(all_param_values):
            if param_value.dtype != theano.config.floatX:
                all_param_values[i] = param_value.astype(theano.config.floatX)
        self.set_all_param_values(all_param_values)
