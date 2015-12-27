from __future__ import division

import os
import numpy as np
import h5py
import cPickle
import theano
import theano.tensor as T
import lasagne
import predictor


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
    def __init__(self, net_name, input_vars, pred_layers, loss, loss_deterministic=None, prediction_name=None, postfix=''):
        self.input_vars = input_vars
        self.pred_layers = pred_layers
        self.l_x_next_pred = pred_layers['X_next_pred']
        self.l_y_diff_pred = pred_layers['Y_diff_pred']
        self.loss = loss
        self.loss_deterministic = loss_deterministic or loss
        input_layers = {layer.input_var.name: layer for layer in lasagne.layers.get_all_layers(self.l_x_next_pred) if type(layer) == lasagne.layers.InputLayer}
        x_shape, u_shape = input_layers['X'].shape[1:], input_layers['U'].shape[1:]
        self.X_var, self.U_var, self.X_diff_var = input_vars.values()
        self.prediction_name = prediction_name or pred_layers.keys()[0]
        self.pred_vars = {}
        self.pred_fns = {}
        self.jacobian_var = self.jacobian_fn = None
        predictor.FeaturePredictor.__init__(self, x_shape, u_shape, net_name=net_name, postfix=postfix)

    def train(self, train_hdf5_fname, val_hdf5_fname=None,
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
        param_l2_penalty = lasagne.regularization.regularize_network_params(self.l_x_next_pred, lasagne.regularization.l2)
        loss = self.loss + weight_decay * param_l2_penalty / 2.

        # training function
        params = lasagne.layers.get_all_params(self.l_x_next_pred, trainable=True)
        learning_rate = theano.shared(np.asarray(base_lr, dtype=theano.config.floatX))
        if solver_type == 'SGD':
            if momentum:
                updates = lasagne.updates.momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
            else:
                updates = lasagne.updates.sgd(loss, params, learning_rate=learning_rate)
        elif solver_type == 'ADAM':
            updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate, beta1=momentum, beta2=momentum2)
        else:
            raise
        print "Compiling training function..."
        train_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], loss, updates=updates)

        validate = test_interval and val_hdf5_fname is not None
        if validate:
            # validation loss
            test_loss = self.loss_deterministic + weight_decay * param_l2_penalty / 2.

            # validation function
            print "Compiling validation function..."
            val_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], test_loss)

        print("Starting training...")
        iter_ = 0
        while iter_ < max_iter:
            if validate and iter_ % test_interval == 0:
                self.test_all(val_fn, val_hdf5_fname, batch_size, test_iter)

            current_step = iter_ / stepsize
            rate = base_lr * gamma ** current_step
            learning_rate.set_value(rate)

            X, U, X_next = next(minibatches)
            loss = train_fn(X, U, X_next)

            if display and iter_ % display == 0:
                print("Iteration {} of {}, lr = {}".format(iter_, max_iter, learning_rate.get_value()))
                print("    training loss = {:.6f}".format(float(loss)))
            iter_ += 1
            if snapshot and iter_ % snapshot == 0 and iter_ > 0:
                self.snapshot(iter_, snapshot_prefix)

        if snapshot and not (snapshot and iter_ % snapshot == 0 and iter_ > 0):
            self.snapshot(iter_, snapshot_prefix)
        if display and iter_ % display == 0:
            print("Iteration {} of {}, lr = {}".format(iter_, max_iter, learning_rate.get_value()))
            print("    training loss = {:.6f}".format(float(loss)))
        if validate and iter_ % test_interval == 0:
            self.test_all(val_fn, val_hdf5_fname, batch_size, test_iter)

    def _predict(self, X, U, prediction_name=None):
        prediction_name = prediction_name or self.prediction_name
        if prediction_name == 'image_next_pred':
            prediction_name = 'X_next_pred'
        if prediction_name in self.pred_fns:
            pred_fn = self.pred_fns[prediction_name]
        else:
            if prediction_name not in self.pred_vars:
                self.pred_vars[prediction_name] = lasagne.layers.get_output(self.pred_layers[prediction_name], deterministic=True)
            input_vars = [self.X_var, self.U_var] if U is not None else [self.X_var]
            pred_fn = theano.function(input_vars, self.pred_vars[prediction_name])
            self.pred_fns[prediction_name] = pred_fn
        if U is None:
            pred = pred_fn(X)
        else:
            pred = pred_fn(X, U)
        return pred
    
    def predict(self, X, U, prediction_name=None):
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        is_batched = X.shape == self.x_shape
        if X.dtype != theano.config.floatX:
            X = X.astype(theano.config.floatX)
        if U is not None and U.dtype != theano.config.floatX:
            U = U.astype(theano.config.floatX)
        if is_batched:
            X = X[None, ...]
            if U is not None:
                U = U[None, :]
        pred = self._predict(X, U, prediction_name=prediction_name)
        if is_batched:
            pred = np.squeeze(pred, 0)
        return pred

    def _jacobian_control(self, X, U):
        if self.jacobian_fn is None:
            prediction_name = 'Y_diff_pred'
            if prediction_name in self.pred_vars:
                Y_diff_pred_var = self.pred_vars[prediction_name]
            else:
                Y_diff_pred_var = lasagne.layers.get_output(self.pred_layers[prediction_name], deterministic=True)
                self.pred_vars[prediction_name] = Y_diff_pred_var
            self.jacobian_var, updates = theano.scan(lambda i, Y_diff_pred_var, U_var: theano.gradient.jacobian(Y_diff_pred_var[i], U_var)[:, i, :], sequences=T.arange(Y_diff_pred_var.shape[0]), non_sequences=[Y_diff_pred_var, self.U_var])
            input_vars = [self.X_var, self.U_var] if U is not None else [self.X_var]
            jacobian_fn = theano.function(input_vars, self.jacobian_var, updates=updates)
            self.jacobian_fn = jacobian_fn
        if U is None:
            jac = jacobian_fn(X)
        else:
            jac = jacobian_fn(X, U)
        return jac

    def jacobian_control(self, X, U):
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        is_batched = X.shape == self.x_shape
        if X.dtype != theano.config.floatX:
            X = X.astype(theano.config.floatX)
        if U is not None and U.dtype != theano.config.floatX:
            U = U.astype(theano.config.floatX)
        if is_batched:
            X = X[None, ...]
            if U is not None:
                U = U[None, :]
        jac = self._jacobian_control(X, U)
        if is_batched:
            jac = np.squeeze(jac, 0)
        return jac

    def feature_from_input(self, X):
        return self.predict(X, None, 'Y')

    def test_all(self, val_fn, val_hdf5_fname, batch_size, test_iter):
        loss = 0
        minibatches = iterate_minibatches_indefinitely(val_hdf5_fname, 'image_curr', 'vel', 'image_diff',
                                                       batch_size=batch_size, shuffle=False)
        for _ in range(test_iter):
            X, U, X_next = next(minibatches)
            loss += val_fn(X, U, X_next)
        print("    validation loss = {:.6f}".format(loss / test_iter))

    def snapshot(self, iter_, snapshot_prefix):
        snapshot_prefix = snapshot_prefix or self.get_snapshot_prefix()
        snapshot_fname = snapshot_prefix + '_iter_%d.pkl'%iter_
        snapshot_file = file(snapshot_fname, 'wb')
        all_param_values = lasagne.layers.get_all_param_values(self.l_x_next_pred)
        print "Snapshotting to pickle file", snapshot_fname
        cPickle.dump(all_param_values, snapshot_file, protocol=cPickle.HIGHEST_PROTOCOL)
        snapshot_file.close()
    
    def copy_from(self, snapshot_fname):
        print "Copying weights from pickle file", snapshot_fname
        all_param_values = cPickle.load(open(snapshot_fname, 'rb'))
        lasagne.layers.set_all_param_values(self.l_x_next_pred, all_param_values)

    def get_model_dir(self):
        model_dir = predictor.FeaturePredictor.get_model_dir(self)
        model_dir = os.path.join(model_dir, 'theano', self.net_name + self.postfix)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir
