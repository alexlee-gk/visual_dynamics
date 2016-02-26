import numpy as np
import h5py
import pickle
from collections import OrderedDict
import contextlib
import time
import re
import theano
import theano.tensor as T
import lasagne
from . import predictor
import bilinear


def iterate_minibatches_once(*hdf5_fnames, data_names, batch_size=1):
    """
    Iterate through all the data once in order. The data from contiguous files
    are treated as if they are contiguous. All of the returned minibatches
    contain batch_size data points except for possibly the last batch.
    """
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fname, 'r+')) for fname in hdf5_fnames]
        for f in files:
            for data_name in data_names[1:]:
                assert len(f[data_name]) == len(f[data_names[0]])
        files = iter(files)
        f = next(files)
        idx = 0
        while f:
            excerpts_data = []
            total_excerpt_size = 0
            while total_excerpt_size < batch_size and f:
                excerpt_size = min(idx + batch_size - total_excerpt_size, len(f[data_names[0]])) - idx
                excerpt = slice(idx, idx + excerpt_size)
                excerpt_data = tuple(np.asarray(f[data_name][excerpt][()], dtype=theano.config.floatX) for data_name in data_names)
                excerpts_data.append(excerpt_data)
                total_excerpt_size += excerpt_size
                if idx + excerpt_size == len(f[data_names[0]]):
                    try:
                        f = next(files)
                    except StopIteration:
                        f = None
                    idx = 0
                else:
                    idx += excerpt_size
            batch_data = [np.concatenate(data, axis=0) for data in zip(*excerpts_data)]
            yield batch_data


def iterate_minibatches_indefinitely(*hdf5_fnames, data_names, batch_size=1, shuffle=False):
    """
    Iterate through all the data indefinitely. The data from contiguous files
    are treated as if they are contiguous. All of the returned minibatches
    contain batch_size data points. If shuffle=True, the data is iterated in a
    random order, and this order differs for each pass of the data.
    Note: this is not as efficient as it could be when shuffle=False since each
    data point is retrieved one by one regardless of the value of shuffle.
    """
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fname, 'r+')) for fname in hdf5_fnames]
        for f in files:
            for data_name in data_names[1:]:
                assert len(f[data_name]) == len(f[data_names[0]])
        data_sizes = [len(f[data_names[0]]) for f in files]
        total_data_size = sum(data_sizes)
        data_sizes_cs = np.r_[0, np.cumsum(data_sizes)]
        def get_file_local_idx(idx):
            local_idx = idx - data_sizes_cs
            file_idx = np.asscalar(np.where(local_idx >= 0)[0][-1]) # get index of last non-negative entry
            local_idx = local_idx[file_idx]
            return file_idx, local_idx
        indices = []
        while True:
            if len(indices) < batch_size:
                new_indices = np.arange(total_data_size)
                if shuffle:
                    np.random.shuffle(new_indices)
                indices.extend(new_indices)
            excerpt = np.asarray(indices[0:batch_size])
            batch_data = tuple(np.empty((batch_size, *f[data_name].shape[1:]), dtype=theano.config.floatX) for data_name in data_names)
            for data_name, datum in zip(data_names, batch_data):
                datasets = [file[data_name] for file in files] # datasets[file_idx] == files[file_idx][data_name]
                for i, idx in enumerate(excerpt):
                    file_idx, local_idx = get_file_local_idx(idx)
                    datum[i, ...] = np.asarray(datasets[file_idx][local_idx][()], dtype=theano.config.floatX)
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
            if name in self.pred_layers and name_alias not in self.pred_layers:
                self.pred_layers[name_alias] = self.pred_layers[name]
        self.pred_vars = dict(zip(self.pred_layers.keys(),
                                  lasagne.layers.get_output(self.pred_layers.values(), deterministic=True)))
        print("Network %s has %d parameters"%(net_name, lasagne.layers.count_params(self.pred_layers['x0_next_pred'])))
        x_shape, u_shape = (self.pred_layers['x'].shape[1:], self.pred_layers['u'].shape[1:])
        predictor.FeaturePredictor.__init__(self, x_shape, u_shape, net_name=net_name, postfix=postfix, backend='theano')
        self.loss = loss
        self.loss_deterministic = loss_deterministic or loss
        self.pred_fns = {}
        self.jac_fns = {}
        if pretrained_file is not None and not pretrained_file.endswith('.pkl'):
            pretrained_file = self.get_snapshot_prefix() + '_iter_' + pretrained_file + '.pkl'
        if pretrained_file is not None:
            self.copy_from(pretrained_file)

    def train(self, *train_hdf5_fnames, val_hdf5_fname, solverstate_fname=None,
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
              snapshot_prefix=None,
              visualize_response_maps=False):
        # training data
        minibatches = iterate_minibatches_indefinitely(*train_hdf5_fnames, data_names=['image_curr', 'vel', 'image_diff'],
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
            # validation data
            val_minibatches = iterate_minibatches_indefinitely(val_hdf5_fname, data_names=['image_curr', 'vel', 'image_diff'],
                                                               batch_size=batch_size, shuffle=True)
            # validation loss
            test_loss = self.loss_deterministic + weight_decay * param_l2_penalty / 2.
            # validation function
            start_time = time.time()
            print("Compiling validation function...")
            val_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], test_loss)
            print("... finished in %.2f s"%(time.time() - start_time))

        if solverstate_fname is not None and not solverstate_fname.endswith('.pkl'):
            solverstate_fname = self.get_snapshot_prefix() + '_iter_' + solverstate_fname + '.pkl'
        curr_iter = self.restore_solver(solverstate_fname)
        # load losses for visualization
        iters, losses, val_losses = self.restore_losses(curr_iter=curr_iter)
        print("Starting training...")
        for iter_ in range(curr_iter, max_iter):
            current_step = iter_ // stepsize
            learning_rate = base_lr * gamma ** current_step

            if validate and iter_ % test_interval == 0:
                test_loss = sum([val_fn(*next(val_minibatches)) for _ in range(test_iter)]) / test_iter
                print("    validation loss = {:.6f}".format(float(test_loss)))

            X, U, X_diff = next(minibatches)
            loss = train_fn(X, U, X_diff, learning_rate)

            if display and iter_ % display == 0:
                print(("Iteration {} of {}, lr = {}".format(iter_, max_iter, learning_rate)))
                print(("    training loss = {:.6f}".format(float(loss))))
                # update, save and visualize losses
                iters.append(iter_)
                losses.append(loss)
                test_losses = val_losses[0]
                test_loss = sum([val_fn(*next(val_minibatches)) for _ in range(test_iter)]) / test_iter
                test_losses.append(test_loss)
                self.save_losses(iters, losses, val_losses)
            # visualize response maps of first image in batch
            if visualize_response_maps and iter_ % visualize_response_maps == 0:
                X, U, X_diff = next(val_minibatches)
                self.visualize_response_maps(X[0], U[0], x_next=X[0]+X_diff[0])
            iter_ += 1
            if snapshot and iter_ % snapshot == 0 and iter_ > 0:
                self.snapshot(iter_, snapshot_prefix)

        if snapshot and not (snapshot and iter_ % snapshot == 0 and iter_ > 0):
            self.snapshot(iter_, snapshot_prefix)
        if display and iter_ % display == 0:
            print("Iteration {} of {}, lr = {}".format(iter_, max_iter, learning_rate))
            print("    training loss = {:.6f}".format(float(loss)))
        if validate and iter_ % test_interval == 0:
            test_loss = sum([val_fn(*next(val_minibatches)) for _ in range(test_iter)]) / test_iter
            print("    validation loss = {:.6f}".format(float(test_loss)))


    def restore_solver(self, solverstate_fname):
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
            pred_var = self.pred_vars[prediction_name]
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

    def _theano_jacobian(self, output_var, wrt_var, *inputs):
        X = inputs[0]
        singleton_batched = X.shape[0] == 1 and X.shape[1:] == self.x_shape
        if X.shape == self.x_shape or singleton_batched:
            if not singleton_batched:
                inputs = [input_[None, :] for input_ in inputs]
            vars_ = (output_var, wrt_var)
            if vars_ in self.jac_fns:
                jac_fn = self.jac_fns[vars_]
            else:
                jac_var = theano.gradient.jacobian(output_var.flatten(), wrt_var)
                input_vars = [self.X_var]
                if self.U_var in theano.gof.graph.inputs([jac_var]):
                    input_vars.append(self.U_var)
                start_time = time.time()
                print("Compiling jacobian function...")
                jac_fn = theano.function(input_vars, jac_var)
                print("... finished in %.2f s"%(time.time() - start_time))
                self.jac_fns[vars_] = jac_fn
            jac = jac_fn(*inputs)
            jac = jac.swapaxes(0, 1)
            jac = jac.reshape((*jac.shape[:2], -1))
            if not singleton_batched:
                jac = np.squeeze(jac, axis=0)
            return jac
        else:
            return np.asarray([self._theano_jacobian(output_var, wrt_var, *single_inputs) for single_inputs in zip(inputs)])

    def _jacobian_control(self, *inputs, use_theano=False):
        """
        Inputs must be batched
        """
        X = inputs[0]
        assert X.shape[1:] == self.x_shape
        if use_theano:
            xlevels = self.response_maps_from_input(X)
            jac = self._theano_jacobian(self.pred_vars['y_diff_pred'], self.U_var, *inputs)
        else:
            jaclevels, xlevels = self.response_map_jacobians_from_input(*inputs)
            jac = np.concatenate(list(jaclevels.values()), axis=1)
        y = np.concatenate([xlevel.reshape(xlevel.shape[0], -1) for xlevel in xlevels.values()], axis=1)
        return jac, y

    def jacobian_control(self, *inputs):
        X = inputs[0]
        assert X.shape == self.x_shape or X.shape[1:] == self.x_shape
        is_batched = X.shape == self.x_shape
        inputs = [input_.astype(theano.config.floatX, copy=False) for input_ in inputs]
        if is_batched:
            inputs = [input_[None, ...] for input_ in inputs]
        jac, y = self._jacobian_control(*inputs)
        if is_batched:
            jac = np.squeeze(jac, 0)
            y = np.squeeze(y, 0)
        return jac, y

    def feature_from_input(self, X):
        return self.predict(X, prediction_name='y')

    def preprocess_input(self, X):
        return self.predict(X, prediction_name='x0')

    def get_levels(self):
        levels = []
        for key in self.pred_layers.keys():
            match = re.match('x(\d+)$', key)
            if match:
                assert len(match.groups()) == 1
                levels.append(int(match.group(1)))
        levels = sorted(levels)
        return levels

    def get_bilinear_levels(self):
        levels = []
        for key in self.pred_layers.keys():
            match = re.match('x(\d+)_next_trans$', key)
            if match:
                assert len(match.groups()) == 1
                levels.append(int(match.group(1)))
        levels = sorted(levels)
        return levels

    def response_maps_from_input(self, X):
        levels = self.get_levels()
        xlevels = OrderedDict()
        for level in levels:
            output_name = 'x%d'%level
            xlevels[output_name] = self.predict(X, prediction_name=output_name)
        return xlevels

    def predict_response_maps_from_input(self, X, U):
        levels = self.get_levels()
        xlevels_next_pred = OrderedDict()
        for level in levels:
            output_name = 'x%d_next_pred'%level
            xlevels_next_pred[output_name] = self.predict(X, U, prediction_name=output_name)
        return xlevels_next_pred

    def response_map_jacobians_from_input(self, X, U):
        xlevels_all = self.response_maps_from_input(X)
        jaclevels_next_pred = dict()

        def response_map_jacobian(level):
            if 'x%d_next_pred'%level in jaclevels_next_pred:
                jaclevel_next_pred = jaclevels_next_pred['x%d_next_pred'%level]
            else:
                xlevel_next_pred_var = self.pred_vars['x%d_next_pred'%level]
                xlevel_diff_trans_var = self.pred_vars.get('x%d_diff_trans'%level, None)
                xlevelp1_next_pred_var = self.pred_vars.get('x%d_next_pred'%(level+1), None)
                jaclevel_next_pred = 0
                if xlevel_diff_trans_var:
                    xlevel_diff_trans_layer = self.pred_layers['x%d_diff_trans'%level]
                    jaclevel_diff_trans = xlevel_diff_trans_layer.get_output_jacobian_for([xlevels_all['x%d'%level], U])
                    if self.pred_vars['x%d_next_pred'%level] == self.pred_vars['x%d_next_trans'%level]:
                        jaclevel_next_pred += jaclevel_diff_trans
                    else:
                        jaclevel_next_pred += np.einsum('nik,nkj->nij',
                                                        self._theano_jacobian(xlevel_next_pred_var, xlevel_diff_trans_var, X, U),
                                                        jaclevel_diff_trans)
                if xlevelp1_next_pred_var:
                    jaclevelp1_next_pred = response_map_jacobian(level+1)
                    jaclevel_next_pred += np.einsum('nik,nkj->nij',
                                                    self._theano_jacobian(xlevel_next_pred_var, xlevelp1_next_pred_var, X, U),
                                                    jaclevelp1_next_pred)
                jaclevels_next_pred['x%d_next_pred'%level] = jaclevel_next_pred
            return jaclevel_next_pred

        jaclevels = OrderedDict()
        xlevels = OrderedDict()
        for level in self.get_bilinear_levels():
            xlevels[level] = xlevels_all['x%d'%level]
            jaclevels['x%d_next_pred'%level] = response_map_jacobian(level)
        return jaclevels, xlevels

    def get_all_layers(self):
        layers = []
        for pred_layer in list(self.pred_layers.values()):
            layers.extend(lasagne.layers.get_all_layers(pred_layer))
        return lasagne.utils.unique(layers)

    def get_all_params(self, **tags):
        return lasagne.layers.get_all_params(self.pred_layers.values(), **tags)

    def get_all_param_values(self, **tags):
        return lasagne.layers.get_all_param_values(self.pred_layers.values(), **tags)

    def set_all_param_values(self, all_param_values, **tags):
        lasagne.layers.set_all_param_values(self.pred_layers.values(), all_param_values, **tags)

    @staticmethod
    def set_layer_param_tags(layer, **tags):
        for param_tags in layer.params.values():
            for tag, value in tags.items():
                if value:
                    param_tags.add(tag)
                else:
                    param_tags.discard(tag)

    def set_param_tags(self, param, **tags):
        for layer in self.get_all_layers():
            if param in layer.params:
                param_tags = layer.params[param]
                for tag, value in tags.items():
                    if value:
                        param_tags.add(tag)
                    else:
                        param_tags.discard(tag)

    def snapshot(self, iter_, snapshot_prefix):
        snapshot_prefix = snapshot_prefix or self.get_snapshot_prefix()
        snapshot_fname = snapshot_prefix + '_iter_%s.pkl'%str(iter_)
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
        try:
            self.set_all_param_values(all_param_values)
        except ValueError:
            # TODO: better way load parameters from other models
            new_params = self.get_all_params(transformation=True, **dict([('level%d'%self.get_bilinear_levels()[0], True)]))
            for param in new_params:
                self.set_param_tags(param, new=True)
            old_params = self.get_all_params(new=False)
            if len(old_params) != len(all_param_values):
                raise ValueError("mismatch: got %d values to set %d parameters" %
                                 (len(all_param_values), len(old_params)))
            for p, v in zip(old_params, all_param_values):
                if p.get_value().shape != v.shape:
                    pv = p.get_value()
                    pv[-v.shape[0]:, ...] = v
                    p.set_value(pv)
                else:
                    p.set_value(v)


class FcnActionCondEncoderOnlyTheanoNetFeaturePredictor(TheanoNetFeaturePredictor):
    def __init__(self, *args, **kwargs):
        super(FcnActionCondEncoderOnlyTheanoNetFeaturePredictor, self).__init__(*args, **kwargs)
        pretrained_file = kwargs.get('pretrained_file')
        if pretrained_file is None:
            fname = 'models/theano/fcn-32s-pascalcontext.pkl'
            print("Loading weights from pickle file", fname)
            param_dict = pickle.load(open(fname, 'rb'), encoding='latin1')
            all_params = OrderedDict((param.name, param) for param in self.get_all_params() if param.name is not None)
            for level in self.get_levels():
                if level == 0:
                    continue
                for i_conv in [1, 2]:
                    W, b = param_dict['conv%d_%d'%(level, i_conv)]
                    if level == 1 and i_conv == 1:
                        W = W[:, ::-1, :, :] * 255.0 / 2.0
                    param_W = all_params['x%d_conv%d.W'%(level, i_conv)]
                    param_b = all_params['x%d_conv%d.b'%(level, i_conv)]
                    param_W.set_value(W)
                    param_b.set_value(b)
                    self.set_param_tags(param_W, trainable=False)
                    self.set_param_tags(param_b, trainable=False)

    def train(self, *train_hdf5_fnames, val_hdf5_fname, solverstate_fname=None,
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
              snapshot=10000, #TODO: 1000
              snapshot_prefix=None,
              visualize_response_maps=False):
        # training loss
        param_l2_penalty = lasagne.regularization.regularize_network_params(self.pred_layers['x0_next_pred'], lasagne.regularization.l2)
        loss = self.loss + weight_decay * param_l2_penalty / 2.

        # training function
        start_time = time.time()
        print("Compiling training function...")
        train_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], loss)
        print("... finished in %.2f s"%(time.time() - start_time))

        validate = test_interval and val_hdf5_fname is not None
        if validate:
            # validation data
            val_minibatches = iterate_minibatches_indefinitely(val_hdf5_fname, data_names=['image_curr', 'vel', 'image_diff'],
                                                               batch_size=batch_size, shuffle=True)
            # validation loss
            test_loss = self.loss_deterministic + weight_decay * param_l2_penalty / 2.
            # validation function
            start_time = time.time()
            print("Compiling validation function...")
            val_fn = theano.function([self.X_var, self.U_var, self.X_diff_var], test_loss)
            print("... finished in %.2f s"%(time.time() - start_time))

        if solverstate_fname is not None and not solverstate_fname.endswith('.pkl'):
            solverstate_fname = self.get_snapshot_prefix() + '_iter_' + solverstate_fname + '.pkl'
        self.restore_solver(solverstate_fname)

        start_time = time.time()
        print("Aggregating matrices...")
        N = 0
        Alevels = {}
        Blevels = {}
        post_fit_all = {}
        batch_iter = 0
        for X, U, X_diff in iterate_minibatches_once(*train_hdf5_fnames, data_names=['image_curr', 'vel', 'image_diff'], batch_size=10000):
            print("batch %d"%batch_iter)
            N += X.shape[0]
            Xlevels = self.response_maps_from_input(X)
            Xlevels_next = self.response_maps_from_input(X + X_diff)
            for level in self.get_levels():
                Xlevel = Xlevels['x%d'%level]
                Xlevel_next = Xlevels_next['x%d'%level]
                c_dim = Xlevel.shape[1]
                if level not in Alevels:
                    Alevels[level] = [0] * c_dim
                    Blevels[level] = [0] * c_dim
                for channel in range(c_dim):
                    Xlevelchannel = Xlevel[:, channel, ...].reshape((Xlevel.shape[0], -1))
                    Xlevelchannel_next = Xlevel_next[:, channel, ...].reshape((Xlevel_next.shape[0], -1))
                    Xlevelchannel_diff = Xlevelchannel_next - Xlevelchannel
                    A, B, post_fit = bilinear.BilinearFunction.compute_solver_terms(Xlevelchannel, U, Xlevelchannel_diff)
                    Alevels[level][channel] += A
                    Blevels[level][channel] += B
                    if level not in post_fit_all:
                        post_fit_all[level] = post_fit
            batch_iter += 1
            if batch_iter == 10:
                break
        print("... finished in %2.f s"%(time.time() - start_time))

        start_time = time.time()
        print("Solving linear systems...")
        all_params = OrderedDict((param.name, param) for param in self.get_all_params() if param.name is not None)
        for level in self.get_levels():
            start_time_level = time.time()
            post_fit = post_fit_all[level]
            Qchannels = []
            Rchannels = []
            Schannels = []
            bchannels = []
            for A, B in zip(Alevels[level], Blevels[level]):
                A = A / (2. * N) + weight_decay * np.diag([1.]*(len(A)-1) + [0.]) # don't regularize bias, which is the last one
                B = B / (2. * N)
                Q, R, S, b = post_fit(np.linalg.solve(A, B))
                Qchannels.append(Q)
                Rchannels.append(R)
                Schannels.append(S)
                bchannels.append(b)
            all_params['x%d_diff_pred.Q'%level].set_value(np.asarray(Qchannels, dtype=theano.config.floatX))
            all_params['x%d_diff_pred.R'%level].set_value(np.asarray(Rchannels, dtype=theano.config.floatX))
            all_params['x%d_diff_pred.S'%level].set_value(np.asarray(Schannels, dtype=theano.config.floatX))
            all_params['x%d_diff_pred.b'%level].set_value(np.asarray(bchannels, dtype=theano.config.floatX))
            print("%2.f s"%(time.time() - start_time_level))
        print("... finished in %2.f s"%(time.time() - start_time))
        import IPython as ipy; ipy.embed()

        X, U, X_diff = next(iterate_minibatches_once(val_hdf5_fname, data_names=['image_curr', 'vel', 'image_diff'], batch_size=100))

#         self.snapshot('exact', snapshot_prefix) # TODO
        loss = train_fn(X, U, X_diff)
        print("    training loss = {:.6f}".format(float(loss)))
        test_loss = sum([val_fn(*next(val_minibatches)) for _ in range(test_iter)]) / test_iter
        print("    validation loss = {:.6f}".format(float(test_loss)))
        # visualize response maps of first image in batch
        if visualize_response_maps:
            self.visualize_response_maps(X[0], U[0], x_next=X[0]+X_diff[0])
        import IPython as ipy; ipy.embed();

