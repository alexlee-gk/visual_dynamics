import numpy as np
import h5py
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
from lasagne import init
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

def iterate_minibatches_indefinitely(*data, **kwargs):
    batch_size = kwargs.get('batch_size') or 1
    shuffle = kwargs.get('shuffle') or False
    N = len(data[0])
    for datum in data[1:]:
        assert len(datum) == N
    indices = []
    while True:
        if len(indices) < batch_size:
            new_indices = np.arange(N)
            if shuffle:
                np.random.shuffle(new_indices)
            indices.extend(new_indices)
        excerpt = indices[0:batch_size]
        out = tuple(datum[excerpt] for datum in data)
        del indices[0:batch_size]
        yield out

class BilinearLayer(L.MergeLayer):
    def __init__(self, incomings, M=init.GlorotUniform(),
                 N=init.GlorotUniform(), b=init.Constant(0.), **kwargs):
        super(BilinearLayer, self).__init__(incomings, **kwargs)

        self.y_shape, self.u_shape = [input_shape[1:] for input_shape in self.input_shapes]
        self.y_dim = int(np.prod(self.y_shape))
        self.u_dim,  = self.u_shape

        self.M = self.add_param(M, (self.y_dim, self.y_dim, self.u_dim), name='M')
        self.N = self.add_param(N, (self.y_dim, self.u_dim), name='N')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.y_dim,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        Y_shape, U_shape = input_shapes
        assert Y_shape[0] == U_shape[0]
        return (Y_shape[0], self.y_dim)

    def get_output_for(self, inputs, **kwargs):
        Y, U = inputs
        if Y.ndim > 2:
            Y = Y.flatten(2)

        outer_YU = Y[:, :, None] * U[:, None, :]
        activation = T.dot(outer_YU.flatten(2), self.M.reshape((self.y_dim, self.y_dim * self.u_dim)).T)
        activation = activation + T.dot(U, self.N.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return activation

class NetPredictor(predictor.FeaturePredictor):
    def __init__(self, network):
        self.network = network
        x_shape, u_shape = [input_shape[1:] for input_shape in network.input_shapes]
        predictor.FeaturePredictor.__init__(self, x_shape, u_shape)
        self.X_var, self.U_var = [input_layer.input_var for input_layer in network.input_layers]
        self.prediction_var = lasagne.layers.get_output(network, deterministic=True)
        self.prediction_fn = theano.function([self.X_var, self.U_var], self.prediction_var)
        self.jacobian_var = None
        self.jacobian_fn = None

    def train(self, train_hdf5_fname, val_hdf5_fname=None,
              batch_size=32,
              test_iter = 1000,
              test_interval = 1000,
              base_lr = 0.05,
              gamma = 0.9,
              stepsize = 1000,
              display = 20,
              max_iter=10000,
              momentum = 0.9,
              weight_decay=0.0005):

        # training data
        with h5py.File(train_hdf5_fname, 'r+') as f:
            X_train = f['image_curr'][:]
            U_train = f['vel'][:]
            X_diff_train = f['image_diff'][:]
            Y_diff_train = np.reshape(X_diff_train, (len(X_diff_train), -1))

        # Theano variable for predictions
        Y_diff_var = T.dmatrix('Y_diff')

        # training loss
        prediction_var = lasagne.layers.get_output(self.network) # deterministic=False
        loss = lasagne.objectives.squared_error(prediction_var, Y_diff_var).mean()
        param_l2_penalty = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        loss = loss + weight_decay * param_l2_penalty

        # training function
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        learning_rate = theano.shared(base_lr)
        updates = lasagne.updates.momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
        train_fn = theano.function([self.X_var, self.U_var, Y_diff_var], loss, updates=updates)

        validate = val_hdf5_fname is not None
        if validate:
            # validation data
            with h5py.File(val_hdf5_fname, 'r+') as f:
                X_val = f['image_curr'][:]
                U_val = f['vel'][:]
                X_diff_val = f['image_diff'][:]
                Y_diff_val = np.reshape(X_diff_val, (len(X_diff_val), -1))

            # validation loss
            test_loss = lasagne.objectives.squared_error(self.prediction_var, Y_diff_var).mean()
            test_loss = test_loss + weight_decay * param_l2_penalty

            # validation function
            val_fn = theano.function([self.X_var, self.U_var, Y_diff_var], test_loss)

        print("Starting training...")
        for iter_, batch in enumerate(iterate_minibatches_indefinitely(X_train, U_train, Y_diff_train,
                                                                      batch_size=batch_size, shuffle=True)):
            if iter_ < max_iter:
                if iter != 0 and iter_%stepsize == 0:
                    learning_rate.set_value(learning_rate.get_value() * gamma)
                X, U, Y_diff  = batch
                train_err = train_fn(X, U, Y_diff)

            if display and iter_%display == 0:
                print("Iteration {} of {}, lr {}".format(iter_, max_iter, learning_rate.get_value()))
                print("  training loss:\t\t{:.6f}".format(float(train_err)))

            if validate and iter_%test_interval == 0:
                val_err = 0
                val_batches = 0
                for val_iter, batch in enumerate(iterate_minibatches(X_val, U_val, Y_diff_val,
                                                                     batch_size=batch_size, shuffle=False)):
                    if val_iter >= test_iter:
                        break
                    X, U, Y_diff = batch
                    val_err += val_fn(X, U, Y_diff)
                    val_batches += 1
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            if iter_ >= max_iter: # break after printing information
                break

    def predict(self, X, U):
        return self.prediction_fn(X, U)

    def jacobian_control(self, X, U):
        if self.jacobian_var is None:
            self.jacobian_var = theano.gradient.jacobian(self.prediction_var[0, :], self.U_var)
            self.jacobian_fn = theano.function([self.X_var, self.U_var], self.jacobian_var)
        if U.ndim == 1:
            X, U = X[None, ...], U[None, :]
            jac = self.jacobian_fn(X, U)
            return np.squeeze(jac, 1)
        else:
            return np.asarray([self.jacobian_control(x, u) for x, u in zip(X, U)])

def build_bilinear_net(input_shapes):
    x_shape, u_shape = input_shapes
    X_var = T.dtensor4('X')
    U_var = T.dmatrix('U')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)
    l_y_diff = BilinearLayer([l_x, l_u], b=None)
    return l_y_diff
