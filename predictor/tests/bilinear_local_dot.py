from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import predictor.layers_theano as LT
from utils import tic, toc


batch_size = 64
# x_shape = (2, 5, 5)
# x_shape = (8, 32, 32)
x_shape = (3, 32, 32)
filter_size = (1, 1)
u_shape = (4,)
u_dim, = u_shape

X_var = T.tensor4('X')
U_var = T.matrix('U')
X_next_var = T.tensor4('X_next')

l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)
U = np.random.random((batch_size,) + u_shape).astype(theano.config.floatX)
X_next = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

num_encoding_levels = 5
xlevels_c_dim = OrderedDict(zip(range(num_encoding_levels + 1), [x_shape[0], 64, 128, 256, 512, 512]))
l_xlevels = OrderedDict()
for level in range(num_encoding_levels + 1):
    if level == 0:
        l_xlevel = l_x
    elif level < 3:
        l_xlevel = LT.DilatedVggEncodingLayer(l_xlevels[level - 1], xlevels_c_dim[level], name='x%d' % level)
    else:
        l_xlevel = LT.DilatedVggEncoding3Layer(l_xlevels[level - 1], xlevels_c_dim[level], name='x%d' % level)
    l_xlevels[level] = l_xlevel
# l_y = L.BatchNormLayer(l_xlevel, beta=None)
l_y = l_xlevel

l_y_shape = L.get_output_shape(l_y)
params = []
l_locals = []
for i in range(u_dim + 1):
    l_local = LT.LocallyConnected2DLayer(l_y, l_y_shape[1], filter_size=filter_size, stride=1, pad='same',
                                         untie_biases=True, channelwise=True, nonlinearity=None,
                                         name='local%d' % i)
    params += L.get_all_params(l_local, trainable=True)
    l_locals.append(l_local)
# l_y_next_pred = LT.BatchwiseSumLayer(l_locals + [l_u])

seq = T.arange(u_dim+1)
outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
Y_next_pred_var, scan_updates = theano.scan(fn=lambda i, sum_so_far: sum_so_far + L.get_output(l_locals[i], deterministic=True),
                                        outputs_info=outputs_info,
                                        sequences=seq)

# l_local = LT.LocallyConnected2DLayer(l_y, l_y_shape[1], filter_size=filter_size, stride=1, pad='same',
#                                      untie_biases=True, channelwise=True, nonlinearity=None)
# params = L.get_all_params(l_local, trainable=True)
# Y_next_pred_var = L.get_output(l_local, deterministic=False)
# for i in range(u_dim):
#     # l_local = LT.LocallyConnected2DLayer(l_y, l_y_shape[1], filter_size=filter_size, stride=1, pad='same',
#     #                                      untie_biases=True, channelwise=True, nonlinearity=None,
#     #                                      name='local%d' % i)
#     l_local = L.Conv2DLayer(l_y, l_y_shape[1], filter_size=filter_size, stride=1, pad='same',
#                                          untie_biases=True, nonlinearity=None,
#                                          name='local%d' % i)
#     params += L.get_all_params(l_local, trainable=True)
#     Y_next_pred_var += L.get_output(l_local, deterministic=False)



Y_next_var = L.get_output(l_y, inputs=X_next_var, deterministic=False)
# Y_next_pred_var = L.get_output(l_y_next_pred, deterministic=False)
loss_var = ((Y_next_var - Y_next_pred_var) ** 2).mean(axis=0).sum() / 2.
# params = L.get_all_params(l_y_next_pred, trainable=True)
updates = lasagne.updates.adam(loss_var, params, learning_rate=0.001)

tic()
train_fn = theano.function([X_var, U_var, X_next_var], loss_var, updates=updates, on_unused_input='ignore')
toc("train loss compile")
tic()
train_loss = train_fn(X, U, X_next)
toc("train loss")

import IPython as ipy; ipy.embed()
