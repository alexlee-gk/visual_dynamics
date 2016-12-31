import numpy as np
import time
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne.layers.cuda_convnet as CC
import predictor.layers_theano as LT

start_time = None


def tic():
    global start_time
    start_time = time.time()


def toc(name=None):
    if name:
        print(name, time.time() - start_time)
    else:
        print(time.time() - start_time)


batch_size = 10
x_shape = (512, 32, 32)
# x1_shape = (512, 16, 16)
# x2_shape = (512, 8, 8)
u_shape = (2,)
u_dim, = u_shape

X_var = T.tensor4('X')
U_var = T.matrix('U')

conv2d_kwargs = dict(filter_size=5, pad='same', nonlinearity=None, flip_filters=True)
Conv2DLayer = lambda incoming, num_filters, name=None: L.Conv2DLayer(incoming, num_filters, name=name, **conv2d_kwargs)

l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
# l_x1 = LT.Downscale2DLayer(l_x, 2)
# l_x2 = LT.Downscale2DLayer(l_x1, 2)
l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

l_y0 = Conv2DLayer(l_x, x_shape[0], name='y0')
l_y1 = Conv2DLayer(l_x, x_shape[0], name='y1')

# l_x1u = LT.OuterProductLayer([l_x1, l_u], name='x1u')
# l_y1 = Conv2DLayer(l_x1u, x1_shape[0], name='y1')
#
# l_x2u = LT.OuterProductLayer([l_x2, l_u], name='x2u')
# l_y2 = Conv2DLayer(l_x2u, x2_shape[0], name='y2')

X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)
U = np.random.random((batch_size,) + u_shape).astype(theano.config.floatX)
x = X[0]
u = U[0]

Y0_var = L.get_output(l_y0)
Y1_var = L.get_output(l_y1)

import IPython as ipy; ipy.embed()
Y_var = Y0_var * U_var[:, 0, None, None, None] + Y1_var * U_var[:, 1, None, None, None]
Y_fn = theano.function([X_var, U_var], Y_var)
tic()
Y = Y_fn(X, U)  # (10, 512, 32, 32)
toc("Y")

Y_grad_var = theano.grad((Y_var ** 2).mean(axis=0).sum(), L.get_all_params([l_y0, l_y1]))
Y_grad_fn = theano.function([X_var, U_var], Y_grad_var)
tic()
Y_grad = Y_grad_fn(X, U)  # (512, 3584, 5, 5), (512,)
toc("Y_grad")

j_0_var = theano.Rop(Y_var.flatten(), U_var, np.eye(u_dim)[0][None, :].astype(theano.config.floatX))
j_0_fn = theano.function([X_var, U_var], j_0_var)
tic()
j_0 = j_0_fn(x[None, :], u[None, :])  # (6, 524288)
toc("j_0")

j_var, _ = theano.scan(lambda eval_points, output_var, wrt_var: theano.gradient.Rop(output_var, wrt_var, eval_points),
                       sequences=np.eye(u_dim)[:, None, :],
                       non_sequences=[Y_var.flatten(), U_var])
j_fn = theano.function([X_var, U_var], j_var)
tic()
j = j_fn(x[None, :], u[None, :])
toc("j")

Y012_var = T.concatenate([Y_var.flatten(), Y1_var.flatten(), Y2_var.flatten()])
j012_var, _ = theano.scan(lambda eval_points, output_var, wrt_var: theano.gradient.Rop(output_var, wrt_var, eval_points),
                          sequences=np.eye(u_dim)[:, None, :],
                          non_sequences=[Y012_var, U_var])
j012_fn = theano.function([X_var, U_var], j012_var)
tic()
j012 = j012_fn(x[None, :], u[None, :])
toc("j012")




# # very slow
# j_var = theano.gradient.jacobian(Y_var.flatten(), U_var)
# j_fn = theano.function([X_var, U_var], j_var)
# tic()
# j = j_fn(x[None, :], u[None, :])
# toc("j")

# conv times
# ('Y', 0.12461996078491211)
# ('Y_grad', 0.32723498344421387)
# ('j_0', 0.037940025329589844)
# ('j', 0.11305403709411621)
# ('j012', 0.20110011100769043)

# ('Y', 0.15274906158447266)
# ('Y_grad', 0.30619001388549805)
# ('j_0', 0.03817605972290039)
# ('j', 0.1158449649810791)
# ('j012', 0.18796110153198242)

# group conv times
# ('Y', 0.06687712669372559)
# ('Y_grad', 0.13892889022827148)
# ('j_0', 0.06555581092834473)
# ('j', 0.3788340091705322)
# ('j012', 1.101274013519287)

# ('Y', 0.06748700141906738)
# ('Y_grad', 0.1392660140991211)
# ('j_0', 0.06485199928283691)
# ('j', 0.3681480884552002)
# ('j012', 0.9862430095672607)



import IPython as ipy; ipy.embed()
