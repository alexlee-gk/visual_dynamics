import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
from lasagne import init
import predictor.layers_theano as LT
from nose2 import tools
from utils import tic, toc


@tools.params(((4, 32, 32), 4, 4),
              ((8, 32, 32), 4, 2),
              ((4, 32, 32), 4, 1),
              ((512, 32, 32), 512, 512)
              )
def test_group_conv(x_shape, num_filters, groups, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = LT.GroupConv2DLayer(l_x, num_filters, filter_size=3, stride=1, pad='same',
                                 untie_biases=True, groups=groups, nonlinearity=None,
                                 W=init.Uniform(), b=init.Uniform())
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    tic()
    conv = conv_fn(X)
    toc("conv time for x_shape=%r, num_filters=%r, groups=%r, batch_size=%r\n\t" %
        (x_shape, num_filters, groups, batch_size))

    l_scan_conv = LT.ScanGroupConv2DLayer(l_x, num_filters, filter_size=3, stride=1, pad='same',
                                          untie_biases=True, groups=groups, nonlinearity=None,
                                          W=l_conv.W, b=l_conv.b)
    scan_conv_var = L.get_output(l_scan_conv)
    scan_conv_fn = theano.function([X_var], scan_conv_var)
    tic()
    scan_conv = scan_conv_fn(X)
    toc("scan_conv time for x_shape=%r, num_filters=%r, groups=%r, batch_size=%r\n\t" %
        (x_shape, num_filters, groups, batch_size))

    assert np.allclose(conv, scan_conv)


@tools.params(((3, 32, 32), (6,)),
              ((512, 32, 32), (6,))
              )
def test_bilinear_group_conv(x_shape, u_shape, batch_size=2):
    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)
    U = np.random.random((batch_size,) + u_shape).astype(theano.config.floatX)

    l_xu_outer = LT.OuterProductLayer([l_x, l_u])
    l_x_diff_pred = LT.GroupConv2DLayer(l_xu_outer, x_shape[0], filter_size=5, stride=1, pad='same',
                                        untie_biases=True, groups=x_shape[0], nonlinearity=None,
                                        W=init.Uniform(), b=init.Uniform())
    X_diff_pred_var = L.get_output(l_x_diff_pred)
    X_diff_pred_fn = theano.function([X_var, U_var], X_diff_pred_var)
    X_diff_pred = X_diff_pred_fn(X, U)

    u_dim, = u_shape
    l_x_convs = []
    for i in range(u_dim + 1):
        l_x_conv = LT.GroupConv2DLayer(l_x, x_shape[0], filter_size=5, stride=1, pad='same',
                                       untie_biases=True, groups=x_shape[0], nonlinearity=None,
                                       W=l_x_diff_pred.W.get_value()[:, i:i+1],
                                       b=l_x_diff_pred.b.get_value() if i == u_dim else None)
        l_x_convs.append(l_x_conv)
    l_x_diff_pred_bw = LT.BatchwiseSumLayer(l_x_convs + [l_u])
    X_diff_pred_bw_var = L.get_output(l_x_diff_pred_bw)
    X_diff_pred_bw_fn = theano.function([X_var, U_var], X_diff_pred_bw_var)
    X_diff_pred_bw = X_diff_pred_bw_fn(X, U)

    assert np.allclose(X_diff_pred, X_diff_pred_bw, atol=1e-7)
