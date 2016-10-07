import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
from lasagne import init
import predictor.layers_theano as LT
from nose2 import tools
from utils import tic, toc


def conv2d(X, W, flip_filters=True):
    # 2D convolution with no stride, 'same' padding, no dilation, no bias
    num_batch, input_channels, input_rows, input_cols = X.shape
    assert W.shape[1] == input_channels
    num_filters, input_channels, filter_rows, filter_cols = W.shape
    assert filter_rows % 2 == 1
    assert filter_cols % 2 == 1
    output_rows, output_cols = input_rows, input_cols
    Y = np.zeros((num_batch, num_filters, output_rows, output_cols))
    for b in range(num_batch):
        for f in range(num_filters):
            for c in range(input_channels):
                for i_out in range(output_rows):
                    for j_out in range(output_cols):
                        for i_filter in range(filter_rows):
                            i_in = i_out + i_filter - (filter_rows // 2)
                            if not (0 <= i_in < input_rows):
                                continue
                            for j_filter in range(filter_cols):
                                j_in = j_out + j_filter - (filter_cols // 2)
                                if not (0 <= j_in < input_cols):
                                    continue
                                if flip_filters:
                                    Y[b, f, i_out, j_out] += X[b, c, i_in, j_in] * W[f, c, -i_filter-1, -j_filter-1]
                                else:
                                    Y[b, f, i_out, j_out] += X[b, c, i_in, j_in] * W[f, c, i_filter, j_filter]
    return Y


def locally_connected2d(X, W, flip_filters=True):
    # 2D convolution with untied weights, no stride, 'same' padding, no dilation, no bias
    num_batch, input_channels, input_rows, input_cols = X.shape
    assert W.shape[1] == input_channels
    num_filters, input_channels, filter_rows, filter_cols, output_rows, output_cols = W.shape
    assert filter_rows % 2 == 1
    assert filter_cols % 2 == 1
    Y = np.zeros((num_batch, num_filters, output_rows, output_cols))
    for b in range(num_batch):
        for f in range(num_filters):
            for c in range(input_channels):
                for i_out in range(output_rows):
                    for j_out in range(output_cols):
                        for i_filter in range(filter_rows):
                            i_in = i_out + i_filter - (filter_rows // 2)
                            if not (0 <= i_in < input_rows):
                                continue
                            for j_filter in range(filter_cols):
                                j_in = j_out + j_filter - (filter_cols // 2)
                                if not (0 <= j_in < input_cols):
                                    continue
                                if flip_filters:
                                    Y[b, f, i_out, j_out] += X[b, c, i_in, j_in] * W[f, c, -i_filter-1, -j_filter-1, i_out, j_out]
                                else:
                                    Y[b, f, i_out, j_out] += X[b, c, i_in, j_in] * W[f, c, i_filter, j_filter, i_out, j_out]
    return Y


def channelwise_locally_connected2d(X, W, flip_filters=True):
    # 2D convolution with untied weights, no stride, 'same' padding, no dilation, no bias
    num_batch, input_channels, input_rows, input_cols = X.shape
    num_filters, filter_rows, filter_cols, output_rows, output_cols = W.shape
    assert input_channels == num_filters
    assert filter_rows % 2 == 1
    assert filter_cols % 2 == 1
    Y = np.zeros((num_batch, num_filters, output_rows, output_cols))
    for b in range(num_batch):
        for f in range(num_filters):
            for i_out in range(output_rows):
                for j_out in range(output_cols):
                    for i_filter in range(filter_rows):
                        i_in = i_out + i_filter - (filter_rows // 2)
                        if not (0 <= i_in < input_rows):
                            continue
                        for j_filter in range(filter_cols):
                            j_in = j_out + j_filter - (filter_cols // 2)
                            if not (0 <= j_in < input_cols):
                                continue
                            if flip_filters:
                                Y[b, f, i_out, j_out] += X[b, f, i_in, j_in] * W[f, -i_filter-1, -j_filter-1, i_out, j_out]
                            else:
                                Y[b, f, i_out, j_out] += X[b, f, i_in, j_in] * W[f, i_filter, j_filter, i_out, j_out]
    return Y


@tools.params(((2, 5, 5), 1, 3, False),
              ((2, 5, 5), 1, 3, True),
              ((2, 5, 5), 2, 3, False),
              ((2, 5, 5), 2, 3, True),
              ((2, 5, 5), 4, 3, False),
              ((2, 5, 5), 4, 3, True),
              ((4, 8, 8), 4, 5, False),
              ((4, 8, 8), 4, 5, True),
              )
def test_conv2d(x_shape, num_filters, filter_size, flip_filters, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = L.Conv2DLayer(l_x, num_filters, filter_size=filter_size, stride=1, pad='same',
                           flip_filters=flip_filters, untie_biases=True, nonlinearity=None, b=None)
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    tic()
    conv = conv_fn(X)
    toc("conv time for x_shape=%r, num_filters=%r, filter_size=%r, flip_filters=%r, batch_size=%r\n\t" %
        (x_shape, num_filters, filter_size, flip_filters, batch_size))

    tic()
    loop_conv = conv2d(X, l_conv.W.get_value(), flip_filters=flip_filters)
    toc("loop conv time for x_shape=%r, num_filters=%r, filter_size=%r, flip_filters=%r, batch_size=%r\n\t" %
        (x_shape, num_filters, filter_size, flip_filters, batch_size))

    assert np.allclose(conv, loop_conv, atol=1e-6)


@tools.params(((2, 5, 5), 1, 3, False),
              ((2, 5, 5), 1, 3, True),
              ((2, 5, 5), 2, 3, False),
              ((2, 5, 5), 2, 3, True),
              ((2, 5, 5), 4, 3, False),
              ((2, 5, 5), 4, 3, True),
              ((4, 8, 8), 4, 5, False),
              ((4, 8, 8), 4, 5, True),
              )
def test_locally_connected2d(x_shape, num_filters, filter_size, flip_filters, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = LT.LocallyConnected2DLayer(l_x, num_filters, filter_size=filter_size, stride=1, pad='same',
                                        flip_filters=flip_filters, untie_biases=True, nonlinearity=None, b=None)
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    tic()
    conv = conv_fn(X)
    toc("locally connected time for x_shape=%r, num_filters=%r, filter_size=%r, flip_filters=%r, batch_size=%r\n\t" %
        (x_shape, num_filters, filter_size, flip_filters, batch_size))

    tic()
    loop_conv = locally_connected2d(X, l_conv.W.get_value(), flip_filters=flip_filters)
    toc("loop locally connected time for x_shape=%r, num_filters=%r, filter_size=%r, flip_filters=%r, batch_size=%r\n\t" %
        (x_shape, num_filters, filter_size, flip_filters, batch_size))

    assert np.allclose(conv, loop_conv, atol=1e-6)


@tools.params(((2, 5, 5), 3, False),
              ((2, 5, 5), 3, True),
              ((4, 8, 8), 5, False),
              ((4, 8, 8), 5, True),
              )
def test_channelwise_locally_connected2d(x_shape, filter_size, flip_filters, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = LT.LocallyConnected2DLayer(l_x, x_shape[0], filter_size=filter_size, channelwise=True,
                                        stride=1, pad='same', flip_filters=flip_filters,
                                        untie_biases=True, nonlinearity=None, b=None)
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    tic()
    conv = conv_fn(X)
    toc("channelwise locally connected time for x_shape=%r, filter_size=%r, flip_filters=%r, batch_size=%r\n\t" %
        (x_shape, filter_size, flip_filters, batch_size))

    tic()
    loop_conv = channelwise_locally_connected2d(X, l_conv.W.get_value(), flip_filters=flip_filters)
    toc("loop channelwise locally connected time for x_shape=%r, filter_size=%r, flip_filters=%r, batch_size=%r\n\t" %
        (x_shape, filter_size, flip_filters, batch_size))

    assert np.allclose(conv, loop_conv, atol=1e-7)


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
