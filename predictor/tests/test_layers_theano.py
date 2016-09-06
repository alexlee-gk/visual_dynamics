import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L
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
                                 untie_biases=True, groups=groups, nonlinearity=None)
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
