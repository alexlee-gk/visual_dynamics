import numpy as np
import theano
import theano.tensor as T
import lasagne.layers as L


class LocallyConnected2DLayer(L.Conv2DLayer):
    """Similar to Conv2DLayer except that the filter weights are unshared

    This implementation computes the output tensor by iterating over the filter
    weights and multiplying them with shifted versions of the input tensor.
    Assumes no stride, 'same' padding and no dilation.

    Keras has a more general implementation in here:
    https://github.com/fchollet/keras/blob/master/keras/layers/local.py
    Their implementation iterates over the values of the output tensor, which
    might be slower.
    """

    def __init__(self, incoming, num_filters, filter_size, channelwise=False, **kwargs):
        self.channelwise = channelwise
        super(LocallyConnected2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)
        if self.channelwise:
            assert num_filters == self.input_shape[1]

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        output_shape = self.get_output_shape_for(self.input_shape)
        if self.channelwise:
            num_output_channels = output_shape[1]
            assert num_input_channels == num_output_channels
            return (self.num_filters,) + self.filter_size + output_shape[-2:]
        else:
            return (self.num_filters, num_input_channels) + self.filter_size + output_shape[-2:]

    def convolve(self, input, **kwargs):
        if self.stride != (1, 1) or self.pad != 'same' or self.filter_dilation != (1, 1):
            raise NotImplementedError

        output_shape = self.get_output_shape_for(self.input_shape)

        # start with ii == jj == 0 case to initialize tensor
        i = self.filter_size[0] // 2
        j = self.filter_size[0] // 2
        filter_h_ind = -i-1 if self.flip_filters else i
        filter_w_ind = -j-1 if self.flip_filters else j
        if self.channelwise:
            conved = input * self.W[..., filter_h_ind, filter_w_ind, :, :]
        else:
            conved = (input[:, None, :, :, :] * self.W[..., filter_h_ind, filter_w_ind, :, :]).sum(axis=-3)

        for i in range(self.filter_size[0]):
            filter_h_ind = -i-1 if self.flip_filters else i
            ii = i - (self.filter_size[0] // 2)
            input_h_slice = slice(max(ii, 0), min(ii + output_shape[-2], output_shape[-2]))
            output_h_slice = slice(max(-ii, 0), min(-ii + output_shape[-2], output_shape[-2]))

            for j in range(self.filter_size[1]):
                filter_w_ind = -j-1 if self.flip_filters else j
                jj = j - (self.filter_size[1] // 2)
                input_w_slice = slice(max(jj, 0), min(jj + output_shape[-1], output_shape[-1]))
                output_w_slice = slice(max(-jj, 0), min(-jj + output_shape[-1], output_shape[-1]))
                # skip this case since it was done at the beginning
                if ii == jj == 0:
                    continue
                if self.channelwise:
                    conved = T.inc_subtensor(conved[..., output_h_slice, output_w_slice],
                                             input[..., input_h_slice, input_w_slice] *
                                             self.W[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice])
                else:
                    conved = T.inc_subtensor(conved[..., output_h_slice, output_w_slice],
                                             (input[:, None, :, input_h_slice, input_w_slice] *
                                              self.W[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice]).sum(axis=-3))
        return conved


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


def test_conv2d(x_shape, num_filters, filter_size, flip_filters, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = L.Conv2DLayer(l_x, num_filters, filter_size=filter_size,
                           stride=1, pad='same', flip_filters=flip_filters,
                           untie_biases=True, nonlinearity=None, b=None)
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    conv = conv_fn(X)
    loop_conv = conv2d(X, l_conv.W.get_value(), flip_filters=flip_filters)
    assert np.allclose(conv, loop_conv, atol=1e-7)


def test_locally_connected2d(x_shape, num_filters, filter_size, flip_filters, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = LocallyConnected2DLayer(l_x, num_filters, filter_size=filter_size,
                                     stride=1, pad='same', flip_filters=flip_filters,
                                     untie_biases=True, nonlinearity=None, b=None)
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    conv = conv_fn(X)
    loop_conv = locally_connected2d(X, l_conv.W.get_value(), flip_filters=flip_filters)
    assert np.allclose(conv, loop_conv, atol=1e-7)


def test_channelwise_locally_connected2d(x_shape, filter_size, flip_filters, batch_size=2):
    X_var = T.tensor4('X')
    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

    l_conv = LocallyConnected2DLayer(l_x, x_shape[0], filter_size=filter_size, channelwise=True,
                                        stride=1, pad='same', flip_filters=flip_filters,
                                        untie_biases=True, nonlinearity=None, b=None)
    conv_var = L.get_output(l_conv)
    conv_fn = theano.function([X_var], conv_var)
    conv = conv_fn(X)
    loop_conv = channelwise_locally_connected2d(X, l_conv.W.get_value(), flip_filters=flip_filters)
    assert np.allclose(conv, loop_conv, atol=1e-7)


def main():
    from utils import tic, toc
    batch_size = 32
    for x_shape in ((32, 32, 32), (32, 64, 64)):
        for filter_size in (3, 5):
            for num_filters in (32,):

                X_var = T.tensor4('X')
                l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
                X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)

                l_conv = LocallyConnected2DLayer(l_x, num_filters, filter_size=filter_size,
                                                 stride=1, pad='same', flip_filters=False,
                                                 untie_biases=True, nonlinearity=None, b=None)
                conv_var = L.get_output(l_conv)
                conv_fn = theano.function([X_var], conv_var)

                l_local = LocallyConnected2DLayer(l_x, num_filters, filter_size=filter_size,
                                                 stride=1, pad='same', flip_filters=False,
                                                 untie_biases=True, nonlinearity=None, b=None)
                local_var = L.get_output(l_local)
                local_fn = theano.function([X_var], local_var)

                tic()
                conv = conv_fn(X)
                conv_time = toc('conv')
                tic()
                local = local_fn(X)
                local_time = toc('local')
                print(local_time / conv_time)


# def main():
#     for x_shape in ((2, 5, 5), (4, 8, 8)):
#         for filter_size in (3, 5):
#             for flip_filters in (True, False):
#                 for num_filters in (2, 4):
#                     test_conv2d(x_shape, num_filters, filter_size, flip_filters)
#                     test_locally_connected2d(x_shape, num_filters, filter_size, flip_filters)
#                 test_channelwise_locally_connected2d(x_shape, filter_size, flip_filters)


if __name__ == '__main__':
    main()
