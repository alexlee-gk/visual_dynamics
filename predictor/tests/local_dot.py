import numpy as np
import time
import theano
import theano.tensor as T
import lasagne.layers as L
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
x_shape = (2, 5, 5)
# x_shape = (8, 32, 32)
filter_size = (3, 3)

X_var = T.tensor4('X')

l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
l_y = L.Conv2DLayer(l_x, x_shape[0], filter_size=filter_size, pad='same', nonlinearity=None, flip_filters=True, name='y')

X = np.random.random((batch_size,) + x_shape).astype(theano.config.floatX)
x = X[0]

tic()
Y_var = L.get_output(l_y)
Y_fn = theano.function([X_var], Y_var)
toc("Y compile")
tic()
Y = Y_fn(X)
toc("Y")

l_y1 = LT.LocallyConnected2DLayer(l_x, x_shape[0], filter_size=filter_size, pad='same', nonlinearity=None, flip_filters=True, name='y1')
l_y1.W.set_value(np.tile(l_y.W.get_value()[..., None, None], l_y1.get_W_shape()[-2:]))

tic()
Y1_var = L.get_output(l_y1)
Y1_fn = theano.function([X_var], Y1_var)
toc("Y1 compile")
tic()
Y1 = Y1_fn(X)
toc("Y1")

print(np.allclose(Y, Y1, atol=1e-7))

W = l_y.W.get_value()

h_size, w_size = x_shape[1:]
flip_filters = True

# y1 = np.zeros_like(y)
# if flip_filters:
#     W1 = np.tile(W[:, :, ::-1, ::-1][..., None, None], y.shape[-2:])
# else:
#     W1 = np.tile(W[..., None, None], y.shape[-2:])
#
# y1[:, 1:, 1:] += np.einsum('ihw,oihw->ohw', x[:, :-1, :-1], W1[..., 0, 0, 1:, 1:])
# y1[:, 1:, :] += np.einsum('ihw,oihw->ohw', x[:, :-1, :], W1[..., 0, 1, 1:, :])
# y1[:, 1:, :-1] += np.einsum('ihw,oihw->ohw', x[:, :-1, 1:], W1[..., 0, 2, 1:, :-1])
#
# y1[:, :, 1:] += np.einsum('ihw,oihw->ohw', x[:, :, :-1], W1[..., 1, 0, :, 1:])
# y1[:, :, :] += np.einsum('ihw,oihw->ohw', x[:, :, :], W1[..., 1, 1, :, :])
# y1[:, :, :-1] += np.einsum('ihw,oihw->ohw', x[:, :, 1:], W1[..., 1, 2, :, :-1])
#
# y1[:, :-1, 1:] += np.einsum('ihw,oihw->ohw', x[:, 1:, :-1], W1[..., 2, 0, :-1, 1:])
# y1[:, :-1, :] += np.einsum('ihw,oihw->ohw', x[:, 1:, :], W1[..., 2, 1,:-1, :])
# y1[:, :-1, :-1] += np.einsum('ihw,oihw->ohw', x[:, 1:, 1:], W1[..., 2, 2, :-1, :-1])

Y1 = np.zeros_like(Y)
# W1 = np.tile(W[..., None, None], l_y1.get_W_shape()[-2:])
W1 = l_y1.W.get_value()
for i in range(filter_size[0]):
    filter_h_ind = -i-1 if flip_filters else i
    ii = i - (filter_size[0] // 2)
    input_h_slice = slice(max(ii, 0), min(ii + h_size, h_size))
    output_h_slice = slice(max(-ii, 0), min(-ii + h_size, h_size))

    for j in range(filter_size[1]):
        filter_w_ind = -j-1 if flip_filters else j
        jj = j - (filter_size[1] // 2)
        input_w_slice = slice(max(jj, 0), min(jj + w_size, w_size))
        output_w_slice = slice(max(-jj, 0), min(-jj + w_size, w_size))

        if ii == jj == 0:
            print(output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, output_h_slice, output_w_slice)

        Y1[..., output_h_slice, output_w_slice] += \
            (X[:, None, :, input_h_slice, input_w_slice] * \
                W1[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice]).sum(axis=-3)

        # y1[:, output_h_slice, output_w_slice] += \
        #     np.einsum('ihw,oihw->ohw',
        #         x[:, input_h_slice, input_w_slice],
        #         W1[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice])

        # y1[:, output_h_slice, output_w_slice] += \
        #     np.tensordot(
        #         x[:, input_h_slice, input_w_slice],
        #         W1[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice],
        #         axes=(0, 1))

# channelwise
Y2 = np.zeros_like(Y)
W2 = np.tile(W[..., None, None], l_y1.get_W_shape()[-2:])
W2 = W2[:, 0, ...]
for i in range(filter_size[0]):
    filter_h_ind = -i-1 if flip_filters else i
    ii = i - (filter_size[0] // 2)
    input_h_slice = slice(max(ii, 0), min(ii + h_size, h_size))
    output_h_slice = slice(max(-ii, 0), min(-ii + h_size, h_size))

    for j in range(filter_size[1]):
        filter_w_ind = -j-1 if flip_filters else j
        jj = j - (filter_size[1] // 2)
        input_w_slice = slice(max(jj, 0), min(jj + w_size, w_size))
        output_w_slice = slice(max(-jj, 0), min(-jj + w_size, w_size))

        # if ii == jj == 0:
        #     print(output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, output_h_slice, output_w_slice)

        Y2[..., output_h_slice, output_w_slice] += \
            X[..., input_h_slice, input_w_slice] * \
                W2[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice]

Z2 = np.zeros((5, 5, 3, 3, 5, 5))
for i in range(filter_size[0]):
    filter_h_ind = -i-1 if flip_filters else i
    ii = i - (filter_size[0] // 2)
    input_h_slice = slice(max(ii, 0), min(ii + h_size, h_size))
    output_h_slice = slice(max(-ii, 0), min(-ii + h_size, h_size))

    for j in range(filter_size[1]):
        filter_w_ind = -j-1 if flip_filters else j
        jj = j - (filter_size[1] // 2)
        input_w_slice = slice(max(jj, 0), min(jj + w_size, w_size))
        output_w_slice = slice(max(-jj, 0), min(-jj + w_size, w_size))

        # if ii == jj == 0:
        #     print(output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, output_h_slice, output_w_slice)

        Z2[output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, output_h_slice, output_w_slice] += \
            X[0, 0, input_h_slice, input_w_slice][None, None, ...]

        Y2[..., output_h_slice, output_w_slice] += \
            X[..., input_h_slice, input_w_slice] * \
                W2[..., filter_h_ind, filter_w_ind, output_h_slice, output_w_slice]


channels, height, width = X.shape[1:]
kernel_h, kernel_w = W.shape[-2:]
dilation_h = 1
dilation_w = 1
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1

dil_kernel_h = (kernel_h - 1) * dilation_h + 1
dil_kernel_w = (kernel_w - 1) * dilation_w + 1
height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1
width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1
channels_col = channels * kernel_h * kernel_w

data_col = np.zeros((channels_col, height_col, width_col))
data_im = X[0]
for c in range(channels_col):
    w_offset = c % kernel_w
    h_offset = (c / kernel_w) % kernel_h
    c_im = c / kernel_h / kernel_w
    for h in range(height_col):
        h_pad = h * stride_h - pad_h + h_offset * dilation_h
        for w in range(width_col):
            w_pad = w * stride_w - pad_w + w_offset * dilation_w
            if 0 <= h_pad < height and 0 <= w_pad < width:
                data_col[c, h, w] = data_im[c_im, h_pad, w_pad]
            else:
                data_col[c, h, w] = 0.
print(np.allclose(np.einsum('ijk,i->jk', data_col, W[0][:, ::-1, ::-1].flatten()), Y[0][0]))

data_col1 = np.zeros((channels, kernel_h, kernel_w, height_col, width_col, height_col, width_col))
data_im = X[0]
for c_im in range(channels):
    for h_offset in range(kernel_h):
        for w_offset in range(kernel_w):
            for h in range(height_col):
                h_pad = h * stride_h - pad_h + h_offset * dilation_h
                for w in range(width_col):
                    w_pad = w * stride_w - pad_w + w_offset * dilation_w
                    if 0 <= h_pad < height and 0 <= w_pad < width:
                        data_col1[c_im, h_offset, w_offset, h, w, h, w] = data_im[c_im, h_pad, w_pad]
                    else:
                        data_col1[c_im, h_offset, w_offset, h, w, h, w] = 0.
print(np.allclose(np.einsum('ijk,i->jk', data_col1.reshape((-1, 5, 5)), W1[0][:, ::-1, ::-1].flatten()), Y1[0][0]))

data_col2 = np.zeros((kernel_h, kernel_w, height_col, width_col, height_col, width_col))
data_im = X[0]
c_im = 0
for h_offset in range(kernel_h):
    for w_offset in range(kernel_w):
        for h in range(height_col):
            h_pad = h * stride_h - pad_h + h_offset * dilation_h
            for w in range(width_col):
                w_pad = w * stride_w - pad_w + w_offset * dilation_w
                if 0 <= h_pad < height and 0 <= w_pad < width:
                    data_col2[h_offset, w_offset, h, w, h, w] = data_im[c_im, h_pad, w_pad]
                else:
                    data_col2[h_offset, w_offset, h, w, h, w] = 0.
print(np.allclose(np.einsum('ijk,i->jk', data_col2.reshape((-1, 5, 5)), W2[0][::-1, ::-1].flatten()), Y2[0][0]))


import IPython as ipy; ipy.embed()


def conv2d(X, W, flip_filters=True):
    # 2D convolution with no stride, 'same' padding and no dilation
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
    # 2D convolution with untied weights, no stride, 'same' padding and no dilation
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
    # 2D convolution with untied weights, no stride, 'same' padding and no dilation
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

(10)

Y3 = conv2d(X, W)
print(np.allclose(Y, Y3, atol=1e-7))

Y4 = locally_connected2d(X, W1)
print(np.allclose(Y, Y4, atol=1e-7))

Y5 = channelwise_locally_connected2d(X, W2)
print(np.allclose(Y2, Y5, atol=1e-7))

import IPython as ipy; ipy.embed()
