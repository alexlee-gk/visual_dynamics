from collections import OrderedDict

import lasagne.init as LI
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.nonlinearities as nl
import numpy as np
import theano

from visual_dynamics.predictors import layers_theano as LT


class VggConvNetwork(object):
    def __init__(self, input_shape, output_dim, hidden_sizes,
                 conv_filters, conv_filter_sizes, conv_strides, conv_pads,
                 encoding_levels=None, num_encoding_levels=5, xd_dim=32,
                 hidden_W_init=LI.GlorotUniform(), hidden_b_init=LI.Constant(0.),
                 output_W_init=LI.GlorotUniform(), output_b_init=LI.Constant(0.),
                 hidden_nonlinearity=LN.rectify, output_nonlinearity=None,
                 name=None, input_var=None):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        if len(input_shape) == 3:
            l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var)
            l_hid = L.reshape(l_in, ([0],) + input_shape)
        elif len(input_shape) == 2:
            l_in = L.InputLayer(shape=(None, np.prod(input_shape)), input_var=input_var)
            input_shape = (1,) + input_shape
            l_hid = L.reshape(l_in, ([0],) + input_shape)
        else:
            l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
            l_hid = l_in

        assert input_shape[0] % 2 == 0
        l_hid0 = L.SliceLayer(l_hid, slice(None, input_shape[0] // 2), axis=1)
        l_hid1 = L.SliceLayer(l_hid, slice(input_shape[0] // 2, None), axis=1)
        l_hids = [l_hid0, l_hid1]

        if encoding_levels is None:
            encoding_levels = [num_encoding_levels]
        else:
            assert max(encoding_levels) == num_encoding_levels

        xlevels_c_dim = OrderedDict(zip(range(num_encoding_levels + 1), [3, 64, 128, 256, 512, 512]))

        import h5py
        params_file = h5py.File("models/theano/vgg16_levelsall_model.h5", 'r')
        params_kwargs_list = []
        # encoding
        for ihid, l_hid in enumerate(l_hids):
            l_xlevels = OrderedDict()
            l_xdlevels = OrderedDict()  # downsampled version of l_xlevels at the resolution for servoing
            for level in range(num_encoding_levels + 1):
                if level == 0:
                    l_xlevel = l_hid
                elif level < 3:
                    l_xlevelm1 = l_xlevels[level - 1]
                    if level == 1:
                        # change from BGR to RGB and subtract mean pixel values
                        # (X - mean_pixel_bgr[None, :, None, None])[:, ::-1, :, :]
                        # X[:, ::-1, :, :] - mean_pixel_rgb[None, :, None, None]
                        if ihid == 0:
                            mean_pixel_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)
                            mean_pixel_rgb = mean_pixel_bgr[::-1]
                            W = np.eye(3)[::-1, :].reshape((3, 3, 1, 1)).astype(np.float32)
                            b = -mean_pixel_rgb
                            params_kwargs = dict(W=W, b=b)
                            for k, v in params_kwargs.items():
                                bcast = tuple(s == 1 for s in v.shape)
                                params_kwargs[k] = theano.shared(v, broadcastable=bcast)
                            params_kwargs_list.append(params_kwargs)
                        else:
                            params_kwargs = params_kwargs_list.pop(0)
                        l_xlevelm1 = L.Conv2DLayer(l_xlevelm1, num_filters=3, filter_size=1,
                                                   nonlinearity=nl.identity,
                                                   **params_kwargs)
                        l_xlevelm1.W.name = 'x0.W'
                        l_xlevelm1.params[l_xlevelm1.W].remove('trainable')
                        l_xlevelm1.b.name = 'x0.b'
                        l_xlevelm1.params[l_xlevelm1.b].remove('trainable')
                    if ihid == 0:
                        conv1_W = params_file['conv%d_1.W' % level][()]
                        conv1_b = params_file['conv%d_1.b' % level][()]
                        conv2_W = params_file['conv%d_2.W' % level][()]
                        conv2_b = params_file['conv%d_2.b' % level][()]
                        params_kwargs = dict(conv1_W=conv1_W, conv1_b=conv1_b,
                                             conv2_W=conv2_W, conv2_b=conv2_b)
                        for k, v in params_kwargs.items():
                            bcast = tuple(s == 1 for s in v.shape)
                            params_kwargs[k] = theano.shared(v, broadcastable=bcast)
                        params_kwargs_list.append(params_kwargs)
                    else:
                        params_kwargs = params_kwargs_list.pop(0)
                    l_xlevel = LT.VggEncodingLayer(l_xlevelm1, xlevels_c_dim[level], level=str(level),
                                                   **params_kwargs)
                else:
                    if ihid == 0:
                        conv1_W = params_file['conv%d_1.W' % level][()]
                        conv1_b = params_file['conv%d_1.b' % level][()]
                        conv2_W = params_file['conv%d_2.W' % level][()]
                        conv2_b = params_file['conv%d_2.b' % level][()]
                        conv3_W = params_file['conv%d_3.W' % level][()]
                        conv3_b = params_file['conv%d_3.b' % level][()]
                        params_kwargs = dict(conv1_W=conv1_W, conv1_b=conv1_b,
                                             conv2_W=conv2_W, conv2_b=conv2_b,
                                             conv3_W=conv3_W, conv3_b=conv3_b)
                        for k, v in params_kwargs.items():
                            bcast = tuple(s == 1 for s in v.shape)
                            params_kwargs[k] = theano.shared(v, broadcastable=bcast)
                        params_kwargs_list.append(params_kwargs)
                    else:
                        params_kwargs = params_kwargs_list.pop(0)
                    l_xlevel = LT.VggEncoding3Layer(l_xlevels[level - 1], xlevels_c_dim[level],
                                                    dilation=(2 ** (level - 3),) * 2, level=str(level),
                                                    **params_kwargs)
                # TODO:
                LT.set_layer_param_tags(l_xlevel, trainable=False)
                # downsample to servoing resolution
                xlevel_shape = L.get_output_shape(l_xlevel)
                xlevel_dim = xlevel_shape[-1]
                assert xlevel_shape[-2] == xlevel_dim
                scale_factor = xlevel_dim // xd_dim
                if scale_factor > 1:
                    l_xdlevel = LT.Downscale2DLayer(l_xlevel, scale_factor=scale_factor, name='x%dd' % level)
                elif scale_factor == 1:
                    l_xdlevel = l_xlevel
                else:
                    raise NotImplementedError
                if 0 < level < 3:
                    l_xlevel = L.MaxPool2DLayer(l_xlevel, pool_size=2, stride=2, pad=0,
                                                name='pool%d' % level)
                l_xlevels[level] = l_xlevel
                l_xdlevels[level] = l_xdlevel

            l_ylevels = OrderedDict()  # standarized version of l_xdlevels used as the feature for servoing
            for level in encoding_levels:
                if ihid == 0:
                    offset = params_file['y%d.offset' % level][()]
                    scale = params_file['y%d.scale' % level][()]
                    params_kwargs = dict(offset=offset, scale=scale)
                    for k, v in params_kwargs.items():
                        bcast = tuple(s == 1 for s in v.shape)
                        params_kwargs[k] = theano.shared(v, broadcastable=bcast)
                    params_kwargs_list.append(params_kwargs)
                else:
                    params_kwargs = params_kwargs_list.pop(0)
                l_ylevels[level] = LT.StandarizeLayer(l_xdlevels[level], name='y%d' % level,
                                                      **params_kwargs)

            l_hids[ihid] = L.ConcatLayer([l_ylevels[level] for level in encoding_levels], axis=1)
        assert not params_kwargs_list

        l_hid = L.ConcatLayer(l_hids, axis=1)

        for idx, conv_filter, filter_size, stride, pad in zip(
                range(len(conv_filters)),
                conv_filters,
                conv_filter_sizes,
                conv_strides,
                conv_pads,
        ):
            l_hid = L.Conv2DLayer(
                l_hid,
                num_filters=conv_filter,
                filter_size=filter_size,
                stride=(stride, stride),
                pad=pad,
                nonlinearity=hidden_nonlinearity,
                name="%sconv_hidden_%d" % (prefix, idx),
            )
        conv_out = l_hid
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                name="%shidden_%d" % (prefix, idx),
                W=hidden_W_init,
                b=hidden_b_init,
            )
        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,),
            W=output_W_init,
            b=output_b_init,
        )
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var
        self._conv_out = conv_out

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def conv_output_layer(self):
        return self._conv_out
