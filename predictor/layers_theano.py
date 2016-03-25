import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as nl
from lasagne.layers.dnn import dnn
from lasagne import init
from lasagne.utils import as_tuple


def set_layer_param_tags(layer, params=None, **tags):
    """
    If params is None, update tags of all parameters, else only update tags of parameters in params.
    """
    for param, param_tags in layer.params.items():
        if params is None or param in params:
            for tag, value in tags.items():
                if value:
                    param_tags.add(tag)
                else:
                    param_tags.discard(tag)


class ChannelwiseLayer(L.Layer):
    def __init__(self, incoming, channel_layer_class, name=None, **channel_layer_kwargs):
        super(ChannelwiseLayer, self).__init__(incoming, name=name)
        self.channel_layer_class = channel_layer_class
        self.channel_incomings = []
        self.channel_outcomings = []
        for channel in range(lasagne.layers.get_output_shape(incoming)[0]):
            channel_incoming = L.SliceLayer(incoming, indices=slice(channel, channel+1), axis=1,
                                            name='%s.%s%d' % (name, 'slice', channel) if name is not None else None)
            channel_outcoming = channel_layer_class(channel_incoming, **channel_layer_kwargs,
                                                    name='%s.%s%d' % (name, 'op', channel) if name is not None else None)
            self.channel_incomings.append(channel_incoming)
            self.channel_outcomings.append(channel_outcoming)
        self.outcoming = L.ConcatLayer(self.channel_outcomings, axis=1,
                                       name='%s.%s' % (name, 'concat') if name is not None else None)

    def get_output_shape_for(self, input_shape):
        channel_output_shapes = []
        for channel_incoming, channel_outcoming in zip(self.channel_incomings, self.channel_outcomings):
            channel_input_shape = channel_incoming.get_output_shape_for(input)
            channel_output_shape = channel_outcoming.get_output_shape_for(channel_input_shape)
            channel_output_shapes.append(channel_output_shape)
        output_shape = self.outcoming.get_output_shape_for(channel_output_shapes)
        return output_shape

    def get_output_for(self, input, **kwargs):
        channel_outputs = []
        for channel_incoming, channel_outcoming in zip(self.channel_incomings, self.channel_outcomings):
            channel_input= channel_incoming.get_output_for(input, **kwargs)
            channel_output = channel_outcoming.get_output_for(channel_input, **kwargs)
            channel_outputs.append(channel_output)
        output = self.outcoming.get_output_for(channel_outputs, **kwargs)
        return output


class CompositionLayer(L.Layer):
    def __init__(self, incoming, name=None):
        super(CompositionLayer, self).__init__(incoming, name=name)
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params.update(layer.params)
        return layer

    def get_output_shape_for(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            shape = layer.get_output_shape_for(shape)
        return shape

    def get_output_for(self, input, **kwargs):
        output = input
        for layer in self.layers:
            output = layer.get_output_for(output, **kwargs)
        return output


class VggEncodingLayer(CompositionLayer):
    def __init__(self, incoming, num_filters,
                 conv1_W=init.GlorotUniform(), conv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 conv2_W=init.GlorotUniform(), conv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 batch_norm=False, name=None,
                 **tags):
        super(VggEncodingLayer, self).__init__(incoming, name=name)
        layer = self.l_conv1 = self.add_layer(
            L.Conv2DLayer(incoming, num_filters, filter_size=3, stride=1, pad=1, nonlinearity=None,
                          W=conv1_W,
                          b=conv1_b,
                          name='%s.%s' % (name, 'conv1') if name is not None else None))
        if batch_norm:
            layer = self.l_bn1 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn1_beta,
                                 gamma=bn1_gamma,
                                 mean=bn1_mean,
                                 inv_std=bn1_inv_std,
                                 name='%s.%s' % (name, 'bn1') if name is not None else None))
        else:
            self.l_bn1 = None
        layer = self.l_relu1 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu1') if name is not None else None))

        layer = self.l_conv2 = self.add_layer(
            L.Conv2DLayer(layer, num_filters, filter_size=3, stride=1, pad=1, nonlinearity=None,
                          W=conv2_W,
                          b=conv2_b,
                          name='%s.%s' % (name, 'conv2') if name is not None else None))
        if batch_norm:
            layer = self.l_bn2 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn2_beta,
                                 gamma=bn2_gamma,
                                 mean=bn2_mean,
                                 inv_std=bn2_inv_std,
                                 name='%s.%s' % (name, 'bn2') if name is not None else None))
        else:
            self.l_bn2 = None
        layer = self.l_relu2 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu2') if name is not None else None))

        self.pool = self.add_layer(L.Pool2DLayer(layer, pool_size=2, stride=2, pad=0, mode='average_inc_pad',
                                                 name='%s.%s' % (name, 'pool')))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['encoding'] = tags.get('encoding', True)
        set_layer_param_tags(self, **tags)


class VggDecodingLayer(CompositionLayer):
    def __init__(self, incoming, num_filters,
                 deconv1_W=init.GlorotUniform(), deconv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 deconv2_W=init.GlorotUniform(), deconv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 batch_norm=False, last_nonlinearity=None, name=None,
                 **tags):
        super(VggDecodingLayer, self).__init__(incoming, name=name)
        layer = self.upscale = self.add_layer(
            L.Upscale2DLayer(incoming, scale_factor=2,
                             name='%s.%s' % (name, 'upscale') if name is not None else None))

        incoming_num_filters = lasagne.layers.get_output_shape(incoming)[1]
        layer = self.l_deconv2 = self.add_layer(
            Deconv2DLayer(layer, incoming_num_filters, filter_size=3, stride=1, pad=1, nonlinearity=None,
                          W=deconv2_W,
                          b=deconv2_b,
                          name='%s.%s' % (name, 'deconv2') if name is not None else None))
        if batch_norm:
            layer = self.l_bn2 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn2_beta,
                                 gamma=bn2_gamma,
                                 mean=bn2_mean,
                                 inv_std=bn2_inv_std,
                                 name='%s.%s' % (name, 'bn2') if name is not None else None))
        else:
            self.l_bn2 = None
        layer = self.l_relu2 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu2') if name is not None else None))

        layer = self.l_deconv1 = self.add_layer(
            Deconv2DLayer(layer, num_filters, filter_size=3, stride=1, pad=1, nonlinearity=None,
                          W=deconv1_W,
                          b=deconv1_b,
                          name='%s.%s' % (name, 'deconv1') if name is not None else None))
        if batch_norm:
            layer = self.l_bn1 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn1_beta,
                                 gamma=bn1_gamma,
                                 mean=bn1_mean,
                                 inv_std=bn1_inv_std,
                                 name='%s.%s' % (name, 'bn1') if name is not None else None))
        else:
            self.l_bn1 = None
        if last_nonlinearity is not None:
            self.l_nl1 = self.add_layer(
                L.NonlinearityLayer(layer, nonlinearity=last_nonlinearity,
                                    name='%s.%s' % (name, 'relu1') if name is not None else None))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['decoding'] = tags.get('decoding', True)
        set_layer_param_tags(self, **tags)


# deconv_length and Deconv2DLayer adapted from https://github.com/ebenolson/Lasagne/blob/deconv/lasagne/layers/dnn.py
def deconv_length(output_length, filter_size, stride, pad=0):
    if output_length is None:
        return None

    output_length = output_length * stride
    if pad == 'valid':
        input_length = output_length + filter_size - 1
    elif pad == 'full':
        input_length = output_length - filter_size + 1
    elif pad == 'same':
        input_length = output_length
    elif isinstance(pad, int):
        input_length = output_length - 2 * pad + filter_size - stride
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    return input_length


class Deconv2DLayer(L.Layer):
    def __init__(self, incoming, num_filters, filter_size, stride=(2, 2),
                 pad=0, untie_biases=False, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nl.rectify,
                 flip_filters=False, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nl.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.flip_filters = flip_filters

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad == 'full':
            self.pad = 'full'
        elif pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
            self.pad = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2],
                                self.output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = deconv_length(input_shape[2],
                                    self.filter_size[0],
                                    self.stride[0],
                                    pad[0])

        output_columns = deconv_length(input_shape[3],
                                       self.filter_size[1],
                                       self.stride[1],
                                       pad[1])

        return (batch_size, self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, **kwargs):
        # by default we assume 'cross', consistent with corrmm.
        conv_mode = 'conv' if self.flip_filters else 'cross'

        image = T.alloc(0., input.shape[0], *self.output_shape[1:])
        conved = dnn.dnn_conv(img=image,
                              kerns=self.W,
                              subsample=self.stride,
                              border_mode=self.pad,
                              conv_mode=conv_mode
                              )

        grad = T.grad(conved.sum(), wrt=image, known_grads={conved: input})

        if self.b is None:
            activation = grad
        elif self.untie_biases:
            activation = grad + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = grad + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(activation)


class BilinearLayer(L.MergeLayer):
    def __init__(self, incomings, axis=1, Q=init.Normal(std=0.001),
                 R=init.Normal(std=0.001), S=init.Normal(std=0.001),
                 b=init.Constant(0.), **kwargs):
        """
        axis: The first axis of Y to be lumped into a single bilinear model.
            The bilinear model are computed independently for each element wrt the preceding axes.
        """
        super(BilinearLayer, self).__init__(incomings, **kwargs)
        assert axis >= 1
        self.axis = axis

        self.y_shape, self.u_shape = [input_shape[1:] for input_shape in self.input_shapes]
        self.y_dim = int(np.prod(self.y_shape[self.axis-1:]))
        self.u_dim,  = self.u_shape

        self.Q = self.add_param(Q, (self.y_dim, self.y_dim, self.u_dim), name='Q')
        self.R = self.add_param(R, (self.y_dim, self.u_dim), name='R')
        self.S = self.add_param(S, (self.y_dim, self.y_dim), name='S')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.y_dim,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        Y_shape, U_shape = input_shapes
        assert Y_shape[0] == U_shape[0]
        return Y_shape

    def get_output_for(self, inputs, **kwargs):
        Y, U = inputs
        if Y.ndim > (self.axis + 1):
            Y = Y.flatten(self.axis + 1)
        assert Y.ndim == self.axis + 1

        outer_YU = Y.dimshuffle(list(range(Y.ndim)) + ['x']) * U.dimshuffle([0] + ['x']*self.axis + [1])
        bilinear = T.dot(outer_YU.reshape((-1, self.y_dim * self.u_dim)), self.Q.reshape((self.y_dim, self.y_dim * self.u_dim)).T)
        if self.axis > 1:
            bilinear = bilinear.reshape((-1,) + self.y_shape[:self.axis-1] + (self.y_dim,))
        linear_u = T.dot(U, self.R.T)
        if self.axis > 1:
            linear_u = linear_u.dimshuffle([0] + ['x']*(self.axis-1) + [1])
        linear_y = T.dot(Y, self.S.T)
        if self.axis > 1:
            linear_y = linear_y.reshape((-1,) + self.y_shape[:self.axis-1] + (self.y_dim,))
        activation = bilinear + linear_u + linear_y
        if self.b is not None:
            activation += self.b.dimshuffle(['x']*self.axis + [0])

        activation = activation.reshape((-1,) + self.y_shape)
        return activation

    def get_output_jacobian_for(self, inputs):
        Y, U = inputs
        assert Y.shape[1:] == self.y_shape
        assert U.shape[1:] == self.u_shape
        assert Y.shape[0] == U.shape[0]
        n_dim = Y.shape[0]
        c_dim = self.y_shape[0]
        if self.axis == 1:
            jac = np.einsum("kij,ni->nkj", self.Q.get_value(), Y.reshape((n_dim, self.y_dim)))
        elif self.axis == 2:
            jac = np.einsum("kij,nci->nckj", self.Q.get_value(), Y.reshape((n_dim, c_dim, self.y_dim)))
        else:
            raise NotImplementedError("Implemented for axis=1 and axis=2, axis=%d given"%self.axis)
        jac += self.R.get_value()
        jac = jac.reshape(n_dim, -1, self.u_dim)
        return jac


class BilinearChannelwiseLayer(L.MergeLayer):
    def __init__(self, incomings, Q=init.Normal(std=0.001),
                 R=init.Normal(std=0.001), S=init.Normal(std=0.001),
                 b=init.Constant(0.), **kwargs):
        super(BilinearChannelwiseLayer, self).__init__(incomings, **kwargs)

        self.y_shape, self.u_shape = [input_shape[1:] for input_shape in self.input_shapes]
        self.c_dim = self.y_shape[0]
        self.y_dim = int(np.prod(self.y_shape[1:]))
        self.u_dim,  = self.u_shape

        self.Q = self.add_param(Q, (self.c_dim, self.y_dim, self.y_dim, self.u_dim), name='Q')
        self.R = self.add_param(R, (self.c_dim, self.y_dim, self.u_dim), name='R')
        self.S = self.add_param(S, (self.c_dim, self.y_dim, self.y_dim), name='S')
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.c_dim, self.y_dim), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        Y_shape, U_shape = input_shapes
        assert Y_shape[0] == U_shape[0]
        return Y_shape

    def get_output_for(self, inputs, **kwargs):
        Y, U = inputs
        Y = Y.flatten(3)
        outer_YU = Y.dimshuffle([0, 1, 2, 'x']) * U.dimshuffle([0, 'x', 'x', 1])
        bilinear, _ = theano.scan(fn=lambda Q, outer_YU2: T.dot(outer_YU2, Q.T),
                                  sequences=[self.Q.reshape((self.c_dim, self.y_dim, self.y_dim * self.u_dim)),
                                             outer_YU.dimshuffle([1, 0, 2, 3]).reshape((self.c_dim, -1, self.y_dim * self.u_dim))])
        linear_u, _ = theano.scan(fn=lambda R, U2: T.dot(U2, R.T),
                                  sequences=[self.R],
                                  non_sequences=U)
        linear_y, _ = theano.scan(fn=lambda S, Y2: T.dot(Y2, S.T),
                                  sequences=[self.S, Y.dimshuffle([1, 0, 2])])
        activation = bilinear + linear_u + linear_y
        if self.b is not None:
            activation += self.b.dimshuffle([0, 'x', 1])
        activation = activation.dimshuffle([1, 0, 2]).reshape((-1,) + self.y_shape)
        return activation


def create_bilinear_layer(l_xlevel, l_u, level, bilinear_type='share', name=None):
        if bilinear_type == 'full':
            l_xlevel_diff_pred = BilinearLayer([l_xlevel, l_u], axis=1, name=name)
            set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d'%level, True)]))
        elif bilinear_type == 'share':
            l_xlevel_diff_pred = BilinearLayer([l_xlevel, l_u], axis=2, name=name)
            set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d'%level, True)]))
        elif bilinear_type == 'channelwise':
            l_xlevel_diff_pred = BilinearChannelwiseLayer([l_xlevel, l_u], name=name)
            set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d'%level, True)]))
#             l_xlevel_shape = lasagne.layers.get_output_shape(l_xlevel)
#             l_xlevel_diff_pred_channels = []
#             for channel in range(l_xlevel_shape[1]):
#                 l_xlevel_channel = L.SliceLayer(l_xlevel, indices=slice(channel, channel+1), axis=1)
#                 l_xlevel_diff_pred_channel = BilinearLayer([l_xlevel_channel, l_u], name='x%d_diff_pred_%d'%(level, channel))
#                 l_xlevel_diff_pred_channels.append(l_xlevel_diff_pred_channel)
#             l_xlevel_diff_pred = L.ConcatLayer(l_xlevel_diff_pred_channels, axis=1)
        elif bilinear_type == 'factor':
            l_xlevel_shape = lasagne.layers.get_output_shape(l_xlevel)
#             3 * 32**2 = 3072
#             64 * 16**2 = 16384
#             128 * 8**2 = 8192
#             256 * 4**2 = 4096
#             num_factor_weights = 2048
            num_factor_weights = min(np.prod(l_xlevel_shape[1:]) / 4, 4096)
            l_ud = L.DenseLayer(l_u, num_factor_weights, W=init.Uniform(0.1), nonlinearity=None)
            set_layer_param_tags(l_ud, transformation=True, **dict([('level%d'%level, True)]))
            l_xleveld = L.DenseLayer(l_xlevel, num_factor_weights, W=init.Uniform(1.0), nonlinearity=None)
            set_layer_param_tags(l_xleveld, transformation=True, **dict([('level%d'%level, True)]))
            l_xleveld_diff_pred = L.ElemwiseMergeLayer([l_xleveld, l_ud], T.mul)
            l_xlevel_diff_pred_flat = L.DenseLayer(l_xleveld_diff_pred, np.prod(l_xlevel_shape[1:]), W=init.Uniform(1.0), nonlinearity=None)
            set_layer_param_tags(l_xlevel_diff_pred_flat, transformation=True, **dict([('level%d'%level, True)]))
            l_xlevel_diff_pred = L.ReshapeLayer(l_xlevel_diff_pred_flat, ([0],) + l_xlevel_shape[1:], name=name)
        else:
            raise ValueError('bilinear_type should be either full, share channelwise or factor, given %s'%bilinear_type)
        return l_xlevel_diff_pred
