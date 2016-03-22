from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as nl
from lasagne.layers.dnn import dnn
from lasagne import init
from lasagne.utils import as_tuple
from predictor.predictor_theano import TheanoNetFeaturePredictor


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
        TheanoNetFeaturePredictor.set_layer_param_tags(self, **tags)


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
        TheanoNetFeaturePredictor.set_layer_param_tags(self, **tags)


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
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d'%level, True)]))
        elif bilinear_type == 'share':
            l_xlevel_diff_pred = BilinearLayer([l_xlevel, l_u], axis=2, name=name)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d'%level, True)]))
        elif bilinear_type == 'channelwise':
            l_xlevel_diff_pred = BilinearChannelwiseLayer([l_xlevel, l_u], name=name)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d'%level, True)]))
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
            TheanoNetFeaturePredictor.set_layer_param_tags(l_ud, transformation=True, **dict([('level%d'%level, True)]))
            l_xleveld = L.DenseLayer(l_xlevel, num_factor_weights, W=init.Uniform(1.0), nonlinearity=None)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xleveld, transformation=True, **dict([('level%d'%level, True)]))
            l_xleveld_diff_pred = L.ElemwiseMergeLayer([l_xleveld, l_ud], T.mul)
            l_xlevel_diff_pred_flat = L.DenseLayer(l_xleveld_diff_pred, np.prod(l_xlevel_shape[1:]), W=init.Uniform(1.0), nonlinearity=None)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_diff_pred_flat, transformation=True, **dict([('level%d'%level, True)]))
            l_xlevel_diff_pred = L.ReshapeLayer(l_xlevel_diff_pred_flat, ([0],) + l_xlevel_shape[1:], name=name)
        else:
            raise ValueError('bilinear_type should be either full, share channelwise or factor, given %s'%bilinear_type)
        return l_xlevel_diff_pred


def build_bilinear_net(input_shapes, X_var=None, U_var=None, X_diff_var=None, axis=1):
    x_shape, u_shape = input_shapes
    X_var = X_var or T.tensor4('X')
    U_var = U_var or T.matrix('U')
    X_diff_var = X_diff_var or T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var)
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var)

    l_x_diff_pred = BilinearLayer([l_x, l_u], axis=axis)
    l_x_next_pred = L.ElemwiseMergeLayer([l_x, l_x_diff_pred], T.add)
    l_y = L.flatten(l_x)
    l_y_diff_pred = L.flatten(l_x_diff_pred)

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'BilinearNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred), ('y', l_y), ('x0_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss

def build_action_cond_encoder_net(input_shapes, **kwargs):
    x_shape, u_shape = input_shapes

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x0 = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    l_x1 = L.Conv2DLayer(l_x0, 64, filter_size=6, stride=2, pad=0,
                         nonlinearity=nl.rectify,
                         name='x1')
    l_x2 = L.Conv2DLayer(l_x1, 64, filter_size=6, stride=2, pad=2,
                         nonlinearity=nl.rectify,
                         name='x2')
    l_x3 = L.Conv2DLayer(l_x2, 64, filter_size=6, stride=2, pad=2,
                         nonlinearity=nl.rectify,
                         name='x3')
    l_x3_shape = lasagne.layers.get_output_shape(l_x3)

    l_y4 = L.DenseLayer(l_x3, 1024, nonlinearity=nl.rectify, name='y')
    l_y4d = L.DenseLayer(l_y4, 2048, W=init.Uniform(1.0), nonlinearity=None)
    l_ud = L.DenseLayer(l_u, 2048, W=init.Uniform(0.1), nonlinearity=None)

    l_y4d_diff_pred = L.ElemwiseMergeLayer([l_y4d, l_ud], T.mul)
    l_y4_diff_pred = L.DenseLayer(l_y4d_diff_pred, 1024, W=init.Uniform(1.0), nonlinearity=None, name='y_diff_pred')

    l_y4_next_pred = L.ElemwiseMergeLayer([l_y4, l_y4_diff_pred], T.add, name='y_next_pred')

    l_y3_next_pred = L.DenseLayer(l_y4_next_pred, np.prod(l_x3_shape[1:]), nonlinearity=nl.rectify)
    l_x3_next_pred = L.ReshapeLayer(l_y3_next_pred, ([0],) + l_x3_shape[1:],
                                   name='x3_next_pred')

    l_x2_next_pred = Deconv2DLayer(l_x3_next_pred, 64, filter_size=6, stride=2, pad=2,
                                   nonlinearity=nl.rectify,
                                   name='x2_next_pred')
    l_x1_next_pred = Deconv2DLayer(l_x2_next_pred, 64, filter_size=6, stride=2, pad=2,
                                   nonlinearity=nl.rectify,
                                   name='x1_next_pred')
    l_x0_next_pred = Deconv2DLayer(l_x1_next_pred, 3, filter_size=6, stride=2, pad=0,
                                   nonlinearity=None,
                                   name='x0_next_pred')

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = loss_fn(X_next_var, lasagne.layers.get_output(l_x0_next_pred))

    net_name = 'ActionCondEncoderNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('x0_next_pred', l_x0_next_pred)])
    return net_name, input_vars, pred_layers, loss

def build_small_action_cond_encoder_net(input_shapes):
    x_shape, u_shape = input_shapes
    x2_c_dim = x1_c_dim = x_shape[0]
    x1_shape = (x1_c_dim, x_shape[1]//2, x_shape[2]//2)
    x2_shape = (x2_c_dim, x1_shape[1]//2, x1_shape[2]//2)
    y2_dim = 64
    X_var = T.tensor4('X')
    U_var = T.matrix('U')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    l_x1 = L.Conv2DLayer(l_x, x1_c_dim, filter_size=6, stride=2, pad=2,
                         W=init.Normal(std=0.01),
                         nonlinearity=nl.rectify)
    l_x2 = L.Conv2DLayer(l_x1, x2_c_dim, filter_size=6, stride=2, pad=2,
                         W=init.Normal(std=0.01),
                         nonlinearity=nl.rectify)

    l_y2 = L.DenseLayer(l_x2, y2_dim, nonlinearity=None, name='y')
    l_y2_diff_pred = BilinearLayer([l_y2, l_u], name='y_diff_pred')
    l_y2_next_pred = L.ElemwiseMergeLayer([l_y2, l_y2_diff_pred], T.add)
    l_x2_next_pred_flat = L.DenseLayer(l_y2_next_pred, np.prod(x2_shape), nonlinearity=None)
    l_x2_next_pred = L.ReshapeLayer(l_x2_next_pred_flat, ([0],) + x2_shape)

    l_x1_next_pred = Deconv2DLayer(l_x2_next_pred, x2_c_dim, filter_size=6, stride=2, pad=2,
                                   W=init.Normal(std=0.01),
                                   nonlinearity=nl.rectify)
    l_x_next_pred = Deconv2DLayer(l_x1_next_pred, x1_c_dim, filter_size=6, stride=2, pad=2,
                                  W=init.Normal(std=0.01),
                                  nonlinearity=nl.tanh,
                                  name='x_next_pred')

    X_next_pred_var = lasagne.layers.get_output(l_x_next_pred)
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var
    loss = ((X_next_var - X_next_pred_var) ** 2).mean(axis=0).sum() / 2.

    net_name = 'SmallActionCondEncoderNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y2_diff_pred), ('y', l_y2), ('x0_next_pred', l_x_next_pred)])
    return net_name, input_vars, pred_layers, loss

def build_fcn_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, num_downsample=0, bilinear_type='share', ladder_loss=True, batch_norm=False, concat=False, tanh=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    # preprocess
    l_x0 = l_x
    x0_shape = x_shape
    if num_downsample > 0:
        import cv2
        ds_kernel = cv2.getGaussianKernel(ksize=2, sigma=-1)
        ds_weight_filler = ds_kernel.dot(ds_kernel.T).astype(theano.config.floatX)
        ds_weight_filler = ds_weight_filler.reshape((1, 1, *ds_weight_filler.shape))
    for i_ds in range(num_downsample):
        l_x0_channels = []
        for channel in range(x0_shape[0]):
            l_x0_channel = L.SliceLayer(l_x0, indices=slice(channel, channel+1), axis=1)
            l_x0_channel = L.Conv2DLayer(l_x0_channel, 1, filter_size=2, stride=2, pad=0,
                                         W=ds_weight_filler,
                                         b=init.Constant(0.),
                                         nonlinearity=None)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_x0_channel, trainable=False)
            l_x0_channels.append(l_x0_channel)
        l_x0 = L.ConcatLayer(l_x0_channels, axis=1, name='x0' if i_ds == (num_downsample-1) else 'x0_ds%d'%(i_ds+1))
        x0_shape = (x0_shape[0], x0_shape[1]//2, x0_shape[2]//2)

    # encoding
    l_xlevels = OrderedDict()
    l_ylevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x0
        else:
            if level == 1:
                xlevelm1_c_dim = x0_shape[0]
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevelm1_c_dim
            l_xlevel_1 = L.Conv2DLayer(l_xlevels[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       nonlinearity=None,
                                       name='x%d_conv1'%level)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_1, encoding=True, **dict([('level%d'%level, True)]))
            if batch_norm:
                l_xlevel_1 = L.BatchNormLayer(l_xlevel_1, name='x%d_bn1'%level)
                TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_1, encoding=True, **dict([('level%d'%level, True)]))
            l_xlevel_1 = L.NonlinearityLayer(l_xlevel_1, nonlinearity=nl.rectify, name='x%d_1'%level)
            l_xlevel_2 = L.Conv2DLayer(l_xlevel_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       nonlinearity=None,
                                       name='x%d_conv2'%level)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_2, encoding=True, **dict([('level%d'%level, True)]))
            if batch_norm:
                l_xlevel_2 = L.BatchNormLayer(l_xlevel_2, name='x%d_bn2'%level)
                TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_2, encoding=True, **dict([('level%d'%level, True)]))
            l_xlevel_2 = L.NonlinearityLayer(l_xlevel_2, nonlinearity=nl.rectify, name='x%d_2'%level)
            l_xlevel = L.MaxPool2DLayer(l_xlevel_2, pool_size=2, stride=2, pad=0, name='x%d'%level)
        l_xlevels[level] = l_xlevel
        if level in levels:
            l_ylevels[level] = L.FlattenLayer(l_xlevel, name='y%d'%level)
    l_y = L.ConcatLayer(l_ylevels.values(), name='y')

    # bilinear
    l_xlevels_next_trans = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_trans = create_bilinear_layer(l_xlevel, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_trans'%level)
        l_xlevel_next_trans = L.ElemwiseMergeLayer([l_xlevel, l_xlevel_diff_trans], T.add, name='x%d_next_trans'%level)
        l_xlevels_next_trans[level] = l_xlevel_next_trans

    # decoding
    l_xlevels_next_pred = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            l_xlevel_next_pred = l_xlevels_next_trans[level]
        else:
            if level == 0:
                xlevel_c_dim = x1_c_dim
                xlevelm1_c_dim = x0_shape[0]
            elif level < levels[-1]-1:
                xlevel_c_dim = xlevelm1_c_dim
                xlevelm1_c_dim = xlevel_c_dim // 2
            l_xlevel_next_pred_2 = Deconv2DLayer(l_xlevels_next_pred[level+1], xlevel_c_dim, filter_size=2, stride=2, pad=0,
                                                 nonlinearity=None,
                                                 name='x%d_next_pred_2'%(level+1)) # TODO initialize with bilinear weights # TODO should rectify? # TODO: channel-wise (groups) # TODO: no bias term
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred_2, decoding=True, **dict([('level%d'%(level+1), True)]))
            l_xlevel_next_pred_1 = Deconv2DLayer(l_xlevel_next_pred_2, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                                 nonlinearity=None,
                                                 name='x%d_next_pred_deconv2'%(level+1))
            TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred_1, decoding=True, **dict([('level%d'%(level+1), True)]))
            if batch_norm:
                l_xlevel_next_pred_1 = L.BatchNormLayer(l_xlevel_next_pred_1, name='x%d_next_pred_bn2'%(level+1))
                TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred_1, decoding=True, **dict([('level%d'%(level+1), True)]))
            l_xlevel_next_pred_1 = L.NonlinearityLayer(l_xlevel_next_pred_1, nonlinearity=nl.rectify, name='x%d_next_pred_1'%(level+1))
            if concat:
                if level in l_xlevels_next_trans:
                    l_xlevel_next_pred_1 = L.ConcatLayer([l_xlevels_next_trans[level], l_xlevel_next_pred_1], name='x%d_next_pred_concat1'%(level+1))
                l_xlevel_next_pred = Deconv2DLayer(l_xlevel_next_pred_1, xlevelm1_c_dim, filter_size=3, stride=1, pad=1,
                                                   nonlinearity=None,
                                                   name='x%d_next_pred_deconv1'%(level+1))
                TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
                if batch_norm: # TODO batch normat level 0?
                    l_xlevel_next_pred = L.BatchNormLayer(l_xlevel_next_pred, name='x%d_next_pred_bn1'%(level+1))
                    TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
            else:
                l_xlevel_next_pred = Deconv2DLayer(l_xlevel_next_pred_1, xlevelm1_c_dim, filter_size=3, stride=1, pad=1,
                                                      nonlinearity=None,
                                                      name='x%d_next_pred_deconv1'%(level+1))
                TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
                if batch_norm:
                    l_xlevel_next_pred = L.BatchNormLayer(l_xlevel_next_pred, name='x%d_next_pred_bn1'%(level+1))
                    TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
                if level in l_xlevels_next_trans:
                    l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevels_next_trans[level], l_xlevel_next_pred], coeffs=[0.5, 0.5], name='x%d_next_pred_sum1'%(level+1)) # TODO weights should be learned
                    TheanoNetFeaturePredictor.set_layer_param_tags(l_xlevel_next_pred, decoding=True, **dict([('level%d'%(level+1), True)]))
            l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred, nonlinearity=nl.tanh if (level == 0 and tanh) else nl.rectify, name='x%d_next_pred'%level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred
        if level in levels:
            l_ylevels_diff_pred[level] = L.FlattenLayer(L.ElemwiseSumLayer([l_xlevel_next_pred, l_xlevels[level]], coeffs=[1.0, -1.0], name='x%d_diff_pred'%level), name='y%d_diff_pred'%level)
    l_y_diff_pred = L.ConcatLayer(list(l_ylevels_diff_pred.values())[::-1], name='y_diff_pred')
    l_y_next_pred = L.ElemwiseSumLayer([l_y, l_y_diff_pred], name='y_next_pred')

    l_x0_next_pred = l_xlevels_next_pred[0]

    # preprocess
    l_x0_next = l_x_next
    for i_ds in range(num_downsample):
        l_x0_next_channels = []
        for channel in range(x0_shape[0]):
            l_x0_next_channel = L.SliceLayer(l_x0_next, indices=slice(channel, channel+1), axis=1)
            l_x0_next_channel = L.Conv2DLayer(l_x0_next_channel, 1, filter_size=2, stride=2, pad=0,
                                              W=ds_weight_filler,
                                              b=init.Constant(0.),
                                              nonlinearity=None)
            TheanoNetFeaturePredictor.set_layer_param_tags(l_x0_next_channel, trainable=False)
            l_x0_next_channels.append(l_x0_next_channel)
        l_x0_next = L.ConcatLayer(l_x0_next_channels, axis=1, name='x0_next' if i_ds == (num_downsample-1) else 'x0_next_ds%d'%(i_ds+1))

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = loss_fn(lasagne.layers.get_output(l_x0_next),
                   lasagne.layers.get_output(l_x0_next_pred))
    loss_deterministic = loss_fn(lasagne.layers.get_output(l_x0_next, deterministic=True),
                                 lasagne.layers.get_output(l_x0_next_pred, deterministic=True))

    if ladder_loss:
        encoder_layers = OrderedDict((layer.name, layer) for layer in lasagne.layers.get_all_layers(l_xlevels[levels[-1]]) if layer.name is not None)
        # encoding of next image
        l_xlevels_next = OrderedDict()
        for level in range(levels[-1]+1):
            if level == 0:
                l_xlevel_next = l_x0_next
            else:
                if level == 1:
                    xlevelm1_c_dim = x0_shape[0]
                    xlevel_c_dim = x1_c_dim
                else:
                    xlevelm1_c_dim = xlevel_c_dim
                    xlevel_c_dim = 2 * xlevelm1_c_dim
                l_xlevel_next_1 = L.Conv2DLayer(l_xlevels_next[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                                W=encoder_layers['x%d_conv1'%level].W,
                                                b=encoder_layers['x%d_conv1'%level].b,
                                                nonlinearity=None,
                                                name='x%d_next_conv1'%level)
                if batch_norm: # TODO: share batch normalization variables?
                    l_xlevel_next_1 = L.BatchNormLayer(l_xlevel_next_1,
                                                       beta=encoder_layers['x%d_bn1'%level].beta,
                                                       gamma=encoder_layers['x%d_bn1'%level].gamma,
                                                       mean=encoder_layers['x%d_bn1'%level].mean,
                                                       inv_std=encoder_layers['x%d_bn1'%level].inv_std,
                                                       name='x%d_next_bn1'%level)
                l_xlevel_next_1 = L.NonlinearityLayer(l_xlevel_next_1, nonlinearity=nl.rectify, name='x%d_next_1'%level)
                l_xlevel_next_2 = L.Conv2DLayer(l_xlevel_next_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                                W=encoder_layers['x%d_conv2'%level].W,
                                                b=encoder_layers['x%d_conv2'%level].b,
                                                nonlinearity=None,
                                                name='x%d_next_conv2'%level)
                if batch_norm:
                    l_xlevel_next_2 = L.BatchNormLayer(l_xlevel_next_2,
                                                       beta=encoder_layers['x%d_bn2'%level].beta,
                                                       gamma=encoder_layers['x%d_bn2'%level].gamma,
                                                       mean=encoder_layers['x%d_bn2'%level].mean,
                                                       inv_std=encoder_layers['x%d_bn2'%level].inv_std,
                                                       name='x%d_next_bn2'%level)
                l_xlevel_next_2 = L.NonlinearityLayer(l_xlevel_next_2, nonlinearity=nl.rectify, name='x%d_next_2'%level)
                l_xlevel_next = L.MaxPool2DLayer(l_xlevel_next_2, pool_size=2, stride=2, pad=0, name='x%d_next'%level)
            l_xlevels_next[level] = l_xlevel_next

        for level in levels:
            if level == 0:
                continue
            loss += loss_fn(lasagne.layers.get_output(l_xlevels_next[level]),
                            lasagne.layers.get_output(l_xlevels_next_pred[level]))
            loss_deterministic += loss_fn(lasagne.layers.get_output(l_xlevels_next[level], deterministic=True),
                                          lasagne.layers.get_output(l_xlevels_next_pred[level], deterministic=True))

    net_name = 'FcnActionCondEncoderNet'
    net_name +='_levels' + ''.join([str(level) for level in levels])
    net_name += '_x1cdim' + str(x1_c_dim)
    net_name += '_numds' + str(num_downsample)
    net_name += '_bi' + bilinear_type
    net_name += '_ladder' + str(int(ladder_loss))
    net_name += '_bn' + str(int(batch_norm))
    net_name += '_concat' + str(int(concat))

    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred),
                               ('y', l_y),
                               ('y_next_pred', l_y_next_pred),
                               ('x0_next_pred', l_x0_next_pred),
                               ('x%d_next_pred'%levels[-1], l_xlevels_next_pred[levels[-1]])])
    if ladder_loss:
        pred_layers.update([('x%d_next'%levels[-1], l_xlevels_next[levels[-1]])])
    return net_name, input_vars, pred_layers, loss, loss_deterministic


def build_laplacian_fcn_action_cond_encoder_net(input_shapes, levels=None, x1_c_dim=16, bilinear_type='share', batch_norm=False, tanh=False):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('x')
    U_var = T.matrix('u')
    X_next_var = T.tensor4('x_next')

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    xlevels_c_dim = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            xlevels_c_dim[level] = x_shape[0]
        else:
            xlevels_c_dim[level] = x1_c_dim * 2**(level-1)

    # encoding
    l_xlevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x
        else:
            l_xlevel = VggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], batch_norm=batch_norm, name='x%d' % level)
        l_xlevels[level] = l_xlevel

    # bilinear
    l_xlevels_next_res = OrderedDict()
    l_xlevels_dec = OrderedDict()
    l_ylevels = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        if level == levels[-1]:
            l_xlevel_res = l_xlevel
        else:
            l_xlevels_dec[level] = VggDecodingLayer(l_xlevels[level+1], xlevels_c_dim[level],
                                                    batch_norm=batch_norm, name='x%d_dec' % level)
            l_xlevel_res = L.ElemwiseSumLayer([l_xlevel, l_xlevels_dec[level]], coeffs=[1.0, -1.0], name='x%d_res' % level)
        l_xlevel_diff_res = create_bilinear_layer(l_xlevel_res, l_u, level, bilinear_type=bilinear_type, name='x%d_diff_res' % level)
        l_xlevels_next_res[level] = L.ElemwiseSumLayer([l_xlevel_res, l_xlevel_diff_res], name='x%d_next_res' % level)
        l_ylevels[level] = L.FlattenLayer(l_xlevel_res, name='y%d' % level)
        l_ylevels_diff_pred[level] = L.FlattenLayer(l_xlevel_diff_res, name='y%d_diff_pred' % level)
    l_y = L.ConcatLayer(l_ylevels.values(), name='y')
    l_y_diff_pred = L.ConcatLayer(l_ylevels_diff_pred.values(), name='y_diff_pred')
    l_y_next_pred = L.ElemwiseSumLayer([l_y, l_y_diff_pred], name='y_next_pred')

    # decoding
    l_xlevels_next_pred = OrderedDict()
    for level in range(levels[-1]+1)[::-1]:
        if level == levels[-1]:
            l_xlevel_next_pred = l_xlevels_next_res[level]
        else:
            if level in l_xlevels_dec:
                params = l_xlevels_dec[level].get_params()
            else:
                params = tuple()
            l_xlevel_next_pred = VggDecodingLayer(l_xlevels_next_pred[level+1], xlevels_c_dim[level],
                                                  *params,
                                                  batch_norm=batch_norm, name='x%d_next_dec' % level)
            if level in l_xlevels_next_res:
                l_xlevel_next_pred = L.ElemwiseSumLayer([l_xlevels_next_res[level], l_xlevel_next_pred], name='x%d_next_dec_res_sum' % level)
            l_xlevel_next_pred = L.NonlinearityLayer(l_xlevel_next_pred, nonlinearity=nl.tanh if (level == 0 and tanh) else nl.rectify, name='x%d_next_pred'%level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred

    l_x_next_pred = l_xlevels_next_pred[0]

    # encoding of next image
    l_xlevels_next = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel_next = l_x_next
        else:
            l_xlevel_next = VggEncodingLayer(l_xlevels_next[level-1], xlevels_c_dim[level],
                                             *l_xlevels[level].get_params(),
                                             batch_norm=batch_norm, name='x%d_next'%level)
        l_xlevels_next[level] = l_xlevel_next

    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred),
                               ('y', l_y),
                               ('y_next_pred', l_y_next_pred),
                               ('x_next_pred', l_xlevels_next_pred[0]),
                               ('x0_next_pred', l_xlevels_next_pred[0]),
                               ('x_next', l_x_next),
                               ('x0_next', l_x_next),
                               ('x%d_next_pred' % levels[-1], l_xlevels_next_pred[levels[-1]]),
                               ('x%d_next' % levels[-1], l_xlevels_next[levels[-1]])])
    return pred_layers


def build_fcn_action_cond_encoder_only_net(input_shapes, levels=None, x1_c_dim=16, num_downsample=0, bilinear_type='share', ladder_loss=True, batch_norm=False, **kwargs):
    x_shape, u_shape = input_shapes
    assert len(x_shape) == 3
    assert len(u_shape) == 1
    levels = levels or [3]
    levels = sorted(set(levels))

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')
    l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

    # preprocess
    if num_downsample > 0:
        raise NotImplementedError
    l_x0 = l_x
    x0_shape = x_shape

    # encoding
    l_xlevels = OrderedDict()
    l_ylevels = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel = l_x0
        else:
            if level == 1:
                xlevelm1_c_dim = x0_shape[0]
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevelm1_c_dim
            l_xlevel_1 = L.Conv2DLayer(l_xlevels[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       nonlinearity=None,
                                       name='x%d_conv1'%level)
            if batch_norm:
                l_xlevel_1 = L.BatchNormLayer(l_xlevel_1, name='x%d_bn1'%level)
            l_xlevel_1 = L.NonlinearityLayer(l_xlevel_1, nonlinearity=nl.rectify, name='x%d_1'%level)
            l_xlevel_2 = L.Conv2DLayer(l_xlevel_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                       nonlinearity=None,
                                       name='x%d_conv2'%level)
            if batch_norm:
                l_xlevel_2 = L.BatchNormLayer(l_xlevel_2, name='x%d_bn2'%level)
            l_xlevel_2 = L.NonlinearityLayer(l_xlevel_2, nonlinearity=nl.rectify, name='x%d_2'%level)
            l_xlevel = L.MaxPool2DLayer(l_xlevel_2, pool_size=2, stride=2, pad=0, name='x%d'%level)
        l_xlevels[level] = l_xlevel
        l_ylevels[level] = L.FlattenLayer(l_xlevel, name='y%d'%level)
    l_y = L.ConcatLayer(l_ylevels.values(), name='y')

    # bilinear
    l_xlevels_next_pred = OrderedDict()
    l_ylevels_diff_pred = OrderedDict()
    for level in levels:
        l_xlevel = l_xlevels[level]
        l_xlevel_diff_pred = create_bilinear_layer(l_xlevel, l_u, bilinear_type=bilinear_type, name='x%d_diff_pred'%level)
        l_xlevel_next_pred = L.ElemwiseMergeLayer([l_xlevel, l_xlevel_diff_pred], T.add, name='x%d_next_pred'%level)
        l_xlevels_next_pred[level] = l_xlevel_next_pred
        l_ylevels_diff_pred[level] = L.FlattenLayer(L.ElemwiseSumLayer([l_xlevel_next_pred, l_xlevels[level]], coeffs=[1.0, -1.0], name='x%d_diff_pred'%level), name='y%d_diff_pred'%level)
    l_y_diff_pred = L.ConcatLayer(l_ylevels_diff_pred.values(), name='y_diff_pred')

    l_x0_next_pred = l_xlevels_next_pred[0]

    # preprocess
    l_x0_next = l_x_next

    encoder_layers = OrderedDict((layer.name, layer) for layer in lasagne.layers.get_all_layers(l_xlevels[levels[-1]]) if layer.name is not None)
    # encoding of next image
    l_xlevels_next = OrderedDict()
    for level in range(levels[-1]+1):
        if level == 0:
            l_xlevel_next = l_x0_next
        else:
            if level == 1:
                xlevelm1_c_dim = x0_shape[0]
                xlevel_c_dim = x1_c_dim
            else:
                xlevelm1_c_dim = xlevel_c_dim
                xlevel_c_dim = 2 * xlevelm1_c_dim
            l_xlevel_next_1 = L.Conv2DLayer(l_xlevels_next[level-1], xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                            W=encoder_layers['x%d_conv1'%level].W,
                                            b=encoder_layers['x%d_conv1'%level].b,
                                            nonlinearity=None,
                                            name='x%d_next_conv1'%level)
            if batch_norm: # TODO: share batch normalization variables?
                l_xlevel_next_1 = L.BatchNormLayer(l_xlevel_next_1,
                                                   beta=encoder_layers['x%d_bn1'%level].beta,
                                                   gamma=encoder_layers['x%d_bn1'%level].gamma,
                                                   mean=encoder_layers['x%d_bn1'%level].mean,
                                                   inv_std=encoder_layers['x%d_bn1'%level].inv_std,
                                                   name='x%d_next_bn1'%level)
            l_xlevel_next_1 = L.NonlinearityLayer(l_xlevel_next_1, nonlinearity=nl.rectify, name='x%d_next_1'%level)
            l_xlevel_next_2 = L.Conv2DLayer(l_xlevel_next_1, xlevel_c_dim, filter_size=3, stride=1, pad=1,
                                            W=encoder_layers['x%d_conv2'%level].W,
                                            b=encoder_layers['x%d_conv2'%level].b,
                                            nonlinearity=None,
                                            name='x%d_next_conv2'%level)
            if batch_norm:
                l_xlevel_next_2 = L.BatchNormLayer(l_xlevel_next_2,
                                                   beta=encoder_layers['x%d_bn2'%level].beta,
                                                   gamma=encoder_layers['x%d_bn2'%level].gamma,
                                                   mean=encoder_layers['x%d_bn2'%level].mean,
                                                   inv_std=encoder_layers['x%d_bn2'%level].inv_std,
                                                   name='x%d_next_bn2'%level)
            l_xlevel_next_2 = L.NonlinearityLayer(l_xlevel_next_2, nonlinearity=nl.rectify, name='x%d_next_2'%level)
            l_xlevel_next = L.MaxPool2DLayer(l_xlevel_next_2, pool_size=2, stride=2, pad=0, name='x%d_next'%level)
        l_xlevels_next[level] = l_xlevel_next

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = 0
    loss_deterministic = 0
    for level in levels:
        loss += loss_fn(lasagne.layers.get_output(l_xlevels_next[level]),
                        lasagne.layers.get_output(l_xlevels_next_pred[level]))
        loss_deterministic += loss_fn(lasagne.layers.get_output(l_xlevels_next[level], deterministic=True),
                                      lasagne.layers.get_output(l_xlevels_next_pred[level], deterministic=True))

    net_name = 'FcnActionCondEncoderOnlyNet'
    net_name +='_levels' + ''.join([str(level) for level in levels])
    net_name += '_x1cdim' + str(x1_c_dim)
    net_name += '_numds' + str(num_downsample)
    net_name += '_bi' + bilinear_type
    net_name += '_ladder' + str(int(ladder_loss))
    net_name += '_bn' + str(int(batch_norm))

    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('y_diff_pred', l_y_diff_pred),
                               ('y', l_y)])
    for level in set([0, *levels]):
        pred_layers.update([('x%d_next_pred'%level, l_xlevels_next_pred[level])])
    return net_name, input_vars, pred_layers, loss, loss_deterministic

def build_small_cifar10nin_net(input_shapes, **kwargs):
    x_shape, u_shape = input_shapes

    X_var = T.tensor4('X')
    U_var = T.matrix('U')
    X_diff_var = T.tensor4('X_diff')
    X_next_var = X_var + X_diff_var

    l_x0 = L.InputLayer(shape=(None,) + x_shape, input_var=X_var, name='x')
    l_u = L.InputLayer(shape=(None,) + u_shape, input_var=U_var, name='u')

    l_x1 = L.Conv2DLayer(l_x0,
                         num_filters=192,
                         filter_size=5,
                         pad=2,
                         flip_filters=False,
                         name='x1')
    l_x2 = L.Conv2DLayer(l_x1, num_filters=160, filter_size=1, flip_filters=False,
                         name='x2')
    l_x3 = L.Conv2DLayer(l_x2, num_filters=96, filter_size=1, flip_filters=False,
                         name='x3')
    l_x4 = L.MaxPool2DLayer(l_x3,
                            pool_size=3,
                            stride=2,
                            ignore_border=False,
                            name='x4')

    l_x4_diff_pred = BilinearLayer([l_x4, l_u], axis=2, name='x4_diff_pred')
    l_x4_next_pred = L.ElemwiseMergeLayer([l_x4, l_x4_diff_pred], T.add, name='x4_next_pred')

    l_x3_next_pred = Deconv2DLayer(l_x4_next_pred,
                                   num_filters=96,
                                   filter_size=3,
                                   stride=2,
                                   nonlinearity=None,
                                   name='x3_next_pred')
    l_x2_next_pred = Deconv2DLayer(l_x3_next_pred, num_filters=160, filter_size=1, flip_filters=False,
                                   name='x2_next_pred')
    l_x1_next_pred = Deconv2DLayer(l_x2_next_pred, num_filters=192, filter_size=1, flip_filters=False,
                                   name='x1_next_pred')
    l_x0_next_pred = Deconv2DLayer(l_x1_next_pred,
                                   num_filters=3,
                                   filter_size=5,
                                   pad=2,
                                   flip_filters=False,
                                   nonlinearity=None,
                                   name='x0_next_pred')

    loss_fn = lambda X, X_pred: ((X - X_pred) ** 2).mean(axis=0).sum() / 2.
    loss = loss_fn(X_next_var, lasagne.layers.get_output(l_x0_next_pred))

    net_name = 'SmallCifar10ninNet'
    input_vars = OrderedDict([(var.name, var) for var in [X_var, U_var, X_diff_var]])
    pred_layers = OrderedDict([('x0_next_pred', l_x0_next_pred)])
    return net_name, input_vars, pred_layers, loss
