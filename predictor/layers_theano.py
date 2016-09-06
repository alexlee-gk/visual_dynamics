import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as nl
from lasagne.layers.dnn import dnn
from lasagne import init
from lasagne.utils import as_tuple
from lasagne.layers.conv import conv_output_length


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


from lasagne import nonlinearities
from lasagne.layers import MergeLayer, Layer
from lasagne.utils import unroll_scan
from lasagne.layers.recurrent import Gate


class Conv2DGate:
    def __init__(self, W_in=init.GlorotUniform(), W_hid=init.GlorotUniform(),
                 W_cell=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity


class Conv2DLSTMLayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)

    A long short-term memory (LSTM) layer.

    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by

    .. math ::

        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.

    Examples
    --------

    >>> import numpy as np
    >>> from lasagne.layers import *
    >>> from predictor.layers_theano import *
    >>> l_inp = InputLayer((None, 2, 3, 32, 32))
    >>> l_lstm = Conv2DLSTMLayer(l_inp, num_filters=7, filter_size=3, pad='same', unroll_scan=True, precompute_input=False)
    >>> # TODO: fix the errors from compiling
    >>> out = get_output(l_lstm)
    >>> out_fn = theano.function([l_inp.input_var], out)
    >>> input_ = np.random.random((5, 2, 3, 32, 32)).astype(np.float32)

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """

    def __init__(self, incoming, num_filters, filter_size,
                 stride=1, pad=0, untie_biases=False,
                 flip_filters=True, convolution=T.nnet.conv2d,
                 ingate=Conv2DGate(),
                 forgetgate=Conv2DGate(),
                 cell=Conv2DGate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Conv2DGate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, initial hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(Conv2DLSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.n = 2
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, self.n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, self.n, int)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        self.learn_init = learn_init
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_input_channels = input_shape[2]
        output_shape = self.get_output_shape_for([input_shape])
        output_size = output_shape[-2:]  # (output_rows, output_cols)

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            if self.untie_biases:
                biases_shape = (num_filters,) + output_size
            else:
                biases_shape = (num_filters,)
            return (self.add_param(gate.W_in, (num_filters, num_input_channels) + self.filter_size,
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_filters, num_filters) + self.filter_size,
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, biases_shape,
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_filters,) + output_size, name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_filters,) + output_size, name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_filters,) + output_size, name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_filters) + output_size,
                name="cell_init", trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, num_filters) + output_size,
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        output_rows, output_columns = tuple(conv_output_length(input, filter, stride, p)
                                            for input, filter, stride, p
                                            in zip(input_shape[3:], self.filter_size,
                                                   self.stride, pad))
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_filters, output_rows, output_columns
        # Otherwise, the shape will be (n_batch, n_steps, num_filters, output_rows, output_columns)
        else:
            return input_shape[0], input_shape[1], self.num_filters, output_rows, output_columns

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Input should be provided as (n_batch, n_time_steps, num_input_channels, input_rows, input_columns)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, num_input_channels, input_rows, input_columns)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        # Stack input weight tensors into a (4*num_filters, num_input_channels, filter_rows, filter_columns)
        # tensor, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=0)

        # Same for hidden weight tensors
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=0)

        # Stack biases into a (4*num_filters, 1, 1) or a (4*num_filters, output_rows, output_columns) tensor
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)
        bias_dims = list(range(b_stacked.ndim)) + ['x'] * (3 - b_stacked.ndim)
        b_stacked = b_stacked.dimshuffle(bias_dims)

        def convolve(input, W):
            W_shape = (4 * self.num_filters, None) + self.filter_size
            border_mode = 'half' if self.pad == 'same' else self.pad
            return self.convolution(input, W,
                                    None, W_shape,
                                    subsample=self.stride,
                                    border_mode=border_mode,
                                    filter_flip=self.flip_filters)

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, trailing dimensions...) to
            # (seq_len*batch_size, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len*num_batch,) + trailing_dims)
            input = convolve(input, W_in_stacked) + b_stacked

            # Reshape back to (seq_len, batch_size, trailing dimensions...)
            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # At each call to scan, input_n will be (n_time_steps, 4*num_filters, ...).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n * self.num_filters:(n + 1) * self.num_filters, :, :]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = convolve(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + convolve(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous * self.W_cell_to_ingate
                forgetgate += cell_previous * self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            if self.peepholes:
                outgate += cell * self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate * self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x', 'x', 'x')  # TODO: broadcastable dimensions
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        if not isinstance(self.cell_init, Layer):
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            dot_dims = (list(range(1, self.cell_init.ndim - 1)) +
                        [0, self.cell_init.ndim - 1])
            cell_init = T.dot(T.ones((num_batch, 1)),
                             self.cell_init.dimshuffle(dot_dims))

        if not isinstance(self.hid_init, Layer):
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        # The hidden-to-hidden weight tensor is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight tensors are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, num_filters, output_rows, output_columns))
            hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class PredNetLSTMLayer(MergeLayer):
    def __init__(self, incoming,
                 hidden_to_hidden,
                 error_to_hidden,
                 upper_hidden_to_hidden,
                 input_and_hidden_to_error,
                 error_to_upper_input,
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 nonlinearity_cell=nonlinearities.tanh,
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity=nonlinearities.tanh,
                 ingate_W_cell=init.GlorotUniform(),
                 forgetgate_W_cell=init.GlorotUniform(),
                 outgate_W_cell=init.GlorotUniform(),
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 error_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, initial hidden state or initial cell state was
        # provided.
        self.num_levels = len(hidden_to_hidden)

        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = []
        self.cell_init_incoming_index = []
        self.error_init_incoming_index = []
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if not isinstance(cell_init, (list, tuple)):
            cell_init = [cell_init] * self.num_levels
        if not isinstance(hid_init, (list, tuple)):
            hid_init = [hid_init] * self.num_levels
        if not isinstance(error_init, (list, tuple)):
            error_init = [error_init] * self.num_levels
        for level in range(self.num_levels):
            if isinstance(hid_init[level], Layer):
                incomings.append(hid_init[level])
                self.hid_init_incoming_index.append(len(incomings) - 1)
            if isinstance(cell_init[level], Layer):
                incomings.append(cell_init[level])
                self.cell_init_incoming_index.append(len(incomings) - 1)
            if isinstance(error_init[level], Layer):
                incomings.append(error_init[level])
                self.error_init_incoming_index.append(len(incomings) - 1)

        # Initialize parent layer
        super(PredNetLSTMLayer, self).__init__(incomings, **kwargs)

        self.hidden_to_hidden = hidden_to_hidden
        self.error_to_hidden = error_to_hidden
        self.upper_hidden_to_hidden = upper_hidden_to_hidden
        self.input_and_hidden_to_error = input_and_hidden_to_error
        self.error_to_upper_input = error_to_upper_input
        assert len(error_to_hidden) == self.num_levels
        assert len(upper_hidden_to_hidden) == (self.num_levels - 1)
        assert len(input_and_hidden_to_error) == self.num_levels
        assert len(error_to_upper_input) == (self.num_levels - 1)

        # If the provided nonlinearity is None, make it linear
        self.nonlinearity_ingate = nonlinearity_ingate or nonlinearities.identity
        self.nonlinearity_forgetgate = nonlinearity_forgetgate or nonlinearities.identity
        self.nonlinearity_cell = nonlinearity_cell or nonlinearities.identity
        self.nonlinearity_outgate = nonlinearity_outgate or nonlinearities.identity
        self.nonlinearity = nonlinearity or nonlinearities.identity

        self.learn_init = learn_init
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            if not isinstance(ingate_W_cell, (list, tuple)):
                ingate_W_cell = [ingate_W_cell] * self.num_levels
            if not isinstance(forgetgate_W_cell, (list, tuple)):
                forgetgate_W_cell = [forgetgate_W_cell] * self.num_levels
            if not isinstance(outgate_W_cell, (list, tuple)):
                outgate_W_cell = [outgate_W_cell] * self.num_levels
            self.W_cell_to_ingate = [None] * self.num_levels
            self.W_cell_to_forgetgate = [None] * self.num_levels
            self.W_cell_to_outgate = [None] * self.num_levels
            for level in range(self.num_levels):
                cell_shape = L.helper.get_output_shape(self.hidden_to_hidden[level])  # same shape as hid_shape
                cell_shape = (cell_shape[0], cell_shape[1] // 4) + cell_shape[2:]

                self.W_cell_to_ingate[level] = self.add_param(
                    ingate_W_cell[level], cell_shape[1:], name="W_cell_to_ingate_%d" % level)

                self.W_cell_to_forgetgate[level] = self.add_param(
                    forgetgate_W_cell[level], cell_shape[1:], name="W_cell_to_forgetgate_%d" % level)

                self.W_cell_to_outgate[level] = self.add_param(
                    outgate_W_cell[level], cell_shape[1:], name="W_cell_to_outgate_%d" % level)

        # Setup initial values for the cell, the hidden and error units
        self.cell_init = [None] * self.num_levels
        self.hid_init = [None] * self.num_levels
        self.error_init = [None] * self.num_levels
        for level in range(self.num_levels):
            hid_shape = L.helper.get_output_shape(self.hidden_to_hidden[level])
            hid_shape = (hid_shape[0], hid_shape[1] // 4) + hid_shape[2:]
            error_shape = L.helper.get_output_shape(self.input_and_hidden_to_error[level])

            if isinstance(cell_init[level], Layer):
                self.cell_init[level] = cell_init[level]
            else:
                self.cell_init[level] = self.add_param(
                    cell_init[level], (1,) + hid_shape[1:],
                    name="cell_init_%d" % level, trainable=learn_init, regularizable=False)

            if isinstance(hid_init[level], Layer):
                self.hid_init[level] = hid_init[level]
            else:
                self.hid_init[level] = self.add_param(
                    hid_init[level], (1,) + hid_shape[1:],
                    name="hid_init_%d" % level, trainable=learn_init, regularizable=False)

            if isinstance(error_init[level], Layer):
                self.error_init[level] = error_init[level]
            else:
                self.error_init[level] = self.add_param(
                    error_init[level], (1,) + error_shape[1:],
                    name="error_init_%d" % level, trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(PredNetLSTMLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += L.helper.get_all_params(self.hidden_to_hidden, **tags)
        params += L.helper.get_all_params(self.error_to_hidden, **tags)
        params += L.helper.get_all_params(self.upper_hidden_to_hidden, **tags)
        params += L.helper.get_all_params(self.input_and_hidden_to_error, **tags)
        params += L.helper.get_all_params(self.error_t_upper_input, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        # output_rows, output_columns = tuple(conv_output_length(input, filter, stride, p)
        #                                     for input, filter, stride, p
        #                                     in zip(input_shape[3:], self.filter_size,
        #                                            self.stride, pad))
        # # When only_return_final is true, the second (sequence step) dimension
        # # will be flattened
        # if self.only_return_final:
        #     return input_shape[0], self.num_filters, output_rows, output_columns
        # # Otherwise, the shape will be (n_batch, n_steps, num_filters, output_rows, output_columns)
        # else:
        #     return input_shape[0], input_shape[1], self.num_filters, output_rows, output_columns

        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return (input_shape[0],) + input_shape[2:]
        # Otherwise, the shape will be (n_batch, n_steps, num_filters, output_rows, output_columns)
        else:
            return input_shape

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        cell_init = [None] * self.num_levels
        hid_init = [None] * self.num_levels
        error_init = [None] * self.num_levels
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        hid_ind = 0
        for level in range(self.num_levels):
            if isinstance(self.hid_init[level], Layer):
                hid_init[level] = inputs[self.hid_init_incoming_index[hid_ind]]
                hid_ind += 1
        assert len(self.hid_init_incoming_index) == hid_ind
        cell_ind = 0
        for level in range(self.num_levels):
            if isinstance(self.cell_init[level], Layer):
                cell_init[level] = inputs[self.cell_init_incoming_index[cell_ind]]
                cell_ind += 1
        assert len(self.cell_init_incoming_index) == cell_ind
        error_ind = 0
        for level in range(self.num_levels):
            if isinstance(self.error_init[level], Layer):
                error_init[level] = inputs[self.error_init_incoming_index[error_ind]]
                error_ind += 1
        assert len(self.error_init_incoming_index) == error_ind

        # Input should be provided as (n_batch, n_time_steps, num_input_channels, input_rows, input_columns)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, num_input_channels, input_rows, input_columns)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]

        # # Stack input weight tensors into a (4*num_filters, num_input_channels, filter_rows, filter_columns)
        # # tensor, which speeds up computation
        # Wlevels_in_stacked = {}
        # for level in range(self.num_levels):
        #     Wlevels_in_stacked[level] = T.concatenate(
        #         [self.W_in_to_ingate, self.W_in_to_forgetgate,
        #          self.W_in_to_cell, self.W_in_to_outgate], axis=0)
        #
        # # Same for hidden weight tensors
        # Wlevels_hid_stacked = {}
        # for level in range(self.num_levels):
        #     Wlevels_hid_stacked[level] = T.concatenate(
        #         [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
        #          self.W_hid_to_cell, self.W_hid_to_outgate], axis=0)
        #
        # Wlevels_uphid_stacked = {}
        # for level in range(self.num_levels - 1):
        #     Wlevels_uphid_stacked[level] = T.concatenate(
        #         [self.W_uphid_to_ingate, self.W_uphid_to_forgetgate,
        #          self.W_uphid_to_cell, self.W_uphid_to_outgate], axis=0)
        #
        # # Stack biases into a (4*num_filters, 1, 1) or a (4*num_filters, output_rows, output_columns) tensor
        # blevels_stacked = {}
        # for level in range(self.num_levels):
        #     b_stacked = T.concatenate(
        #         [self.b_ingate, self.b_forgetgate,
        #          self.b_cell, self.b_outgate], axis=0)
        #     bias_dims = (*range(b_stacked.ndim), *['x'] * (3 - b_stacked.ndim))
        #     blevels_stacked[level] = b_stacked.dimshuffle(bias_dims)
        #
        # def convolve(input, W):
        #     W_shape = (4 * self.num_filters, None) + self.filter_size
        #     border_mode = 'half' if self.pad == 'same' else self.pad
        #     return self.convolution(input, W,
        #                             None, W_shape,
        #                             subsample=self.stride,
        #                             border_mode=border_mode,
        #                             filter_flip=self.flip_filters)
        #
        # if self.precompute_input:
        #     # Because the input is given for all time steps, we can precompute
        #     # the inputs to hidden before scanning. First we need to reshape
        #     # from (seq_len, batch_size, trailing dimensions...) to
        #     # (seq_len*batch_size, trailing dimensions...)
        #     # This strange use of a generator in a tuple was because
        #     # input.shape[2:] was raising a Theano error
        #     trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
        #     input = T.reshape(input, (seq_len * num_batch,) + trailing_dims)
        #     input = convolve(input, W_in_stacked) + b_stacked
        #
        #     # Reshape back to (seq_len, batch_size, trailing dimensions...)
        #     trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
        #     input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # At each call to scan, input_n will be (n_time_steps, 4*num_filters, ...).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n, num_filters):
            return x[:, n * num_filters:(n + 1) * num_filters, :, :]

        def slice_outputs(x, n):
            return x[n * self.num_levels:(n + 1) * self.num_levels]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, *args):
            cell_previous = slice_outputs(args, 0)
            hid_previous = slice_outputs(args, 1)
            error_previous = slice_outputs(args, 2)
            cell = [None] * self.num_levels
            hid = [None] * self.num_levels
            error = [None] * self.num_levels

            for level in range(self.num_levels)[::-1]:
                gates = L.helper.get_output(
                    self.hidden_to_hidden[level], hid_previous[level], **kwargs)
                gates += L.helper.get_output(
                    self.error_to_hidden[level], error_previous[level], **kwargs)
                if level < self.num_levels - 1:
                    gates += L.helper.get_output(
                        self.upper_hidden_to_hidden[level], hid[level + 1], **kwargs)

                # Clip gradients
                if self.grad_clipping:
                    gates = theano.gradient.grad_clip(
                        gates, -self.grad_clipping, self.grad_clipping)

                # Extract the pre-activation gate values
                hid_shape = L.helper.get_output_shape(self.hidden_to_hidden[level])
                num_filters = hid_shape[1] // 4
                ingate = slice_w(gates, 0, num_filters)
                forgetgate = slice_w(gates, 1, num_filters)
                cell_input = slice_w(gates, 2, num_filters)
                outgate = slice_w(gates, 3, num_filters)

                if self.peepholes:
                    # Compute peephole connections
                    ingate += cell_previous[level] * self.W_cell_to_ingate[level]
                    forgetgate += cell_previous[level] * self.W_cell_to_forgetgate[level]

                # Apply nonlinearities
                ingate = self.nonlinearity_ingate(ingate)
                forgetgate = self.nonlinearity_forgetgate(forgetgate)
                cell_input = self.nonlinearity_cell(cell_input)

                # Compute new cell value
                cell[level] = forgetgate * cell_previous[level] + ingate * cell_input

                if self.peepholes:
                    outgate += cell[level] * self.W_cell_to_outgate[level]
                outgate = self.nonlinearity_outgate(outgate)

                # Compute new hidden unit activation
                hid[level] = outgate * self.nonlinearity(cell[level])

            for level in range(self.num_levels):
                input_layers = [layer for layer in L.get_all_layers(self.input_and_hidden_to_error[level])
                                if isinstance(layer, L.InputLayer)]
                conv_layer, = [layer for layer in L.get_all_layers(self.input_and_hidden_to_error[level])
                               if isinstance(layer, L.Conv2DLayer)]
                if input_layers[0] == conv_layer.input_layer:
                    hid_layer, input_layer = input_layers
                else:
                    input_layer, hid_layer = input_layers
                error[level] = L.helper.get_output(
                    self.input_and_hidden_to_error[level], {input_layer: input_n, hid_layer:hid[level]}, **kwargs)
                if level < self.num_levels - 1:
                    input_n = L.helper.get_output(
                        self.error_to_upper_input[level], error[level], **kwargs)

            return cell + hid + error

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x', 'x', 'x')  # TODO: broadcastable dimensions
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        for level in range(self.num_levels):
            if not isinstance(self.cell_init[level], Layer):
                # The code below simply repeats self.cell_init num_batch times in
                # its first dimension.  Turns out using a dot product and a
                # dimshuffle is faster than T.repeat.
                dot_dims = (list(range(1, self.cell_init[level].ndim - 1)) +
                            [0, self.cell_init[level].ndim - 1])
                cell_init[level] = T.dot(T.ones((num_batch, 1)),
                                         self.cell_init[level].dimshuffle(dot_dims))

        for level in range(self.num_levels):
            if not isinstance(self.hid_init[level], Layer):
                # The code below simply repeats self.hid_init num_batch times in
                # its first dimension.  Turns out using a dot product and a
                # dimshuffle is faster than T.repeat.
                dot_dims = (list(range(1, self.hid_init[level].ndim - 1)) +
                            [0, self.hid_init[level].ndim - 1])
                hid_init[level] = T.dot(T.ones((num_batch, 1)),
                                        self.hid_init[level].dimshuffle(dot_dims))

        for level in range(self.num_levels):
            if not isinstance(self.error_init[level], Layer):
                # The code below simply repeats self.error_init num_batch times in
                # its first dimension.  Turns out using a dot product and a
                # dimshuffle is faster than T.repeat.
                dot_dims = (list(range(1, self.error_init[level].ndim - 1)) +
                            [0, self.error_init[level].ndim - 1])
                error_init[level] = T.dot(T.ones((num_batch, 1)),
                                          self.error_init[level].dimshuffle(dot_dims))

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = L.helper.get_all_params(self.hidden_to_hidden)
        non_seqs += L.helper.get_all_params(self.error_to_hidden)
        non_seqs += L.helper.get_all_params(self.upper_hidden_to_hidden)
        non_seqs += L.helper.get_all_params(self.input_and_hidden_to_error)
        non_seqs += L.helper.get_all_params(self.error_to_upper_input)

        # The "peephole" weight tensors are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += self.W_cell_to_ingate
            non_seqs += self.W_cell_to_forgetgate
            non_seqs += self.W_cell_to_outgate

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            outs = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=cell_init + hid_init + error_init,
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            outs = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init + hid_init + error_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]
        hid_out = slice_outputs(outs, 1)
        hid_out = hid_out[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, num_filters, output_rows, output_columns))
            hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class Downscale2DLayer(L.Layer):
    """
    2D downscaling layer

    Performs 2D downscaling over the two trailing axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(Downscale2DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            assert output_shape[2] % self.scale_factor[0] == 0
            output_shape[2] //= self.scale_factor[0]
        if output_shape[3] is not None:
            assert output_shape[3] % self.scale_factor[1] == 0
            output_shape[3] //= self.scale_factor[1]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b = self.scale_factor
        input_shape = input.shape
        downscaled = input.reshape((input_shape[0],
                                    input_shape[1],
                                    input_shape[2] // a,
                                    a,
                                    input_shape[3] // b,
                                    b)).mean(axis=(-3, -1))
        return downscaled


class CaffeConv2DLayer(L.Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, group=1, stride=(1, 1), pad="valid", untie_biases=False, W=init.Uniform(), b=init.Constant(0.), nonlinearity=nl.rectify,convolution=T.nnet.conv2d, **kwargs):
        super(CaffeConv2DLayer, self).__init__(incoming, num_filters, filter_size, stride=stride, pad=pad, untie_biases=untie_biases, W=W, b=b, nonlinearity=nonlinearity,convolution=convolution, **kwargs)
        self.group = group
        assert self.num_filters % self.group == 0

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        assert num_input_channels % self.group == 0
        return (self.num_filters // self.group, num_input_channels // self.group, self.filter_size[0], self.filter_size[1])

    def get_output_for(self, input, input_shape=None, *args, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = (self.input_shape[0], self.input_shape[1] // self.group, self.input_shape[2], self.input_shape[3])

        filter_shape = self.get_W_shape()

        if self.pad in ['valid', 'full']:
            tensors = []
            for g in range(self.group):
                inp = input[:, g*input_shape[1]:(g+1)*input_shape[1], :, :]
                # TODO: should it be self.group instead of 2?
                tensors.append(self.convolution(inp,
                                                self.W[g*(self.num_filters // 2):(g+1)*(self.num_filters // 2), :, :, :],
                                                subsample=self.strides,
                                                image_shape=input_shape,
                                                filter_shape=filter_shape,
                                                border_mode=self.pad))
            conved = T.concatenate(tensors, axis=1)
        elif self.pad == 'same':
            tensors = []
            for g in range(self.group):
                inp = input[:, g*input_shape[1]:(g+1)*input_shape[1], :, :]
                tensors.append(self.convolution(inp,
                                                self.W[g*(self.num_filters // 2):(g+1)*(self.num_filters // 2), :, :, :],
                                                subsample=self.strides,
                                                image_shape=input_shape,
                                                filter_shape=filter_shape,
                                                border_mode='full'))
            conved = T.concatenate(tensors, axis=1)
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input_shape[2] + shift_x, shift_y:input_shape[3] + shift_y]
        else:
            raise RuntimeError("Invalid border mode: '%s'" % self.pad)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)


class ScanGroupConv2DLayer(L.Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, groups=1,
                 W=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nl.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, filter_dilation=(1, 1), **kwargs):
        L.Layer.__init__(self, incoming, **kwargs)
        assert num_filters % groups == 0
        self.groups = groups
        # super(ScanGroupConv2DLayer, self).__init__(incoming, num_filters, filter_size,
        #                                        stride=stride, pad=pad,
        #                                        untie_biases=untie_biases,
        #                                        W=W, b=b,
        #                                        nonlinearity=nonlinearity,
        #                                        flip_filters=flip_filters,
        #                                        convolution=convolution,
        #                                        filter_dilation=filter_dilation,
        #                                        **kwargs)

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        n = 2

        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n + 2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.filter_dilation = as_tuple(filter_dilation, n, int)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        W_shape = self.get_W_shape()
        W_shape = (W_shape[0] // self.groups, W_shape[1], W_shape[2], W_shape[3])

        # self.W = []
        # for g in range(self.groups):
        #     self.W.append(self.add_param(W, W_shape, name="W%d" % g))
        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        assert num_input_channels % self.groups == 0
        return (self.num_filters, num_input_channels // self.groups, self.filter_size[0], self.filter_size[1])

    def get_output_for(self, input, *args, **kwargs):
        input_shape = (self.input_shape[0], self.input_shape[1] // self.groups, self.input_shape[2], self.input_shape[3])
        W_shape = self.get_W_shape()
        W_shape = (W_shape[0] // self.groups, W_shape[1], W_shape[2], W_shape[3])

        # tensors = []
        # for g in range(self.groups):
        #     inp = input[:, g * input_shape[1]:(g + 1) * input_shape[1], :, :]
        #     W = self.W[g * W_shape[0]:(g + 1) * W_shape[0], :, :, :]
        #     tensors.append(self.convolution(inp, W,
        #                                     input_shape, W_shape,
        #                                     subsample=self.stride,
        #                                     filter_dilation=self.filter_dilation,
        #                                     border_mode=border_mode,
        #                                     filter_flip=self.flip_filters))
        # conved = T.concatenate(tensors, axis=1)

        # W = T.concatenate(self.W, axis=0)
        # TODO: verify implementation of group convolution (e.g. test channelwise)?
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved, updates = theano.scan(fn=lambda g:
                                             self.convolution(input[:, g * input_shape[1]:(g + 1) * input_shape[1], :, :],
                                                              self.W[g * W_shape[0]:(g + 1) * W_shape[0], :, :, :],
                                                              input_shape, W_shape,
                                                              subsample=self.stride,
                                                              filter_dilation=self.filter_dilation,
                                                              border_mode=border_mode,
                                                              filter_flip=self.flip_filters),
                                      outputs_info=None,
                                      sequences=[T.arange(self.groups)])
        conved = conved.dimshuffle([1, 0, 2, 3, 4])
        conved = conved.reshape((conved.shape[0], conved.shape[1] * conved.shape[2],
                                 conved.shape[3], conved.shape[4]))

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)


class GroupConv2DLayer(L.Conv2DLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False, groups=1,
                 W=init.Uniform(), b=init.Constant(0.),
                 nonlinearity=nl.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, filter_dilation=(1, 1), **kwargs):
        assert num_filters % groups == 0
        self.groups = groups
        super(GroupConv2DLayer, self).__init__(incoming, num_filters, filter_size,
                                               stride=stride, pad=pad,
                                               untie_biases=untie_biases,
                                               W=W, b=b,
                                               nonlinearity=nonlinearity,
                                               flip_filters=flip_filters,
                                               convolution=convolution,
                                               filter_dilation=filter_dilation,
                                               **kwargs)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        assert num_input_channels % self.groups == 0
        return (self.num_filters, num_input_channels // self.groups, self.filter_size[0], self.filter_size[1])

    def convolve(self, input, **kwargs):
        W_shape = self.get_W_shape()
        W_shape = (W_shape[0], W_shape[1] * self.groups, W_shape[2], W_shape[3])

        # the following is the symbolic equivalent of
        # W = np.zeros(W_shape)
        # for g in range(self.groups):
        #     input_slice = slice(g * self.input_shape[1] // self.groups,
        #                         (g + 1) * self.input_shape[1] // self.groups)
        #     output_slice = slice(g * self.num_filters // self.groups, (g + 1) * self.num_filters // self.groups)
        #     W[output_slice, input_slice, :, :] = self.W.get_value()[output_slice, :, :, :]

        # repeat W across the second dimension and then mask the terms outside the block diagonals
        mask = np.zeros(W_shape[:2]).astype(theano.config.floatX)
        for g in range(self.groups):
            input_slice = slice(g * self.input_shape[1] // self.groups,
                                (g + 1) * self.input_shape[1] // self.groups)
            output_slice = slice(g * self.num_filters // self.groups, (g + 1) * self.num_filters // self.groups)
            mask[output_slice, input_slice] = 1

        # elementwise multiplication along broadcasted dimensions is faster than T.tile
        # the following is equivalent to
        # W = T.tile(self.W, (1, self.groups, 1, 1)) * mask[:, :, None, None]
        W = (T.ones((1, self.groups, 1, 1, 1)) * self.W[:, None, :, :, :]).reshape(W_shape) * mask[:, :, None, None]

        # similarly for T.repeat but we don't use that in here
        # W = T.repeat(self.W, self.groups, axis=1) * mask[:, :, None, None]
        # W = (T.ones((1, 1, self.groups, 1, 1)) * self.W[:, :, None, :, :]).reshape(W_shape) * mask[:, :, None, None]

        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, W,
                                  self.input_shape, W_shape,
                                  subsample=self.stride,
                                  filter_dilation=self.filter_dilation,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved


class CrossConv2DLayer(L.MergeLayer):
    def __init__(self, incomings, num_filters, filter_size, stride=1,
                 pad=0, untie_biases=False,
                 b=init.Constant(0.),
                 nonlinearity=nl.rectify, flip_filters=True,
                 convolution=T.nnet.conv2d, filter_dilation=1, **kwargs):
        super(CrossConv2DLayer, self).__init__(incomings, **kwargs)
        self.input_shape, W_shape = self.input_shapes
        assert self.input_shape[0] == W_shape[0]

        if nonlinearity is None:
            self.nonlinearity = nl.identity
        else:
            self.nonlinearity = nonlinearity

        n = 2
        if n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n + 2, n))
        self.n = n
        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, n, int)
        self.flip_filters = flip_filters
        self.stride = as_tuple(stride, n, int)
        self.filter_dilation = as_tuple(filter_dilation, n, int)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = as_tuple(0, n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, n, int)

        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels) + self.filter_size

    def get_output_shape_for(self, input_shapes):
        input_shape, W_shape = input_shapes
        assert input_shape[0] == W_shape[0]
        assert W_shape[1:] == self.get_W_shape()
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def get_output_for(self, inputs, *args, **kwargs):
        input_, W = inputs
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved, updates = theano.scan(fn=lambda input_, W:
                                             self.convolution(input_[None, :, :, :],
                                                              W,
                                                              (1,) + self.input_shape[1:], self.get_W_shape(),
                                                              subsample=self.stride,
                                                              filter_dilation=self.filter_dilation,
                                                              border_mode=border_mode,
                                                              filter_flip=self.flip_filters).dimshuffle(*range(4)[1:]),
                                      outputs_info=None,
                                      sequences=[input_, W])

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)
        return self.nonlinearity(activation)


class ChannelwiseLayer(L.Layer):
    def __init__(self, incoming, channel_layer_class, name=None, **channel_layer_kwargs):
        super(ChannelwiseLayer, self).__init__(incoming, name=name)
        self.channel_layer_class = channel_layer_class
        self.channel_incomings = []
        self.channel_outcomings = []
        for channel in range(lasagne.layers.get_output_shape(incoming)[0]):
            channel_incoming = L.SliceLayer(incoming, indices=slice(channel, channel+1), axis=1,
                                            name='%s.%s%d' % (name, 'slice', channel) if name is not None else None)
            channel_outcoming = channel_layer_class(channel_incoming,
                                                    name='%s.%s%d' % (name, 'op', channel) if name is not None else None,
                                                    **channel_layer_kwargs)
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
    def __init__(self, incoming, layers=None, name=None):
        super(CompositionLayer, self).__init__(incoming, name=name)
        self.layers = []
        for layer in layers or []:
            self.add_layer(layer)

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

    def get_param_kwargs(self, **tags):
        params = self.get_params(**tags)
        return dict([(self.param_keys[param], param) for param in params])


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

        self.param_keys = dict()
        for layer, base_name in [(self.l_conv1, 'conv1'), (self.l_conv2, 'conv2')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })

class VggEncoding3Layer(CompositionLayer):
    def __init__(self, incoming, num_filters,
                 conv1_W=init.GlorotUniform(), conv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 conv2_W=init.GlorotUniform(), conv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 conv3_W=init.GlorotUniform(), conv3_b=init.Constant(0.),
                 bn3_beta=init.Constant(0.), bn3_gamma=init.Constant(1.),
                 bn3_mean=init.Constant(0.), bn3_inv_std=init.Constant(1.),
                 batch_norm=False, name=None,
                 **tags):
        super(VggEncoding3Layer, self).__init__(incoming, name=name)
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

        layer = self.l_conv3 = self.add_layer(
            L.Conv2DLayer(layer, num_filters, filter_size=3, stride=1, pad=1, nonlinearity=None,
                          W=conv3_W,
                          b=conv3_b,
                          name='%s.%s' % (name, 'conv3') if name is not None else None))
        if batch_norm:
            layer = self.l_bn3 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn3_beta,
                                 gamma=bn3_gamma,
                                 mean=bn3_mean,
                                 inv_std=bn3_inv_std,
                                 name='%s.%s' % (name, 'bn3') if name is not None else None))
        else:
            self.l_bn3 = None
        layer = self.l_relu3 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu3') if name is not None else None))

        self.pool = self.add_layer(L.Pool2DLayer(layer, pool_size=2, stride=2, pad=0, mode='average_inc_pad',
                                                 name='%s.%s' % (name, 'pool')))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['encoding'] = tags.get('encoding', True)
        set_layer_param_tags(self, **tags)

        self.param_keys = dict()
        for layer, base_name in [(self.l_conv1, 'conv1'), (self.l_conv2, 'conv2'), (self.l_conv3, 'conv3')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2'), (self.l_bn3, 'bn3')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })


class VggDecodingLayer(CompositionLayer):
    def __init__(self, incoming, num_filters,
                 deconv2_W=init.GlorotUniform(), deconv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 deconv1_W=init.GlorotUniform(), deconv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
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

        self.param_keys = dict()
        for layer, base_name in [(self.l_deconv1, 'deconv1'), (self.l_deconv2, 'deconv2')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })


class DilatedVggEncodingLayer(CompositionLayer):
    def __init__(self, incoming, num_filters, filter_size=3, dilation=(2, 2),
                 conv1_W=init.GlorotUniform(), conv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 conv2_W=init.GlorotUniform(), conv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 batch_norm=False, name=None,
                 **tags):
        super(DilatedVggEncodingLayer, self).__init__(incoming, name=name)
        layer = self.l_conv1 = self.add_layer(
            L.Conv2DLayer(incoming, num_filters, filter_size=filter_size, stride=1, pad='same', nonlinearity=None,
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

        # layer = self.l_pad2 = self.add_layer(
        #     L.PadLayer(layer, (filter_size - 1) * dilation // 2))  # 'same' padding
        # layer = self.l_conv2 = self.add_layer(
        #     L.DilatedConv2DLayer(layer, num_filters, filter_size=filter_size, dilation=dilation, nonlinearity=None,
        #                          W=conv2_W,
        #                          b=conv2_b,
        #                          name='%s.%s' % (name, 'conv2') if name is not None else None))
        layer = self.l_conv2 = self.add_layer(
            L.Conv2DLayer(layer, num_filters, filter_size=filter_size, filter_dilation=dilation, pad='same', nonlinearity=None,
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
        self.l_relu2 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu2') if name is not None else None))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['encoding'] = tags.get('encoding', True)
        set_layer_param_tags(self, **tags)

        self.param_keys = dict()
        for layer, base_name in [(self.l_conv1, 'conv1'), (self.l_conv2, 'conv2')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })


class DilatedVggEncoding3Layer(CompositionLayer):
    def __init__(self, incoming, num_filters, filter_size=3, dilation=(2, 2),
                 conv1_W=init.GlorotUniform(), conv1_b=init.Constant(0.),
                 bn1_beta=init.Constant(0.), bn1_gamma=init.Constant(1.),
                 bn1_mean=init.Constant(0.), bn1_inv_std=init.Constant(1.),
                 conv2_W=init.GlorotUniform(), conv2_b=init.Constant(0.),
                 bn2_beta=init.Constant(0.), bn2_gamma=init.Constant(1.),
                 bn2_mean=init.Constant(0.), bn2_inv_std=init.Constant(1.),
                 conv3_W=init.GlorotUniform(), conv3_b=init.Constant(0.),
                 bn3_beta=init.Constant(0.), bn3_gamma=init.Constant(1.),
                 bn3_mean=init.Constant(0.), bn3_inv_std=init.Constant(1.),
                 batch_norm=False, name=None,
                 **tags):
        super(DilatedVggEncoding3Layer, self).__init__(incoming, name=name)
        layer = self.l_conv1 = self.add_layer(
            L.Conv2DLayer(incoming, num_filters, filter_size=filter_size, stride=1, pad='same', nonlinearity=None,
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
            L.Conv2DLayer(layer, num_filters, filter_size=filter_size, stride=1, pad='same', nonlinearity=None,
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

        # layer = self.l_pad3 = self.add_layer(
        #     L.PadLayer(layer, (filter_size - 1) * dilation // 2))  # 'same' padding
        # layer = self.l_conv3 = self.add_layer(
        #     L.DilatedConv2DLayer(layer, num_filters, filter_size=filter_size, dilation=dilation, nonlinearity=None,
        #                          W=conv3_W,
        #                          b=conv3_b,
        #                          name='%s.%s' % (name, 'conv3') if name is not None else None))
        layer = self.l_conv3 = self.add_layer(
            L.Conv2DLayer(layer, num_filters, filter_size=filter_size, filter_dilation=dilation, pad='same', nonlinearity=None,
                          W=conv3_W,
                          b=conv3_b,
                          name='%s.%s' % (name, 'conv3') if name is not None else None))
        if batch_norm:
            layer = self.l_bn3 = self.add_layer(
                L.BatchNormLayer(layer,
                                 beta=bn3_beta,
                                 gamma=bn3_gamma,
                                 mean=bn3_mean,
                                 inv_std=bn3_inv_std,
                                 name='%s.%s' % (name, 'bn3') if name is not None else None))
        else:
            self.l_bn3 = None
        self.l_relu3 = self.add_layer(
            L.NonlinearityLayer(layer, nonlinearity=nl.rectify,
                                name='%s.%s' % (name, 'relu3') if name is not None else None))

        for tag in tags.keys():
            if not isinstance(tag, str):
                raise ValueError("tag should be a string, %s given" % type(tag))
        tags['encoding'] = tags.get('encoding', True)
        set_layer_param_tags(self, **tags)

        self.param_keys = dict()
        for layer, base_name in [(self.l_conv1, 'conv1'), (self.l_conv2, 'conv2'), (self.l_conv3, 'conv3')]:
            self.param_keys.update({
                layer.W: '%s_W' % base_name,
                layer.b: '%s_b' % base_name,
            })
        for layer, base_name in [(self.l_bn1, 'bn1'), (self.l_bn2, 'bn2'), (self.l_bn3, 'bn3')]:
            if layer is not None:
                self.param_keys.update({
                    layer.beta: '%s_beta' % base_name,
                    layer.gamma: '%s_gamma' % base_name,
                    layer.mean: '%s_mean' % base_name,
                    layer.inv_std: '%s_inv_std' % base_name
                })


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


class OuterProductLayer(L.MergeLayer):
    def __init__(self, incomings, use_bias=True, **kwargs):
        super(OuterProductLayer, self).__init__(incomings, **kwargs)
        self.use_bias = use_bias

    def get_output_shape_for(self, input_shapes):
        Y_shape, U_shape = input_shapes
        assert Y_shape[0] == U_shape[0]
        y_shape = Y_shape[1:]
        u_shape = U_shape[1:]
        u_dim, = u_shape
        outer_shape = (y_shape[0]*(u_dim + int(self.use_bias)),) + y_shape[1:]
        return (Y_shape[0],) + outer_shape

    def get_output_for(self, inputs, **kwargs):
        Y, U = inputs
        if self.use_bias:
            U = T.concatenate([U, T.ones((U.shape[0], 1))], axis=1)
        outer_YU = Y.dimshuffle([0, 1, 'x'] + list(range(2, Y.ndim))) * U.dimshuffle([0, 'x', 1] + ['x']*(Y.ndim-2))
        return outer_YU.reshape((Y.shape[0], -1, Y.shape[2], Y.shape[3]))


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


class BatchwiseSumLayer(L.ElemwiseMergeLayer):
    def __init__(self, incomings, **kwargs):
        super(BatchwiseSumLayer, self).__init__(incomings, T.add, **kwargs)

    def get_output_shape_for(self, input_shapes):
        x_shapes = input_shapes[:-1]
        u_shape = input_shapes[-1]
        x_shape = x_shapes[0]
        for shape in x_shapes[1:]:
            assert shape == x_shape
        batch_size, u_dim = u_shape
        assert len(x_shapes) in (u_dim, u_dim + 1)
        assert x_shape[0] == batch_size
        return x_shape

    def get_output_for(self, inputs, **kwargs):
        xs = inputs[:-1]
        u = inputs[-1]
        _, u_dim = u.shape
        xs = [x if i == 6 else x * u[:, i, None, None, None] for i, x in enumerate(xs)]
        return super(BatchwiseSumLayer, self).get_output_for(xs, **kwargs)


def create_bilinear_layer(l_xlevel, l_u, level, bilinear_type='share', name=None):
        if bilinear_type == 'convolution':
            l_x_shape = L.get_output_shape(l_xlevel)
            l_xlevel_u_outer = OuterProductLayer([l_xlevel, l_u], name='x%d_u_outer' % level)
            l_xlevel_diff_pred = L.Conv2DLayer(l_xlevel_u_outer, l_x_shape[1], filter_size=5, stride=1, pad='same',
                                               untie_biases=True, nonlinearity=None, name=name)
            set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d' % level, True)]))
            # TODO: param_tags?
        elif bilinear_type == 'group_convolution':
            l_x_shape = L.get_output_shape(l_xlevel)
            _, u_dim = L.get_output_shape(l_u)
            l_xlevel_gconvs = []
            for i in range(u_dim + 1):
                l_xlevel_gconv = GroupConv2DLayer(l_xlevel, l_x_shape[1], filter_size=5, stride=1, pad='same',
                                                 untie_biases=True, groups=l_x_shape[1], nonlinearity=None,
                                                 name='%s_gconv%d' % (name, i))
                l_xlevel_gconvs.append(l_xlevel_gconv)
            l_xlevel_diff_pred = BatchwiseSumLayer(l_xlevel_gconvs + [l_u], name=name)
            # l_xlevel_u_outer = OuterProductLayer([l_xlevel, l_u], name='x%d_u_outer' % level)
            # l_xlevel_diff_pred = GroupConv2DLayer(l_xlevel_u_outer, l_x_shape[1], filter_size=5, stride=1, pad='same',
            #                                       untie_biases=True, groups=l_x_shape[1], nonlinearity=None, name=name)
            set_layer_param_tags(l_xlevel_diff_pred, transformation=True, **dict([('level%d' % level, True)]))
        elif bilinear_type == 'full':
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
            raise ValueError('bilinear_type should be either convolution, full, share channelwise or factor, given %s'%bilinear_type)
        return l_xlevel_diff_pred
