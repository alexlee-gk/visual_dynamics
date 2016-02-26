"""
Copied from https://gist.github.com/ebenolson/1682625dc9823e27d771

Functions to create network diagrams from a list of Layers.

Examples:

    Draw a minimal diagram to a pdf file:
        layers = lasagne.layers.get_all_layers(output_layer)
        draw_to_file(layers, 'network.pdf', output_shape=False)

    Draw a verbose diagram in an IPython notebook:
        from IPython.display import Image #needed to render in notebook

        layers = lasagne.layers.get_all_layers(output_layer)
        dot = get_pydot_graph(layers, verbose=True)
        return Image(dot.create_png())
"""

import pydot


def get_hex_color(layer_type):
    """
    Determines the hex color for a layer. Some classes are given
    default values, all others are calculated pseudorandomly
    from their name.
    :parameters:
        - layer_type : string
            Class name of the layer

    :returns:
        - color : string containing a hex color.

    :usage:
        >>> color = get_hex_color('MaxPool2DDNN')
        '#9D9DD2'
    """

    if 'Input' in layer_type:
        return '#A2CECE'
    if 'Conv' in layer_type:
        return '#7C9ABB'
    if 'Dense' in layer_type:
        return '#6CCF8D'
    if 'Pool' in layer_type:
        return '#9D9DD2'
    else:
        return '#{0:x}'.format(hash(layer_type) % 2**24)


def get_pydot_graph(layers, output_shape=True, verbose=False):
    """
    Creates a PyDot graph of the network defined by the given layers.
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - output_shape: (default `True`)
            If `True`, the output shape of each layer will be displayed.
        - verbose: (default `False`)
            If `True`, layer attributes like filter shape, stride, etc.
            will be displayed.
        - verbose:
    :returns:
        - pydot_graph : PyDot object containing the graph

    """
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    pydot_nodes = {}
    pydot_edges = []
    for i, layer in enumerate(layers):
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_type
        if layer.name is not None:
            label += ' ({0})'.format(layer.name)
        color = get_hex_color(layer_type)
        if verbose:
            for attr in ['num_filters', 'num_units', 'ds',
                         'filter_shape', 'stride', 'strides', 'p']:
                if hasattr(layer, attr):
                    label += '\n' + \
                        '{0}: {1}'.format(attr, getattr(layer, attr))
            if hasattr(layer, 'nonlinearity'):
                try:
                    nonlinearity = layer.nonlinearity.__name__
                except AttributeError:
                    nonlinearity = layer.nonlinearity.__class__.__name__
                label += '\n' + 'nonlinearity: {0}'.format(nonlinearity)

        if output_shape:
            try:
                label += '\n' + \
                    'Output shape: {0}'.format(layer.get_output_shape())
            except AttributeError:
                pass
        pydot_nodes[key] = pydot.Node(key,
                                      label=label,
                                      shape='record',
                                      fillcolor=color,
                                      style='filled',
                                      )

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                pydot_edges.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            pydot_edges.append([repr(layer.input_layer), key])

    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge[0]], pydot_nodes[edge[1]]))
    return pydot_graph


def draw_to_file(layers, filename, **kwargs):
    """
    Draws a network diagram to a file
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - filename: string
            The filename to save output to.
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    dot = get_pydot_graph(layers, **kwargs)

    ext = filename[filename.rfind('.') + 1:]
    with open(filename, 'wb') as fid:
        fid.write(dot.create(format=ext))


def draw_to_notebook(layers, **kwargs):
    """
    Draws a network diagram in an IPython notebook
    :parameters:
        - layers : list
            List of the layers, as obtained from lasange.layers.get_all_layers
        - **kwargs: see docstring of get_pydot_graph for other options
    """
    from IPython.display import Image  # needed to render in notebook

    dot = get_pydot_graph(layers, **kwargs)
    return Image(dot.create_png())


def main():
    import argparse
    import lasagne
    from predictor import net_theano
    from predictor import predictor_theano

    parser = argparse.ArgumentParser()
    parser.add_argument('--x_shape', type=int, nargs='+', default=[3, 32, 32])
    parser.add_argument('--u_shape', type=int, nargs='+', default=[5])
    parser.add_argument('--predictor', '-p', type=str, default='build_fcn_action_cond_encoder_net')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--levels', type=int, nargs='+', default=[3], help='net parameter')
    parser.add_argument('--x1_c_dim', '--x1cdim', type=int, default=16, help='net parameter')
    parser.add_argument('--num_downsample', '--numds', type=int, default=0, help='net parameter')
    parser.add_argument('--share_bilinear_weights', '--share', type=int, default=1, help='net parameter')
    parser.add_argument('--ladder_loss', '--ladder', type=int, default=0, help='net parameter')
    parser.add_argument('--batch_normalization', '--bn', type=int, default=0, help='net parameter')
    parser.add_argument('--concat', type=int, default=0, help='net parameter')
    parser.add_argument('--axis', type=int, default=2, help='net parameter')

    args = parser.parse_args()
    args.output = args.output or (args.predictor + '.pdf')

    build_net = getattr(net_theano, args.predictor)
    input_shapes = (tuple(args.x_shape), tuple(args.u_shape))
    if args.predictor == 'build_fcn_action_cond_encoder_only_net':
        TheanoNetFeaturePredictor = predictor_theano.FcnActionCondEncoderOnlyTheanoNetFeaturePredictor
    else:
        TheanoNetFeaturePredictor = predictor_theano.TheanoNetFeaturePredictor
    feature_predictor = TheanoNetFeaturePredictor(*build_net(input_shapes,
                                                             levels=args.levels,
                                                             x1_c_dim=args.x1_c_dim,
                                                             num_downsample=args.num_downsample,
                                                             share_bilinear_weights=args.share_bilinear_weights,
                                                             ladder_loss=args.ladder_loss,
                                                             batch_normalization=args.batch_normalization,
                                                             concat=args.concat,
                                                             axis=args.axis))
    layers = feature_predictor.get_all_layers()
    print('Writing to %s'%args.output)
    draw_to_file(layers, args.output, output_shape=True, verbose=args.verbose)


if __name__ == "__main__":
    main()
