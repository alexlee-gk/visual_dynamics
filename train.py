from __future__ import division, print_function
import os
import argparse
import yaml
import theano
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('transformers_fname', nargs='?', type=str)
    parser.add_argument('solver_fname', nargs='?', type=str)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    # solver
    if args.solver_fname:
        with open(args.solver_fname) as solver_file:
            solver_config = yaml.load(solver_file)
    else:
        try:
            solver_config = predictor_config['solvers'][-1]
            args.no_train = True
        except IndexError:
            raise ValueError('solver_fname was not specified but predictor does not have a solver')

    # extract info from solver
    data_fnames = list(solver_config['train_data_fnames'])
    if solver_config['val_data_fname'] is not None:
        data_fnames.append(solver_config['val_data_fname'])
    with utils.container.MultiDataContainer(data_fnames) as data_container:
        sensor_names = data_container.get_info('environment_config')['sensor_names']
        data_names = solver_config.get('data_names', sensor_names + ['action'])
        input_shapes = [data_container.get_datum_shape(name) for name in data_names]

    # input_shapes
    if 'input_shapes' in predictor_config:
        if input_shapes != predictor_config['input_shapes']:
            raise ValueError('conflicting values for input_shapes')
    else:
        predictor_config['input_shapes'] = input_shapes

    # transformers
    if args.transformers_fname:
        with open(args.transformers_fname) as tranformers_file:
            transformers = utils.from_yaml(tranformers_file)
        action_space = utils.from_config(data_container.get_info('environment_config')['action_space'])
        observation_space = utils.from_config(data_container.get_info('environment_config')['observation_space'])
        data_spaces = dict(zip(sensor_names + ['action'], observation_space.spaces + [action_space]))

        for data_name, transformer in transformers.items():
            for nested_transformer in utils.transformer.get_all_transformers(transformer):
                if isinstance(nested_transformer, (utils.transformer.NormalizerTransformer,
                                                   utils.transformer.DepthImageTransformer)):
                    if data_name in data_spaces:
                        nested_transformer.space = data_spaces[data_name]

        # TODO: better way to map data and input names
        if 'input_names' not in predictor_config:
            predictor_config['input_names'] = ['x', 'u']
        data_to_input_name = dict(zip(data_names, predictor_config['input_names']))
        transformers = utils.get_config({data_to_input_name[k]: v for k, v in transformers.items() if k in data_names})
        if 'transformers' in predictor_config:
            if transformers != predictor_config['transformers']:
                raise ValueError('conflicting values for transformers')
        else:
            predictor_config['transformers'] = transformers

    if 'name' not in predictor_config:
        predictor_config['name'] = os.path.splitext(os.path.split(args.predictor_fname)[1])[0]

    feature_predictor = utils.config.from_config(predictor_config)

    if not args.no_train:
        feature_predictor.train(args.solver_fname)

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        solver = utils.from_config(solver_config)
        data_to_input_name = dict(zip(solver.data_names, solver.input_names))
        transformers = {data_name: feature_predictor.transformers[data_to_input_name[data_name]] for data_name in solver.data_names}
        val_data_gen = utils.generator.DataGenerator(solver.val_data_fname,
                                                     data_name_offset_pairs=solver.data_name_offset_pairs,
                                                     transformers=transformers,
                                                     once=True,
                                                     batch_size=0,
                                                     shuffle=False,
                                                     dtype=theano.config.floatX)

        fig = plt.figure(figsize=(12, 12), frameon=False, tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        plt.show(block=False)

        if args.record_file:
            FFMpegWriter = manimation.writers['ffmpeg']
            dt = feature_predictor.environment_config.get('dt', 1 / 30)
            writer = FFMpegWriter(fps=1.0 / dt)
            writer.setup(fig, args.record_file, fig.dpi)

        other_output_names = []
        output_labels = []
        for output_name_pair in solver.output_names:
            # first add the ones with times
            for output_name in output_name_pair:
                if isinstance(output_name, (list, tuple)):
                    other_output_name, t = output_name
                    other_output_names.append(other_output_name)
                    output_labels.append(other_output_name)
            for output_name in output_name_pair:
                output_labels.append(output_name)
        # assume there is only one other output per output pair
        assert len(other_output_names) == len(solver.output_names)
        image_visualizer = GridImageVisualizer(fig, gs[0], rows=len(solver.output_names), cols=3, labels=output_labels)

        done = False
        for data in val_data_gen:
            outputs = solver.get_outputs(feature_predictor, *data, preprocessed=True)
            other_outputs = feature_predictor.predict(other_output_names, data[0], preprocessed=True)
            vis_outputs = []
            assert len(other_outputs) == len(outputs)
            for other_output, output_pair in zip(other_outputs, outputs):
                for output in (other_output,) + output_pair:
                    if output.ndim == 3 and output.shape[0] == 3:  # TODO: better way of identifying RGB images
                        output = feature_predictor.transformers['x'].deprocess(output)
                    vis_outputs.append(output)
            try:
                image_visualizer.update(vis_outputs)
                if args.record_file:
                    writer.grab_frame()
            except:
                done = True
            if done:
                break
        if args.record_file:
            writer.finish()


if __name__ == "__main__":
    main()
