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
    parser.add_argument('transformers_fname', type=str)
    parser.add_argument('solver_fname', type=str)
    parser.add_argument('data_fname', type=str)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--visualize', '-v', type=int, default=None)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    with open(args.solver_fname) as solver_file:
        solver_config = yaml.load(solver_file)

    with open(args.data_fname) as data_file:
        data_config = yaml.load(data_file)
    if 'train_data_fnames' in data_config:
        solver_config['train_data_fnames'] = data_config['train_data_fnames']
    if 'val_data_fnames' in data_config:
        solver_config['val_data_fnames'] = data_config['val_data_fnames']

    # data_names and input_names
    data_names = solver_config['data_names']
    input_names = predictor_config['input_names']
    # if predictor_config['input_names'] != input_names:
    #     raise ValueError('conflicting values for input_names')

    # extract info from data
    data_fnames = solver_config.get('train_data_fnames', []) + solver_config.get('val_data_fnames', [])
    with utils.container.MultiDataContainer(data_fnames) as data_container:
        env_spec = utils.from_config(data_container.get_info('env_spec_config'))
        input_shapes = [data_container.get_datum_shape(name) for name in data_names]

    # input_shapes
    if 'input_shapes' in predictor_config:
        if input_shapes != predictor_config['input_shapes']:
            raise ValueError('conflicting values for input_shapes')
    else:
        predictor_config['input_shapes'] = input_shapes

    # transformers
    with open(args.transformers_fname) as transformers_file:
        transformers_config = yaml.load(transformers_file)
    transformers = dict()
    for data_name, transformer_config in transformers_config.items():
        if data_name == 'action':
            replace_config = {'space': env_spec.action_space}
        elif data_name in env_spec.observation_space.spaces:
            replace_config = {'space': env_spec.observation_space.spaces[data_name]}
        else:
            replace_config = {}
        transformers[data_name] = utils.from_config(transformers_config[data_name], replace_config=replace_config)

    input_to_data_name = dict(zip(input_names, data_names))
    transformers.update([(input_name, transformers[input_to_data_name[input_name]]) for input_name in input_names])
    transformers_config = utils.get_config(transformers)
    if 'transformers' in predictor_config:
        if transformers_config != predictor_config['transformers']:
            raise ValueError('conflicting values for transformers')
    else:
        predictor_config['transformers'] = transformers_config

    if 'name' not in predictor_config:
        predictor_config['name'] = os.path.join(os.path.splitext(os.path.split(args.predictor_fname)[1])[0],
                                                os.path.splitext(os.path.split(args.transformers_fname)[1])[0])

    feature_predictor = utils.config.from_config(predictor_config)

    if 'snapshot_prefix' not in solver_config:
        snapshot_prefix = os.path.join(os.path.splitext(os.path.split(args.solver_fname)[1])[0],
                                                        os.path.splitext(os.path.split(args.data_fname)[1])[0])
        solver_config['snapshot_prefix'] = feature_predictor.get_snapshot_prefix(snapshot_prefix)

    if not args.no_train:
        solver = utils.from_config(solver_config)
        feature_predictor.train(solver)
    else:
        solver = None

    if args.record_file and not args.visualize:
        args.visualize = 1
    if args.visualize:
        if solver is None:
            solver = utils.from_config(solver_config)

        data_gen = utils.generator.DataGenerator(solver.val_data_fnames if solver.val_data_fnames else solver.train_data_fnames,
                                                 data_name_offset_pairs=solver.data_name_offset_pairs,
                                                 transformers=transformers,
                                                 once=True,
                                                 batch_size=0,
                                                 shuffle=False,
                                                 dtype=theano.config.floatX)

        fig = plt.figure(figsize=(12, 12), frameon=False, tight_layout=True)
        fig.canvas.set_window_title(solver.snapshot_prefix)
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
        for data in data_gen:
            outputs = solver.get_outputs(feature_predictor, *data, preprocessed=True)
            other_outputs = feature_predictor.predict(other_output_names, data[0], preprocessed=True)
            vis_outputs = []
            assert len(other_outputs) == len(outputs)
            for other_output, output_pair in zip(other_outputs, outputs):
                for output in (other_output,) + output_pair:
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
