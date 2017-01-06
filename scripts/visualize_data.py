from __future__ import division, print_function
import argparse
import theano
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
from gui.grid_image_visualizer import GridImageVisualizer
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--record_file', '-r', type=str, default=None)
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        feature_predictor = utils.from_yaml(predictor_file)

    transformers = feature_predictor.transformers
    solver = feature_predictor.solvers[-1]

    # visualization is the same as what's at the end of train.py
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
        other_outputs = feature_predictor.predict(other_output_names, [data[0]], preprocessed=True)
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
