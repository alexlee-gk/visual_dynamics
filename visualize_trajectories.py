import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gui.grid_image_visualizer import GridImageVisualizer
from gui.arrow_plotter import ArrowPlotter
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fname', type=str, help='file name of data container')
    parser.add_argument('transformers_fname', nargs='?', type=str)
    parser.add_argument('--steps', '-t', type=int, default=10)
    args = parser.parse_args()

    container = utils.container.ImageDataContainer(args.data_fname)
    environment_config = container.get_info('environment_config')
    action_space = utils.config.from_config(environment_config['action_space'])

    # transformers
    if args.transformers_fname:
        with open(args.transformers_fname) as tranformers_file:
            transformers = utils.from_yaml(tranformers_file)
        action_space = utils.from_config(container.get_info('environment_config')['action_space'])
        observation_space = utils.from_config(container.get_info('environment_config')['observation_space'])
        sensor_names = container.get_info('environment_config')['sensor_names']
        data_spaces = dict(zip(sensor_names + ['action'], observation_space.spaces + [action_space]))

        for data_name, transformer in transformers.items():
            for nested_transformer in utils.transformer.get_all_transformers(transformer):
                if isinstance(nested_transformer, (utils.transformer.NormalizerTransformer,
                                                   utils.transformer.DepthImageTransformer)):
                    if data_name in data_spaces:
                        nested_transformer.space = data_spaces[data_name]
    else:
        transformers = None

    fig = plt.figure(figsize=(2*args.steps, 2), frameon=False, tight_layout=True)
    gs = gridspec.GridSpec(2, args.steps)
    image_visualizer = GridImageVisualizer(fig, gs[0, :], args.steps, rows=1)
    labels = ['pan', 'tilt']
    limits = [action_space.low, action_space.high]
    arrow_plotters = [ArrowPlotter(fig, gs[1, i], labels, limits) for i in range(args.steps)]
    plt.show(block=False)

    num_trajs, num_steps = container.get_data_shape('action')
    assert container.get_data_shape('state') == (num_trajs, num_steps + 1)
    assert num_steps % args.steps == 0

    for traj_iter in range(num_trajs):
        images = []
        actions = []
        for step_iter in range(num_steps):
            image, action = container.get_datum(traj_iter, step_iter, ['image', 'action'])
            if transformers:
                image = transformers['image'].preprocess(image)
                image = transformers['image'].transformers[-1].deprocess(image)
                action = transformers['action'].preprocess(action)
            images.append(image.copy())
            actions.append(action.copy())
            if len(images) == args.steps:
                image_visualizer.update(images)
                for action, arrow_plotter in zip(actions, arrow_plotters):
                    arrow_plotter.update(action)
                images = []
                actions = []
    container.close()


if __name__ == "__main__":
    main()
