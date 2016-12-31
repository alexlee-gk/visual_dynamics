from __future__ import division, print_function
import argparse
import numpy as np
import policy
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gui.grid_image_visualizer import GridImageVisualizer
import _tkinter


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


def entropy_gaussian(variance):
    return .5 * (1 + np.log(2 * np.pi)) + .5 * np.log(variance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('transformers_fname', nargs='?', type=str)
    parser.add_argument('--positive_data_fnames', '--pos', nargs='+', type=str)
    parser.add_argument('--negative_data_fnames', '--neg', nargs='+', type=str)
    parser.add_argument('--learn_masks', '-m', action='store_true')
    args = parser.parse_args()

    assert args.positive_data
    assert args.negative_data

    predictor = utils.from_yaml(open(args.predictor_fname))
    # predictor.environment_config['car_color'] = 'random'
    # predictor.environment_config['car_color'] = 'green'
    env = utils.from_config(predictor.environment_config)

    pos_gen = utils.DataGenerator(args.positive_data_fnames, [('image', 0)], transformers=transformers, once=True)

    with utils.container.MultiDataContainer(args.positive_data_fnames) as data_container:
        sensor_names = data_container.get_info('environment_config')['sensor_names']
        data_names = solver_config.get('data_names', sensor_names + ['action'])
        input_shapes = [data_container.get_datum_shape(name) for name in data_names]




    images = np.array(images)
    labels = np.array(labels)
    if args.use_features:
        Xs = []
        ind = 0
        while ind < len(images):
            X = predictor.feature(np.array(images[ind:ind+100]))[0]
            Xs.append(X)
            ind += 100
        X = np.concatenate(Xs)
    else:
        X = predictor.preprocess(images)[0]
    y = labels

    if args.learn_masks:
        std_axes = 0
    else:
        std_axes = (0, 2, 3)
    p_y1 = (y == 1).mean()
    p_y0 = 1 - p_y1
    x_std = X.std(axis=std_axes).flatten()
    x_y0_std = X[y == 0].std(axis=std_axes).flatten()
    x_y1_std = X[y == 1].std(axis=std_axes).flatten()
    information_gain = entropy_gaussian(x_std ** 2) - \
                       (p_y0 * entropy_gaussian(x_y0_std ** 2) + \
                        p_y1 * entropy_gaussian(x_y1_std ** 2))
    # assert np.allclose(information_gain, 0.5 * (np.log(x_std ** 2) - p_y0 * np.log(x_y0_std ** 2) - p_y1 * np.log(x_y1_std ** 2)))
    information_gain = information_gain.reshape((X.shape[1:]))
    information_gain[np.isnan(information_gain)] = 0
    information_gain[np.isinf(information_gain)] = 0

    # TODO: check if equality holds
    import IPython as ipy; ipy.embed()

    fig = plt.figure(figsize=(4 * X.shape[1], 4), frameon=False, tight_layout=True)
    gs = gridspec.GridSpec(1, 1)
    image_visualizer = GridImageVisualizer(fig, gs[0], X.shape[1], rows=1)
    plt.show(block=False)
    image_visualizer.update(information_gain)

    print(np.sort(information_gain.sum(axis=(1, 2)))[::-1])
    print(np.argsort(information_gain.sum(axis=(1, 2)))[::-1])
    if args.use_features:
        alex_inds = [36, 178, 307, 490]
        print([np.where(np.argsort(information_gain.sum(axis=(1, 2)))[::-1] == ind)[0] for ind in alex_inds])

    # from sklearn.feature_selection import chi2
    # features = predictor.feature(np.array(images))
    # X = features[0].mean(axis=(-2, -1))
    # # X = predictor.preprocess(np.array(images))[0].mean(axis=(-2, -1))
    # # X -= X.min()
    # y = np.array(labels)
    # scores, pvalues = chi2(X, y)
    # print(np.argsort(scores)[::-1])

    import IPython as ipy;
    ipy.embed()


if __name__ == '__main__':
    main()
