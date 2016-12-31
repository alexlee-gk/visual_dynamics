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
    parser.add_argument('--select_features', type=int, default=None)
    parser.add_argument('--positive_data_fnames', '--pos', nargs='+', type=str)
    parser.add_argument('--negative_data_fnames', '--neg', nargs='+', type=str)
    args = parser.parse_args()

    if args.select_features:
        assert args.positive_data_fnames
        assert args.negative_data_fnames

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
    if solver_config.get('val_data_fname') is not None:
        data_fnames.append(solver_config['val_data_fname'])
    with utils.container.MultiDataContainer(data_fnames) as data_container:
        # TODO
        try:
            sensor_names = data_container.get_info('environment_config')['sensor_names']
        except KeyError:
            sensor_names = ['quad_to_obj_pos', 'quad_to_obj_rot', 'image']
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
        # TODO
        try:
            action_space = utils.from_config(data_container.get_info('environment_config')['action_space'])
        except KeyError:
            import numpy as np
            import spaces
            action_space = spaces.TranslationAxisAngleSpace(-np.ones(4), np.ones(4), axis=np.array([0, 0, 1]))
        # import IPython as ipy; ipy.embed()
        # TODO: member variables not in config. assumption obs space is tuple. instantiating env without ros.
        try:
            observation_space = utils.from_config(data_container.get_info('environment_config')['observation_space'])
        except KeyError:
            try:
                env = utils.from_config(data_container.get_info('environment_config'))
                observation_space = env.observation_space
            except:
                import numpy as np
                import spaces
                observation_space = spaces.TupleSpace([spaces.BoxSpace(0, 255, shape=(368, 640, 3), dtype=np.uint8)])
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
        # TODO: hack
        predictor_config['name'] += '_' + os.path.splitext(os.path.split(args.transformers_fname)[1])[0]

    if args.select_features:
        predictor_config['name'] += '_infogain%d' % args.select_features
        assert args.positive_data_fnames
        assert args.negative_data_fnames
        import numpy as np
        from information_gain import entropy_gaussian

        predictor_config['no_dynamics'] = True
        feature_predictor = utils.config.from_config(predictor_config)

        pos_data_gen = utils.generator.DataGenerator(args.positive_data_fnames,
                                                     data_name_offset_pairs=[('image', 0)],
                                                     transformers=transformers,
                                                     once=True,
                                                     shuffle=False,
                                                     batch_size=100,
                                                     dtype=theano.config.floatX)
        pos_data_gen = utils.generator.ParallelGenerator(pos_data_gen, nb_worker=4)
        pos_features = [[] for _ in range(len(feature_predictor.feature_name))]
        for X in pos_data_gen:
            if isinstance(X, tuple):
                X, = X
            for pos_feature, feature in zip(pos_features, feature_predictor.feature(X)):
                pos_feature.append(feature)
        for i, pos_feature in enumerate(pos_features):
            pos_features[i] = np.concatenate(pos_feature, axis=0)

        neg_data_gen = utils.generator.DataGenerator(args.negative_data_fnames,
                                                     data_name_offset_pairs=[('image', 0)],
                                                     transformers=transformers,
                                                     once=True,
                                                     shuffle=False,
                                                     batch_size=100,
                                                     dtype=theano.config.floatX)
        neg_data_gen = utils.generator.ParallelGenerator(neg_data_gen, nb_worker=4)
        neg_features = [[] for _ in range(len(feature_predictor.feature_name))]
        for X in neg_data_gen:
            if isinstance(X, tuple):
                X, = X
            for neg_feature, feature in zip(neg_features, feature_predictor.feature(X)):
                neg_feature.append(feature)
        for i, neg_feature in enumerate(neg_features):
            neg_features[i] = np.concatenate(neg_feature, axis=0)

        pos_features = np.concatenate([pos_feature.reshape(pos_feature.shape[:2] + (-1,)) for pos_feature in pos_features], axis=-1)
        neg_features = np.concatenate([neg_feature.reshape(neg_feature.shape[:2] + (-1,)) for neg_feature in neg_features], axis=-1)

        std_axes = (0, 2)
        p_y1 = (pos_features.shape[0] * pos_features.shape[2]) / ((pos_features.shape[0] * pos_features.shape[2]) + (neg_features.shape[0] * neg_features.shape[2]))
        p_y0 = 1 - p_y1
        x_std = np.r_[pos_features, neg_features].std(axis=std_axes).flatten()
        x_y0_std = neg_features.std(axis=std_axes).flatten()
        x_y1_std = pos_features.std(axis=std_axes).flatten()
        information_gain = entropy_gaussian(x_std ** 2) - \
                           (p_y0 * entropy_gaussian(x_y0_std ** 2) + \
                            p_y1 * entropy_gaussian(x_y1_std ** 2))
        information_gain[np.isnan(information_gain)] = 0
        information_gain[np.isinf(information_gain)] = 0

        predictor_config['no_dynamics'] = False
        predictor_config['channel_inds'] = np.argsort(information_gain)[::-1][:args.select_features].tolist()
        print("channel_inds:", predictor_config['channel_inds'])

    feature_predictor = utils.config.from_config(predictor_config)

    # ### START
    # import IPython as ipy; ipy.embed()
    # # TODO: use train_fname
    # train_data_fnames = solver_config['train_data_fnames']
    # data_name_offset_pairs = solver_config['data_name_offset_pairs']
    # aggregating_batch_size = 100
    #
    # import numpy as np
    # from collections import OrderedDict
    # import lasagne.layers as L
    #
    # feature_name = 'x5'
    # pred_layer = feature_predictor.pred_layers[feature_name]
    # last_pred_layer = None
    # prerelu_pred_layers = OrderedDict()
    # postrelu_pred_layers = OrderedDict()
    # while not isinstance(pred_layer, L.InputLayer):
    #     if last_pred_layer is not None:
    #         postrelu_pred_layers[pred_layer.name + '_postrelu'] = last_pred_layer.layers[0]
    #     assert isinstance(pred_layer.layers[-1], L.NonlinearityLayer)
    #     prerelu_pred_layers[pred_layer.name + '_prerelu'] = pred_layer.layers[-2]
    #     last_pred_layer = pred_layer
    #     pred_layer = pred_layer.layers[0].input_layer
    # prerelu_pred_layers = OrderedDict(prerelu_pred_layers.items()[::-1])
    # postrelu_pred_layers = OrderedDict(postrelu_pred_layers.items()[::-1])
    # feature_predictor.pred_layers.update(prerelu_pred_layers)
    # feature_predictor.pred_layers.update(postrelu_pred_layers)
    #
    # prerelu_feature_names = prerelu_pred_layers.keys()
    # online_stats = [utils.OnlineStatistics(axis=(0, 2, 3)) for _ in prerelu_feature_names]
    #
    # from utils import tic, toc
    # train_data_once_gen = utils.generator.DataGenerator(train_data_fnames,
    #                                                     data_name_offset_pairs=data_name_offset_pairs,
    #                                                     transformers=transformers,
    #                                                     once=True,
    #                                                     batch_size=aggregating_batch_size,
    #                                                     shuffle=False,
    #                                                     dtype=theano.config.floatX)
    # train_data_once_gen = utils.generator.ParallelGenerator(train_data_once_gen, nb_worker=4)
    # tic()
    # for batch_data in train_data_once_gen:
    #     X = batch_data[0]
    #     features = feature_predictor.predict(prerelu_feature_names, X)
    #
    #     for feature, online_stat in zip(features, online_stats):
    #         online_stat.add_data(feature)
    # toc()
    #
    # for online_stat, prerelu_pred_layer, postrelu_pred_layer in \
    #         zip(online_stats, prerelu_pred_layers.values(), list(postrelu_pred_layers.values()) + [None]):
    #     W, b = prerelu_pred_layer.W, prerelu_pred_layer.b
    #     W.set_value((W.get_value() / online_stat.std[:, None, None, None]).astype(theano.config.floatX))
    #     b.set_value((b.get_value() / online_stat.std).astype(theano.config.floatX))
    #     if postrelu_pred_layer is not None:
    #         W = postrelu_pred_layer.W
    #         W.set_value((W.get_value() * online_stat.std[None, :, None, None]).astype(theano.config.floatX))
    #
    # feature_predictor.save_model('models/theano/standarized_vgg/simplequad_bigdata_model.yaml')
    #
    # # TODO: unit test to check output is the same
    # # TODO: unit test to check std is close to 1
    # # maybe not...
    #
    # check = True
    # if check:
    #     for online_stat in online_stats:
    #         online_stat.reset()
    #
    #     from utils import tic, toc
    #     train_data_once_gen = utils.generator.DataGenerator(train_data_fnames,
    #                                                         data_name_offset_pairs=data_name_offset_pairs,
    #                                                         transformers=transformers,
    #                                                         once=True,
    #                                                         batch_size=aggregating_batch_size,
    #                                                         shuffle=False,
    #                                                         dtype=theano.config.floatX)
    #     train_data_once_gen = utils.generator.ParallelGenerator(train_data_once_gen, nb_worker=4)
    #     for batch_data in train_data_once_gen:
    #         X = batch_data[0]
    #         check_features = feature_predictor.predict(prerelu_feature_names, X)
    #
    #         for feature, online_stat in zip(check_features, online_stats):
    #             online_stat.add_data(feature)
    #
    #     for online_stat in online_stats:
    #         assert np.allclose(online_stat.std, np.ones_like(online_stat.std))
    #
    #     np.allclose(features[-1], check_features[-1])
    #
    # ### END

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
