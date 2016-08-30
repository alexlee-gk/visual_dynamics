import os
import argparse
from collections import OrderedDict
import cv2
import yaml
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('transformers_fname', type=str)
    parser.add_argument('solver_fname', nargs='?', type=str)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='S', help='rescale image by S for visualization')

    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)
    if args.solver_fname:
        with open(args.solver_fname) as solver_file:
            solver_config = yaml.load(solver_file)
    else:
        try:
            solver_config = predictor_config['solvers'][-1]
            args.no_train = True
        except IndexError:
            raise ValueError('solver_fname was not specified but predictor does not have a solver')

    # input_shapes
    data_fnames = list(solver_config['train_data_fnames'])
    if solver_config['val_data_fname'] is not None:
        data_fnames.append(solver_config['val_data_fname'])
    with utils.container.MultiDataContainer(data_fnames) as data_container:
        sensor_names = data_container.get_info('environment_config')['sensor_names']
        data_names = solver_config.get('data_names', [*sensor_names, 'action'])
        # input_shapes = [data_container.get_datum_shape(name) for name in ('image', 'vel')]
        input_shapes = [data_container.get_datum_shape(name) for name in data_names]
    if 'input_shapes' in predictor_config:
        if input_shapes != predictor_config['input_shapes']:
            raise ValueError('conflicting values for input_shapes')
    predictor_config['input_shapes'] = input_shapes

    # transformers
    with open(args.transformers_fname) as tranformers_file:
        transformers = utils.from_yaml(tranformers_file)
    action_space = utils.from_config(data_container.get_info('environment_config')['action_space'])
    observation_space = utils.from_config(data_container.get_info('environment_config')['observation_space'])
    data_spaces = dict(zip([*sensor_names, 'action'], [*observation_space.spaces, action_space]))

    for data_name, transformer in transformers.items():
        for nested_transformer in utils.transformer.get_all_transformers(transformer):
            if isinstance(nested_transformer, (utils.transformer.NormalizerTransformer,
                                               utils.transformer.DepthImageTransformer)):
                if data_name in data_spaces:
                    nested_transformer.space = data_spaces[data_name]

    # with open(args.image_transformer_fname) as image_tranformer_file:
    #     image_transformer = utils.from_yaml(image_tranformer_file)
    # action_space = utils.from_config(data_container.get_info('environment_config')['action_space'])
    # observation_space = utils.from_config(data_container.get_info('environment_config')['observation_space'])
    # data_spaces = dict(zip([*sensor_names, 'action'], [*observation_space.spaces, action_space]))
    # transformers = OrderedDict()
    # for data_name in data_names:
    #     data_space = data_spaces[data_name]
    #     if data_name.endswith('image'):
    #         transpose = (2, 0, 1)
    #     else:
    #         transpose = None
    #     if data_name.endswith('depth_image'):
    #         scale = 1.0 / data_space.high
    #         offset = 0.0
    #         exponent = -1.0
    #     else:
    #         scale = 2.0 / (data_space.high - data_space.low)
    #         offset = -scale * (data_space.low + data_space.high) / 2.0
    #         exponent = 1.0
    #     ops_transformer = utils.transformer.OpsTransformer(scale=scale, offset=offset, exponent=exponent, transpose=transpose)
    #     if data_name.endswith('image'):
    #         transformers[data_name] = \
    #             utils.transformer.CompositionTransformer([image_transformer, ops_transformer])
    #     else:
    #         transformers[data_name] = ops_transformer

    # # transformers
    # transformers = [utils.config.from_config(transformer_config) for transformer_config in (predictor_config.get('transformers') or [])]
    # image_transformer, vel_transformer = transformers
    # # image transformer
    # image_sot_transformer = None
    # try:
    #     image_transformers = image_transformer.transformers
    # except AttributeError:
    #     image_transformers = [image_transformer]
    # for transformer in image_transformers:
    #     if isinstance(transformer, utils.transformeniar.ScaleOffsetTransposeTransformer):
    #         image_sot_transformer = transformer
    # if image_sot_transformer is not None:
    #     if image_sot_transformer.scale is None:
    #         image_sot_transformer.scale = 2.0/255.0
    #     if image_sot_transformer.offset is None:
    #         image_sot_transformer.offset = -1.0
    #     if image_sot_transformer.transpose is None:
    #         image_sot_transformer.transpose = (2, 0, 1)
    #
    # import spaces
    # box = spaces.Box(0, 255, (480, 640), dtype=np.uint8)
    # scale = 2.0 / (box.high - box.low)
    # offset = -scalegenera * (box.low + box.high) / 2.0
    #
    # scale = 2.0 / (action_space.high - action_space.low)
    # offset = -scale * (action_space.low + action_space.high) / 2.0
    #
    # action_space = utils.from_config(data_container.get_info('simulator_config')['action_space'])
    # observation_space = utils.from_config(data_container.get_info('simulator_config')['observation_space'])
    #
    # # velocity transformer
    # if isinstance(vel_transformer, utils.transformer.ScaleOffsetTransposeTransformer):
    #     with utils.container.MultiDataContainer(data_fnames) as data_container:
    #         vel_limits = [data_container.get_info('simulator_config')['action_space']['low'],
    #                       data_container.get_info('simulator_config')['action_space']['high']]
    #         # vel_limits = data_container.get_info('simulator_config')['action_spage']
    #     vel_min, vel_max = (np.asarray(limit) for limit in vel_limits)
    #     # TODO: write get_normalizer_transformer for action_space
    #     vel_min = np.append(vel_min, [vel_min[-1]]*2)
    #     vel_max = np.append(vel_max, [vel_max[-1]]*2)
    #     if vel_transformer.scale is None:
    #         vel_transformer.scale = utils.math_utils.divide_nonzero(2.0, vel_max - vel_min)
    #     if vel_transformer.offset is None:
    #         vel_transformer.offset = -vel_transformer.scale * (vel_min + vel_max) / 2.0

    # TODO: better way to map data and input names
    if 'input_names' not in predictor_config:
        predictor_config['input_names'] = ['x', 'u']
    data_to_input_name = dict(zip(data_names, predictor_config['input_names']))
    predictor_config['transformers'] = utils.get_config({data_to_input_name[k]: v for k, v in transformers.items() if k in data_names})

    if 'name' not in predictor_config:
        predictor_config['name'] = os.path.splitext(os.path.split(args.predictor_fname)[1])[0]
    feature_predictor = utils.config.from_config(predictor_config)

    if not args.no_train:
        feature_predictor.train(args.solver_fname)

    return

    if args.visualize:
        # TODO
        val_data_gen = utils.generator.DataGenerator(solver_config['val_data_fname'],
                                                     data_names=data_names,
                                                     transformers=transformers if args.visualize == 1 else None,
                                                     once=True,
                                                     batch_size=0,
                                                     shuffle=False)
        if args.visualize == 1:
            for image_curr, vel, image_next in val_data_gen:
                image_next_pred = feature_predictor.next_maps(image_curr, vel, preprocessed=True)[0]
                image_pred_error = (image_next_pred - image_next)/2.0
                # TODO: how to pass transformer
                done, key = utils.visualization.visualize_images_callback(
                    image_curr, image_next_pred, image_next, image_pred_error,
                    window_name=feature_predictor.name,
                    image_transformer=transformers[0],  # TODO: pass different transformer for each image?
                    vis_scale=args.vis_scale, delay=0)
                if done:
                    break
            cv2.destroyAllWindows()
        elif args.visualize > 1:
            for image_curr, vel, image_next in val_data_gen:
                feature_predictor.plot(image_curr, vel, image_next, preprocessed=False)


if __name__ == "__main__":
    main()
