import argparse
import cv2
import numpy as np
import yaml
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_fnames', nargs='+', type=str)
    parser.add_argument('--val_data_fname', type=str)
    parser.add_argument('--predictor_fname', '-p', type=str, default='config/predictor/predictor.yaml')
    parser.add_argument('--solver_fname', '--sf', type=str, default='config/solver/adam.yaml')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='S', help='rescale image by S for visualization')

    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor_config = yaml.load(predictor_file)

    # input_shapes
    data_fnames = list(args.train_data_fnames)
    if args.val_data_fname is not None:
        data_fnames.append(args.val_data_fname)
    with utils.container.MultiDataContainer(data_fnames) as data_container:
        input_shapes = [data_container.get_datum_shape(name) for name in ('image', 'vel')]
    if 'input_shapes' in predictor_config:
        if input_shapes != predictor_config['input_shapes']:
            raise ValueError('conflicting values for input_shapes')
    predictor_config['input_shapes'] = input_shapes

    # transformers
    transformers = [utils.config.from_config(transformer_config) for transformer_config in (predictor_config.get('transformers') or [])]
    image_transformer, vel_transformer = transformers
    # image transformer
    image_sot_transformer = None
    try:
        image_transformers = image_transformer.transformers
    except AttributeError:
        image_transformers = [image_transformer]
    for transformer in image_transformers:
        if isinstance(transformer, utils.transformer.ScaleOffsetTransposeTransformer):
            image_sot_transformer = transformer
    if image_sot_transformer is not None:
        if image_sot_transformer.scale is None:
            image_sot_transformer.scale = 2.0/255.0
        if image_sot_transformer.offset is None:
            image_sot_transformer.offset = -1.0
        if image_sot_transformer.transpose is None:
            image_sot_transformer.transpose = (2, 0, 1)
    # velocity transformer
    if isinstance(vel_transformer, utils.transformer.ScaleOffsetTransposeTransformer):
        with utils.container.MultiDataContainer(data_fnames) as data_container:
            vel_limits = data_container.get_info('simulator_config')['dof_vel_limits']
        vel_min, vel_max = (np.asarray(limit) for limit in vel_limits)
        if vel_transformer.scale is None:
            vel_transformer.scale = utils.math_utils.divide_nonzero(2.0, vel_max - vel_min)
        if vel_transformer.offset is None:
            vel_transformer.offset = -vel_transformer.scale * (vel_min + vel_max) / 2.0
    predictor_config['transformers'] = [transformer.get_config() for transformer in transformers]

    feature_predictor = utils.config.from_config(predictor_config)

    if not args.no_train:
        feature_predictor.train(*args.train_data_fnames, val_data_fname=args.val_data_fname,
                                solver_fname=args.solver_fname)

    if args.visualize:
        data_names = ['image', 'vel']
        val_data_gen = utils.generator.ImageVelDataGenerator(args.val_data_fname,
                                                             data_names=data_names,
                                                             transformers=transformers,
                                                             once=True,
                                                             batch_size=0,
                                                             shuffle=False)
        for image_curr, vel, image_next in val_data_gen:
            image_next_pred = feature_predictor.next_maps(image_curr, vel, preprocessed=True)[0]
            image_pred_error = (image_next_pred - image_next)/2.0
            vis_image, done = utils.visualization.visualize_images_callback(
                image_curr, image_next_pred, image_next, image_pred_error,
                image_transformer=image_transformer,
                vis_scale=args.vis_scale, delay=0)
            if done:
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
