from __future__ import division, print_function

import argparse
import os
import pickle

import cv2
import matplotlib
import numpy as np

from visual_dynamics.utils.config import from_yaml
from visual_dynamics.utils.transformer import ImageTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('model_images_fname', type=str, help='pickle file with a dictionary containing image, next_image, target_image and action')
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor = from_yaml(predictor_file)

    if args.model_images_fname.endswith('.pkl'):
        with open(args.model_images_fname, 'rb') as model_images_file:
            model_images = pickle.load(model_images_file, encoding='latin1')

        image_names = ['image', 'next_image', 'target_image']
        images = [model_images[image_name] for image_name in image_names]
        os.makedirs(os.path.join(args.output_dir, 'image'), exist_ok=True)
        for image_name, image in zip(image_names, images):
            image_fname = os.path.join(args.output_dir, 'image', '%s.jpg' % image_name)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_fname, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        x_names = [image_name.replace('image', 'x') for image_name in image_names]
        image_transformer = ImageTransformer(crop_size=(256, 256))
        images = [image_transformer.preprocess(image) for image in images]
        xs = [predictor.preprocess([image])[0] for image in images]
        os.makedirs(os.path.join(args.output_dir, 'x'), exist_ok=True)
        for x_name, x in zip(x_names, xs):
            x_fname = os.path.join(args.output_dir, 'x', '%s.jpg' % x_name)
            x = x.transpose((1, 2, 0)).astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            cv2.imwrite(x_fname, x, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        feature_names = [[image_name.replace('image', y_name) for y_name in predictor.feature_name] for image_name in image_names]
        features = [predictor.feature([image]) for image in images]
        image = image_transformer.preprocess(images[0])
        action = model_images['action']
        feature_names.append(predictor.next_feature_name)
        features.append(predictor.next_feature([image, action]))
    else:
        image = cv2.imread(args.model_images_fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feature_names = [predictor.feature_name]
        features = [predictor.feature([image])]

    feature_limits = [None] * len(predictor.feature_name)
    for feature in features:
        for i, (y, feature_limit) in enumerate(zip(feature, feature_limits)):
            if feature_limit is None:
                feature_limit = [y.min(axis=(1, 2)), y.max(axis=(1, 2))]
            else:
                feature_limit = [np.minimum(feature_limit[0], y.min(axis=(1, 2))),
                                 np.maximum(feature_limit[1], y.max(axis=(1, 2)))]
            feature_limits[i] = feature_limit
    os.makedirs(os.path.join(args.output_dir, 'y'), exist_ok=True)
    for feature_name, feature in zip(feature_names, features):
        for y_name, y, y_limit in zip(feature_name, feature, feature_limits):
            y = (y - y_limit[0][:, None, None]) / (y_limit[1] - y_limit[0])[:, None, None]
            y = matplotlib.cm.viridis(y)
            y = (y * 255.0).astype(np.uint8)
            for i_slice, y_slice in enumerate(y):
                y_slice_fname = os.path.join(args.output_dir, 'y', '%s_%03d.jpg' % (y_name, i_slice))
                y_slice = cv2.cvtColor(y_slice, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(y_slice_fname, y_slice, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    main()
