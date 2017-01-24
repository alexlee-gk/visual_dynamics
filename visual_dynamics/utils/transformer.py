from __future__ import division, print_function

import cv2
import numpy as np

from visual_dynamics.utils.config import ConfigObject, get_config, from_config


class Transformer(ConfigObject):
    def preprocess(self, data):
        return data

    def deprocess(self, data):
        return data

    def preprocess_shape(self, shape):
        return shape

    def deprocess_shape(self, shape):
        return shape


class OpsTransformer(Transformer):
    def __init__(self, scale=1.0, offset=0.0, exponent=1.0, transpose=None):
        """
        Scales and offset the numerical values of the input data.
        """
        self.scale = np.asarray(scale) if scale is not None else None
        self.offset = np.asarray(offset) if offset is not None else None
        self.exponent = np.asarray(exponent) if exponent is not None else None
        self.transpose = transpose
        self._data_dtype = None

    def preprocess(self, data):
        self._data_dtype = data.dtype
        # TODO
        if data.shape == (6,) and (self.scale.shape == (4,) or self.offset.shape == (4,)):
            data = np.append(self.scale[:3] * data[:3] + self.offset[:3], self.scale[3] * data[3:] + self.offset[3])
        else:
            data = self.scale * data + self.offset
        if self.exponent != 1.0:
            data = np.power(data, self.exponent)
        if self.transpose:
            data = np.transpose(data, self.transpose)
        return data

    def deprocess(self, data):
        if self.transpose:
            data = np.transpose(data, self.transpose_inv)
        # TODO
        if self.exponent != 1.0:
            data = np.power(data, 1.0 / self.exponent)
        if data.shape == (6,) and (self.scale.shape == (4,) or self.offset.shape == (4,)):
            data = np.append((data[:3] - self.offset[:3]) * (1.0 / self.scale[:3]),
                             (data[3:] - self.offset[3]) * (1.0 / self.scale[3]))
        else:
            data = (data - self.offset) * (1.0 / self.scale)
        if self._data_dtype == np.uint8:
            np.clip(data, 0, 255, out=data)
        return data.astype(self._data_dtype)

    def preprocess_shape(self, shape):
        if self.transpose:
            shape = tuple(shape[axis] for axis in self.transpose)
        return shape

    def deprocess_shape(self, shape):
        if self.transpose:
            shape = tuple(shape[axis] for axis in self.transpose_inv)
        return shape

    @property
    def transpose_inv(self):
        transpose_axis = zip(self.transpose, range(len(self.transpose)))
        axis_transpose_inv = sorted(transpose_axis)
        axis, transpose_inv = zip(*axis_transpose_inv)
        return transpose_inv

    def _get_config(self):
        config = super(OpsTransformer, self)._get_config()
        config.update({'scale': self.scale,
                       'exponent': self.exponent,
                       'offset': self.offset,
                       'transpose': self.transpose})
        self.convert_array_tolist(config)
        return config


class ImageTransformer(Transformer):
    def __init__(self, scale_size=None, crop_size=None, crop_offset=None):
        self.scale_size = scale_size
        self.crop_size = np.asarray(crop_size) if crop_size is not None else None
        self.crop_offset = np.asarray(crop_offset) if crop_offset is not None else None

    def preprocess(self, image):
        need_swap_channels = (image.ndim == 3 and image.shape[0] == 3)
        if need_swap_channels:
            image = image.transpose(1, 2, 0)
        if self.scale_size is not None and self.scale_size != 1.0:
            image = cv2.resize(image, (0, 0), fx=self.scale_size, fy=self.scale_size, interpolation=cv2.INTER_AREA)
        if self.crop_size is not None and tuple(self.crop_size) != image.shape[:2]:
            h, w = image_shape = np.asarray(image.shape[:2])
            crop_h, crop_w = self.crop_size
            if crop_h > h:
                raise ValueError('crop height %d is larger than image height %d (after scaling)' % (crop_h, h))
            if crop_w > w:
                raise ValueError('crop width %d is larger than image width %d (after scaling)' % (crop_w, w))
            crop_origin = image_shape // 2
            if self.crop_offset is not None:
                crop_origin += self.crop_offset
            crop_corner = crop_origin - self.crop_size // 2
            if not (np.all(np.zeros(2) <= crop_corner) and np.all(crop_corner + self.crop_size <= image_shape)):
                raise IndexError('crop indices out of range')
            image = image[crop_corner[0]:crop_corner[0] + crop_h,
                          crop_corner[1]:crop_corner[1] + crop_w,
                          ...]
        if need_swap_channels:
            image = image.transpose(2, 0, 1)
        return image

    def preprocess_shape(self, shape):
        if self.crop_size is not None:
            need_swap_channels = (len(shape) == 3 and shape[0] == 3)
            if need_swap_channels:
                shape = (shape[0],) + tuple(self.crop_size)
            else:
                shape = tuple(self.crop_size) + (shape[-1],)
        return shape

    def _get_config(self):
        config = super(ImageTransformer, self)._get_config()
        config.update({'scale_size': self.scale_size,
                       'crop_size': self.crop_size,
                       'crop_offset': self.crop_offset})
        self.convert_array_tolist(config)
        return config


class NormalizerTransformer(OpsTransformer):
    """
    Normalizes the data to be between -1 and 1.
    """
    def __init__(self, space=None, transpose=None):
        self._space = None
        self.space = space
        self.transpose = transpose

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        self._space = space
        if space is not None:
            self.scale = 2.0 / (space.high - space.low)
            self.offset = -self.scale * (space.low + space.high) / 2.0
            self.exponent = 1.0
        else:
            self.scale = 1.0
            self.offset = 0.0
            self.exponent = 1.0

    def _get_config(self):
        config = Transformer._get_config(self)  # do not get scale, offset and exponent
        config.update({'space': self.space,
                       'transpose': self.transpose})
        return config


class DepthImageTransformer(OpsTransformer):
    def __init__(self, space=None, transpose=None):
        self._space = None
        self.space = space
        self.transpose = transpose

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        self._space = space
        if space is not None:
            self.scale = 1.0 / space.high
            self.offset = 0.0
            self.exponent = -1.0
        else:
            self.scale = 1.0
            self.offset = 0.0
            self.exponent = 1.0

    def _get_config(self):
        config = Transformer._get_config(self)  # do not get scale, offset and exponent
        config.update({'space': self.space})
        return config


class CompositionTransformer(Transformer):
    def __init__(self, transformers):
        self.transformers = transformers

    def preprocess(self, data):
        for transformer in self.transformers:
            data = transformer.preprocess(data)
        return data

    def deprocess(self, data):
        for transformer in reversed(self.transformers):
            data = transformer.deprocess(data)
        return data

    def preprocess_shape(self, shape):
        for transformer in self.transformers:
            shape = transformer.preprocess_shape(shape)
        return shape

    def deprocess_shape(self, shape):
        for transformer in reversed(self.transformers):
            shape = transformer.deprocess_shape(shape)
        return shape

    def _get_config(self):
        config = super(CompositionTransformer, self)._get_config()
        config.update({'transformers': self.transformers})
        return config


def get_all_transformers(transformer):
    """
    Return all transformers contained in this transformer (including this one).
    """
    transformers = [transformer]
    if isinstance(transformer, transformer.CompositionTransformer):
        for nested_transformer in transformer.transformers:
            transformers.extend(get_all_transformers(nested_transformer))
    return transformers


def split_first_transformer(transformer):
    if isinstance(transformer, CompositionTransformer):
        first_transformer = transformer.transformers[0]
        remaining_transformers = transformer.transformers[1:]
        if len(remaining_transformers) == 0:
            transformer = Transformer()
        elif len(remaining_transformers) == 1:
            transformer, = remaining_transformers
        else:
            transformer = CompositionTransformer(remaining_transformers)
    else:
        first_transformer = transformer
        transformer = Transformer()
    return first_transformer, transformer


def extract_image_transformer(transformers):
    image_transformer = None
    for name, transformer in transformers.items():
        if name.endswith('image') or name == 'x':
            image_transformer_, transformer = split_first_transformer(transformer)
            transformers[name] = transformer
            if image_transformer:
                assert get_config(image_transformer) == get_config(image_transformer_)
            else:
                assert isinstance(image_transformer_, ImageTransformer)
                image_transformer = image_transformer_
    return image_transformer


def transfer_image_transformer(predictor_config, image_transformer=None):
    import citysim3d.utils.panda3d_util as putil
    from visual_dynamics import envs

    transformers = from_config(predictor_config['transformers'])
    image_transformer_ = extract_image_transformer(transformers)
    if image_transformer is None:
        image_transformer = image_transformer_
    predictor_config['transformers'] = get_config(transformers)

    environment_config = from_config(predictor_config['environment_config'])
    assert issubclass(environment_config['class'], envs.Panda3dEnv)
    input_names = predictor_config['input_names']
    input_shapes = predictor_config['input_shapes']
    orig_input_shape = None
    for i, (input_name, input_shape) in enumerate(zip(input_names, input_shapes)):
        if input_name.endswith('image') or input_name == 'x':
            if orig_input_shape:
                assert orig_input_shape == input_shape
            else:
                orig_input_shape = input_shape
            input_shapes[i] = image_transformer.preprocess_shape(input_shape)

    assert image_transformer.crop_offset is None or np.all(image_transformer.crop_offset == 0)
    camera_size, camera_hfov = putil.scale_crop_camera_parameters(orig_input_shape[:2][::-1], 60.0,
                                                                  scale_size=image_transformer.scale_size,
                                                                  crop_size=image_transformer.crop_size)
    environment_config['camera_size'] = camera_size
    environment_config['camera_hfov'] = camera_hfov
    predictor_config['environment_config'] = get_config(environment_config)


def main():
    transformers = []
    transformers.append(OpsTransformer(scale=2.0/255.0, offset=-1.0, transpose=(2, 0, 1)))
    transformers.append(ImageTransformer(scale_size=0.125, crop_size=[32, 32], crop_offset=[0, 0]))
    transformers.append(CompositionTransformer(transformers.copy()))

    # get config from the transformer and reconstruct a transformer from this config
    re_transformers = []
    for transformer in transformers:
        yaml_string = transformer.to_yaml()
        print(yaml_string)
        re_transformer = config.from_yaml(yaml_string)
        re_transformers.append(re_transformer)

    # make sure the config from both transformers are the same
    for transformer, re_transformer in zip(transformers, re_transformers):
        print(transformer.to_yaml() == re_transformer.to_yaml())


if __name__ == "__main__":
    main()
