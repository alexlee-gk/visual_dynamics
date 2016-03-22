import numpy as np
import cv2
import utils


class Transformer(utils.config.ConfigObject):
    def preprocess(self, data):
        return data

    def deprocess(self, data):
        return data

    def preprocess_shape(self, shape):
        return shape

    def deprocess_shape(self, shape):
        return shape


class ScaleOffsetTransposeTransformer(Transformer):
    def __init__(self, scale=1.0, offset=0.0, transpose=None):
        """
        Scales and offset the numerical values of the input data.
        """
        self.scale = scale
        self.offset = offset
        self.transpose = transpose
        self._data_dtype = None

    def preprocess(self, data):
        self._data_dtype = data.dtype
        data = self.scale * data + self.offset
        if self.transpose:
            data = np.transpose(data, self.transpose)
        return data

    def deprocess(self, data):
        data = ((data - self.offset) * (1.0 / self.scale)).astype(self._data_dtype)
        if self.transpose:
            data = np.transpose(data, self.transpose_inv)
        return data

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

    def get_config(self):
        config = {'class': self.__class__,
                  'scale': self.scale,
                  'offset': self.offset,
                  'transpose': self.transpose}
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                config[k] = v.tolist()
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
            crop_origin = image_shape/2
            if self.crop_offset is not None:
                crop_origin += self.crop_offset
            crop_corner = crop_origin - self.crop_size/2
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
                shape = tuple([shape[0], *self.crop_size])
            else:
                shape = tuple([*self.crop_size, shape[-1]])
        return shape

    def get_config(self):
        config = {'class': self.__class__,
                  'scale_size': self.scale_size,
                  'crop_size': self.crop_size,
                  'crop_offset': self.crop_offset}
        for k, v in config.items():
            if isinstance(v, np.ndarray):
                config[k] = v.tolist()
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

    def get_config(self):
        config = {'class': self.__class__,
                  'transformers': [transformer.get_config() for transformer in self.transformers]}
        return config

    @classmethod
    def from_config(cls, config):
        transformers = [utils.config.from_config(transformer_config) for transformer_config in config.get('transformers', [])]
        return cls(transformers)


def main():
    transformers = []
    transformers.append(ScaleOffsetTransposeTransformer(scale=2.0/255.0, offset=-1.0, transpose=(2, 0, 1)))
    transformers.append(ImageTransformer(scale_size=0.125, crop_size=[32, 32], crop_offset=[0, 0]))
    transformers.append(CompositionTransformer(transformers.copy()))

    # get config from the transformer and reconstruct a transformer from this config
    re_transformers = []
    for transformer in transformers:
        yaml_string = transformer.to_yaml()
        print(yaml_string)
        re_transformer = utils.config.from_yaml(yaml_string)
        re_transformers.append(re_transformer)

    # make sure the config from both transformers are the same
    for transformer, re_transformer in zip(transformers, re_transformers):
        print(transformer.to_yaml() == re_transformer.to_yaml())


if __name__ == "__main__":
    main()
