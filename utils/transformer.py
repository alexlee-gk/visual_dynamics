import numpy as np
import cv2


class Transformer:
    def preprocess(self, data):
        return data

    def deprocess(self, data):
        return data

    def preprocess_shape(self, shape):
        return shape

    def deprocess_shape(self, shape):
        return shape

    @staticmethod
    def create(transformer, **transformer_args):
        try:
            Transformer = globals()[transformer]
        except KeyError:
            raise ValueError('transformer %s is not supported' % transformer)
        transformer = Transformer(**transformer_args)
        return transformer


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
            data = np.transpose(data, np.arange(len(self.transpose))[list(self.transpose)])
        return data


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

    def deprocess(self, image):
        raise NotImplementedError

    def preprocess_shape(self, shape):
        if self.crop_size is not None:
            need_swap_channels = (len(shape) == 3 and shape[0] == 3)
            if need_swap_channels:
                shape = tuple([shape[0], *self.crop_size])
            else:
                shape = tuple([*self.crop_size, shape[0]])
        return shape

    def deprocess_shape(self, shape):
        raise NotImplementedError


class CompositionTransformer(Transformer):
    def __init__(self, transformers):
        self.transformers = transformers

    def preprocess(self, data):
        for transformer in self.transformers:
            data = transformer.preprocess(data)
        return data

    def deprocess(self, data):
        for transformer in self.transformers:
            data = transformer.deprocess(data)
        return data

    def preprocess_shape(self, shape):
        for transformer in self.transformers:
            shape = transformer.preprocess_shape(shape)
        return shape

    def deprocess_shape(self, shape):
        for transformer in self.transformers:
            shape = transformer.deprocess_shape(shape)
        return shape
