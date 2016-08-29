from __future__ import division, print_function
import io
import os
import cv2
import h5py
import numpy as np
import utils


class DataContainer:
    def __init__(self, data_dir, mode='r'):
        self.data_dir = self._require_data_dir(data_dir, mode)
        self.info_file = None
        self.hdf5_file = None

        info_fname = os.path.join(self.data_dir, 'info.yaml')
        self.info_file = open(info_fname, mode)
        try:
            self.info_dict = utils.from_yaml(self.info_file) or dict()  # store info entries here and dump it only when the container is closed
        except io.UnsupportedOperation:  # file is probably empty
            self.info_dict = dict()

        self.data_shapes_dict = self.info_dict.get('data_shapes', None) or dict()
        self.datum_shapes_dict = self.info_dict.get('datum_shapes', None) or dict()
        data_fname = os.path.join(self.data_dir, 'data.h5')
        self.hdf5_file = h5py.File(data_fname, mode)

    def close(self):
        if self.info_file:
            try:
                self.add_info(data_shapes=self.data_shapes_dict)
                self.add_info(datum_shapes=self.datum_shapes_dict)
                utils.to_yaml(self.info_dict, self.info_file)
            except io.UnsupportedOperation:  # container is probably in read mode
                pass
            self.info_file.close()
            self.info_file = None
        if self.hdf5_file:
            self.hdf5_file.close()
            self.hdf5_file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def add_info(self, **info_dict):
        self.info_dict.update(**info_dict)

    def get_info(self, info_names):
        if isinstance(info_names, str):
            names = list([info_names])
        else:
            names = list(info_names)
        info = []
        for name in names:
            info.append(self.info_dict[name])
        if isinstance(info_names, str):
            info, = info
        return info

    def reserve(self, names, shape):
        if isinstance(names, str):
            names = list([names])
        else:
            names = list(names)
        try:
            shape = tuple(shape)
        except TypeError:
            shape = tuple((shape,))
        for name in names:
            if name in self.data_shapes_dict and self.data_shapes_dict[name] != shape:
                raise ValueError('unable to reserve for %s since it was already reserved with shape %s,'
                                 'but shape %s was given' % (name, self.data_shapes_dict[name], shape))
            self.data_shapes_dict[name] = shape

    def add_datum(self, *inds, **datum_dict):
        for name, value in datum_dict.items():
            if name in self.datum_shapes_dict and self.datum_shapes_dict[name] != value.shape:
                raise ValueError('unable to add datum %s with shape %s since the shape %s was expected' %
                                 (name, value.shape, self.datum_shapes_dict[name]))
            self.datum_shapes_dict[name] = value.shape
            datum_size = self.get_data_size(name)
            shape = (datum_size, ) + value.shape
            dset = self.hdf5_file.require_dataset(name, shape, value.dtype, exact=True)
            datum_ind = self._get_datum_ind(*inds, name=name)
            dset[datum_ind] = value

    def get_datum(self, *inds_and_datum_names):
        inds, datum_names = inds_and_datum_names[:-1], inds_and_datum_names[-1]
        if isinstance(datum_names, str):
            names = list([datum_names])
        else:
            names = list(datum_names)
        datum = []
        for name in names:
            datum_ind = self._get_datum_ind(*inds, name=name)
            datum.append(self.hdf5_file[name][datum_ind][()])
        if isinstance(datum_names, str):
            datum, = datum
        return datum

    def get_datum_shape(self, name):
        shape = self.datum_shapes_dict.get(name, None)
        if shape is None:
            raise ValueError('shape for name %s does not exist' % name)
        return shape

    def get_data_shape(self, name):
        shape = self.data_shapes_dict.get(name, None)
        if shape is None:
            raise ValueError('shape is not reserved for name %s' % name)
        return shape

    def get_data_size(self, name):
        return np.prod(self.get_data_shape(name))

    def _get_canonical_inds(self, *inds_and_name):
        inds, name = inds_and_name[:-1], inds_and_name[-1]
        inds = list(inds)
        shape = self.get_data_shape(name)
        for i, ind in enumerate(inds):
            if ind < 0:
                inds[i] += shape[i]
        return inds

    def _check_ind_range(self, *inds_and_name):
        inds, name = inds_and_name[:-1], inds_and_name[-1]
        shape = self.get_data_shape(name)
        if len(inds) != len(shape):
            raise IndexError('the number of indices does not match the number of dimensions of the data')
        for i, ind in enumerate(inds):
            if not (0 <= ind < shape[i]):
                raise IndexError('index at position %d is out of range for entry with name %s' % (i, name))

    def _get_datum_ind(self, *inds_and_name):
        inds, name = inds_and_name[:-1], inds_and_name[-1]
        inds = self._get_canonical_inds(*inds, name=name)
        self._check_ind_range(*inds, name=name)
        shape = self.get_data_shape(name)
        datum_ind = 0
        for ind, dim in zip(inds, shape):
            datum_ind *= dim
            datum_ind += ind
        assert 0 <= datum_ind < self.get_data_size(name)
        return datum_ind

    def _get_datum_inds(self, datum_ind, name):
        assert 0 <= datum_ind < self.get_data_size(name)
        shape = self.get_data_shape(name)
        ind = datum_ind
        inds = [-1]*len(shape)
        for i, dim in reversed(list(enumerate(shape))):
            inds[i] = ind % dim
            ind //= dim
        assert datum_ind == self._get_datum_ind(*inds, name=name)
        return tuple(inds)

    def _require_data_dir(self, data_dir, mode):
        if 'r' in mode:
            if not os.path.exists(data_dir):
                raise FileNotFoundError('data directory %s not found' % data_dir)
        elif 'a' in mode:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
        elif 'w' in mode:
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    os.remove(os.path.join(data_dir, f))
            else:
                os.makedirs(data_dir)
        elif 'x' in mode:
            if os.path.exists(data_dir):
                raise FileExistsError('data directory %s exists' % data_dir)
            else:
                os.makedirs(data_dir)
        else:
            raise ValueError('mode %s not recognized' % mode)
        return data_dir


class ImageDataContainer(DataContainer):
    def add_datum(self, *inds, **datum_dict):
        other_dict = dict([item for item in datum_dict.items() if not item[0].endswith('image')])
        super(ImageDataContainer, self).add_datum(*inds, **other_dict)
        image_dict = dict([item for item in datum_dict.items() if item[0].endswith('image')])
        for image_name, image in image_dict.items():
            if image_name in self.datum_shapes_dict and self.datum_shapes_dict[image_name] != image.shape:
                raise ValueError('unable to add datum %s with shape %s since the shape %s was expected' %
                                 (image_name, image.shape, self.datum_shapes_dict[image_name]))
            self.datum_shapes_dict[image_name] = image.shape
            image_fname = self._get_image_fname(*inds, name=image_name)
            if image.dtype == np.uint8:
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image = utils.pack_image(image)
            cv2.imwrite(image_fname, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def _get_image_fname(self, *inds_and_name, **kwargs):
        inds, name = inds_and_name[:-1], inds_and_name[-1]
        ext = kwargs.get('ext', '.jpg')
        inds = self._get_canonical_inds(*inds, name=name)
        self._check_ind_range(*inds, name=name)
        shape = self.get_data_shape(name)
        image_fmt = '%s'
        for dim in shape:
            image_fmt += '_%0{:d}d'.format(len(str(dim-1)))
        image_fmt += ext
        image_fname = image_fmt % ((name,) + inds)
        image_fname = os.path.join(self.data_dir, image_fname)
        return image_fname

    def get_datum(self, *inds_and_datum_names):
        inds, datum_names = inds_and_datum_names[:-1], inds_and_datum_names[-1]
        if isinstance(datum_names, str):
            names = list([datum_names])
        else:
            names = list(datum_names)
        other_names = [name for name in names if not name.endswith('image')]
        other_datum = super(ImageDataContainer, self).get_datum(*(inds + (other_names,)))
        image_names = [name for name in names if name.endswith('image')]
        image_datum = []
        for image_name in image_names:
            image_fname = self._get_image_fname(*inds, name=image_name)
            if not os.path.isfile(image_fname):
                raise FileNotFoundError('image file %s does not exist' % image_fname)
            image = cv2.imread(image_fname)
            if not image_name.endswith('depth_image'):
                if image.ndim == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = utils.unpack_image(image)
            image_datum.append(image)
        # reorder items to follow the order of datum_names
        datum = []
        for datum_name in names:
            if 'image' in datum_name:
                datum.append(image_datum.pop(0))
            else:
                datum.append(other_datum.pop(0))
        if isinstance(datum_names, str):
            datum, = datum
        return datum


class MultiDataContainer(DataContainer):
    """
    Light wrapper of multiple data containers to get basic information from a container while ensuring that all
    containers have the same information.
    """
    def __init__(self, data_dirs, mode='r'):
        if mode != 'r':
            raise NotImplementedError
        self.containers = [ImageDataContainer(data_dir, mode=mode) for data_dir in data_dirs]

    def close(self):
        for container in self.containers:
            container.close()

    def add_info(self, **info_dict):
        raise NotImplementedError

    def get_info(self, info_names):
        info = None
        for container in self.containers:
            if info is None:
                info = container.get_info(info_names)
            else:
                other_info = container.get_info(info_names)
                try:
                    equal = other_info == info
                except ValueError:
                    equal = (np.asarray(other_info) == np.asarray(info)).all()
                if not equal:
                    raise ValueError('infos are inconsistent across containers: %r, %r' % (info, other_info))
        return info

    def reserve(self, names, shape):
        raise NotImplementedError

    def add_datum(self, *inds, **datum_dict):
        raise NotImplementedError

    def get_datum(self, *inds_and_datum_names):
        raise NotImplementedError

    def get_datum_shape(self, name):
        shape = None
        for container in self.containers:
            if shape is None:
                shape = container.get_datum_shape(name)
            else:
                other_shape = container.get_datum_shape(name)
                if other_shape != shape:
                    raise ValueError('shapes are inconsistent across containers: %r, %r' % (shape, other_shape))
        return shape

    def get_data_shape(self, name):
        raise NotImplementedError

    def get_data_size(self, name):
        size = 0
        for container in self.containers:
            size += container.get_data_size(name)
        return size
