from __future__ import division

import os
from collections import OrderedDict
import numpy as np
import cv2
import h5py
import util


class DataContainer(object):
    def __init__(self, fname, num_data=None, write=False):
        self.fname = fname
        self.file = h5py.File(self.fname, 'a' if write else 'r')
        self.write = write
        self.num_data = self._require_value('num_data', num_data)
        self.num_trajs = self._require_value('num_trajs', self.num_data)
        self.num_steps = self._require_value('num_steps', 1)

    def add_group(self, group_name, group_dict):
        self._check_writable()
        g = self.file.require_group(group_name)
        for key, value in group_dict.items():
            if key in g:
                g[key][...] = value
            else:
                g[key] = value

    def get_group(self, group_name):
        g = self.file[group_name]
        group_dict = dict()
        for key, value in g.items():
            group_dict[key] = value[()]
        return group_dict

    def add_datum(self, datum_iter, datum_dict):
        self._check_writable()
        self._check_datum_iter_range(datum_iter)
        num_datum = self.num_trajs * self.num_steps
        for name, value in datum_dict.items():
            shape = (num_datum, ) + value.shape
            dset = self.file.require_dataset(name, shape, value.dtype, exact=True)
            dset[datum_iter] = value

    def get_datum(self, datum_iter, datum_names):
        self._check_datum_iter_range(datum_iter)
        datum_dict = OrderedDict()
        for name in datum_names:
            datum_dict[name] = self.file[name][datum_iter][()]
        return datum_dict

    def close(self):
        self.file.close()

    def _check_writable(self):
        if not self.write:
            raise RuntimeError('cannot add datum since data is not writable')

    def _check_datum_iter_range(self, datum_iter):
        if not (0 <= datum_iter < self.num_data):
            raise IndexError('datum index out of range')

    def _require_value(self, name, value=None):
        if value is None:
            if 'metadata' not in self.file:
                raise ValueError('%s was not specified and metadata does not exist'%name)
            metadata = self.get_group('metadata')
            if name not in metadata:
                raise ValueError('%s was not specified and %s does not exist in metadata'%(name, name))
            value = metadata[name][()]
        else:
            if 'metadata' in self.file:
                metadata = self.get_group('metadata')
                if name in metadata and value != metadata[name][()]:
                    raise ValueError('file has %d as %s but %d was given'%(metadata[name][()], name, value))
            if self.write:
                self.add_group('metadata', {name: value})
        return value

    def shuffle(self):
        inds = None
        for key, dataset in self.file.iteritems():
            if type(dataset) != h5py.Dataset:
                continue
            if inds is None:
                inds = np.arange(dataset.shape[0])
                np.random.shuffle(inds)
            else:
                assert len(inds) == dataset.shape[0]
            self.file[key][:] = dataset[()][inds]


class TrajectoryDataContainer(DataContainer):
    def __init__(self, fname, num_trajs=None, num_steps=None, write=False):
        self.fname = fname
        self.file = h5py.File(self.fname, 'a' if write else 'r')
        self.write = write
        self.num_trajs = self._require_value('num_trajs', num_trajs)
        self.num_steps = self._require_value('num_steps', num_steps)
        self.num_data = self._require_value('num_data', self.num_trajs * self.num_steps)

    def add_datum(self, *iters_datum_dict):
        iters, datum_dict = iters_datum_dict[:-1], iters_datum_dict[-1]
        datum_iter = self._get_datum_iter(*iters)
        super(TrajectoryDataContainer, self).add_datum(datum_iter, datum_dict)

    def get_datum(self, *iters_datum_names):
        iters, datum_names = iters_datum_names[:-1], iters_datum_names[-1]
        datum_iter = self._get_datum_iter(*iters)
        return super(TrajectoryDataContainer, self).get_datum(datum_iter, datum_names)

    def _check_traj_step_iter_range(self, traj_iter, step_iter):
        if not (0 <= traj_iter < self.num_trajs):
            raise IndexError('trajectory index out of range')
        if not (0 <= step_iter < self.num_steps):
            raise IndexError('step index out of range')

    def _get_datum_iter(self, *iters):
        if len(iters) == 2:
            traj_iter, step_iter = iters
            self._check_traj_step_iter_range(traj_iter, step_iter)
            datum_iter = traj_iter * self.num_steps + step_iter
        elif len(iters) == 1:
            datum_iter, = iters
        else:
            raise TypeError('_get_datum_iter() takes 2 or 3 arguments (%d given)'%len(iters))
        return datum_iter

    def shuffle(self):
        raise NotImplementedError


class ImageDataContainer(DataContainer):
    def __init__(self, fname, num_data=None, write=False):
        super(ImageDataContainer, self).__init__(fname, num_data=num_data, write=write)
        data_dir = self.required_data_dir(fname)
        self.image_fname_fmt = os.path.join(data_dir, '%s_%0{:d}d.jpg'.format(len(str(self.num_data-1))))

    def add_datum(self, datum_iter, datum_dict, fmt_args=None):
        other_dict = dict([item for item in datum_dict.items() if not item[0].startswith('image')])
        super(ImageDataContainer, self).add_datum(datum_iter, other_dict)
        image_dict = dict([item for item in datum_dict.items() if item[0].startswith('image')])
        for image_name, image in image_dict.items():
            if fmt_args is not None:
                image_fname = self.image_fname_fmt%((image_name,) + fmt_args)
            else:
                image_fname = self.image_fname_fmt%(image_name, datum_iter)
            cv2.imwrite(image_fname, util.image_from_obs(image), [cv2.IMWRITE_JPEG_QUALITY, 90])

    def get_datum(self, datum_iter, datum_names, fmt_args=None):
        other_names = [name for name in datum_names if not name.startswith('image')]
        other_dict = super(ImageDataContainer, self).get_datum(datum_iter, other_names)
        image_names = [name for name in datum_names if name.startswith('image')]
        image_dict = OrderedDict()
        for image_name in image_names:
            if fmt_args is not None:
                image_fname = self.image_fname_fmt%((image_name,) + fmt_args)
            else:
                image_fname = self.image_fname_fmt%((image_name,) + datum_iter)
            if not os.path.isfile(image_fname):
                raise RuntimeError('image file %s does not exist'%image_fname)
            image = util.obs_from_image(cv2.imread(image_fname))
            image_dict[image_name] = image
        # reorder items to follow the order of datum_names
        datum_dict = OrderedDict()
        for datum_name in datum_names:
            if datum_name.startswith('image'):
                datum_dict[datum_name] = image_dict[datum_name]
            else:
                datum_dict[datum_name] = other_dict[datum_name]
        return datum_dict

    def _require_data_dir(self, fname):
        data_dir, _ = os.path.splitext(fname)
        if self.write:
            if os.path.exists(data_dir):
                if util.yes_or_no('directory %s already exists, do you want to its contents fist?'%data_dir):
                    for f in os.listdir(data_dir):
                        os.remove(os.path.join(data_dir, f))
                else:
                    raise RuntimeError('directory %s should be empty.'%data_dir)
            else:
                os.makedirs(data_dir)
        return data_dir

    def shuffle(self):
        raise NotImplementedError


class ImageTrajectoryDataContainer(TrajectoryDataContainer, ImageDataContainer):
    def __init__(self, fname, num_trajs=None, num_steps=None, write=False):
        super(ImageTrajectoryDataContainer, self).__init__(fname, num_trajs=num_trajs, num_steps=num_steps, write=write)
        data_dir = self._require_data_dir(fname)
        self.image_fname_fmt = os.path.join(data_dir, '%s_%0{:d}d_%0{:d}d.jpg'.format(len(str(self.num_trajs-1)), len(str(self.num_steps-1))))

    def add_datum(self, *iters_datum_dict):
        iters, datum_dict = iters_datum_dict[:-1], iters_datum_dict[-1]
        datum_iter = self._get_datum_iter(*iters)
        ImageDataContainer.add_datum(self, datum_iter, datum_dict, fmt_args=iters)

    def get_datum(self, *iters_datum_names):
        iters, datum_names = iters_datum_names[:-1], iters_datum_names[-1]
        datum_iter = self._get_datum_iter(*iters)
        return ImageDataContainer.get_datum(self, datum_iter, datum_names, fmt_args=iters)
