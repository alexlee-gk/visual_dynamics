from __future__ import division, print_function

import contextlib
import os
import threading
import time

if not hasattr(contextlib, 'ExitStack'):
    import contextlib2 as contextlib
import numpy as np
try:
    import queue
except ImportError:
    import Queue as queue

from visual_dynamics.utils.container import ImageDataContainer
from visual_dynamics.utils.transformer import Transformer, OpsTransformer, ImageTransformer, CompositionTransformer


# generator_queue copied from Keras library: https://github.com/fchollet/keras/blob/master/keras/models.py
def generator_queue(generator, max_q_size=10, wait_time=0.05, nb_worker=1):
    q = queue.Queue()
    _stop = threading.Event()

    def data_generator_task():
        while not _stop.is_set():
            try:
                if q.qsize() < max_q_size:
                    try:
                        generator_output = next(generator)
                        # indices_generator
                    except StopIteration:
                        _stop.set()
                        break
                    q.put(generator_output)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                raise

    generator_threads = [threading.Thread(target=data_generator_task)
                         for _ in range(nb_worker)]

    for thread in generator_threads:
        thread.daemon = True
        thread.start()

    return q, _stop, generator_threads


class ParallelGenerator(object):
    def __init__(self, generator, max_q_size=10, wait_time=0.05, nb_worker=1):
        self.wait_time = wait_time
        self.data_gen_queue, self._data_stop, self.generator_threads = \
            generator_queue(generator, max_q_size=max_q_size, wait_time=wait_time, nb_worker=nb_worker)
        self._size = generator.size

    def __iter__(self):
        return self

    def __next__(self):
        while not self._data_stop.is_set() or not self.data_gen_queue.empty() or \
                any([thread.is_alive() for thread in self.generator_threads]):
            if not self.data_gen_queue.empty():
                return self.data_gen_queue.get()
            else:
                time.sleep(self.wait_time)
        raise StopIteration

    def next(self):
        # python 2 compatible
        return self.__next__()

    def __del__(self):
        self._data_stop.set()
        for thread in self.generator_threads:
            thread.join()

    @property
    def size(self):
        return self._size


class DataGenerator(object):
    def __init__(self, container_fnames, data_name_offset_pairs, transformers=None, once=False, batch_size=0, shuffle=False, dtype=None):
        """
        Iterate through all the data once or indefinitely. The data from
        contiguous files are treated as if they are contiguous. All of the
        returned minibatches contain batch_size data points. If shuffle=True,
        the data is iterated in a random order, and this order differs for each
        pass of the data.

        Note: this is not as efficient as it could be when shuffle=False since
        each data point is retrieved one by one regardless of the value of
        shuffle.

        A batch_size of 0 denotes to return data of batch size 1 but with the
        leading singleton dimensioned squeezed.
        """
        if isinstance(container_fnames, str):
            container_fnames = [container_fnames]
        self._container_fnames = [os.path.abspath(fname) for fname in container_fnames]
        self._data_name_offset_pairs = data_name_offset_pairs
        self.transformers_dict = transformers or dict()
        self.once = once
        self._batch_size = None
        self._squeeze = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = dtype
        self._lock = threading.Lock()

        offset_limits = dict()
        for data_name, offset in data_name_offset_pairs:
            offset_min, offset_max = offset_limits.get(data_name, (np.inf, -np.inf))
            if isinstance(offset, int):
                offset_min = min(offset, offset_min)
                offset_max = max(offset, offset_max)
            elif isinstance(offset, slice):
                assert offset.start < offset.stop
                offset_min = min(offset.start, offset_min)
                offset_max = max(offset.stop, offset_max)
            elif isinstance(offset, (tuple, list)):
                offset_min = min(offset_min, *offset)
                offset_max = max(offset_max, *offset)
            else:
                raise ValueError("offset should be int, slice, tuple or list, but %s was given" % offset)
            offset_limits[data_name] = (offset_min, offset_max)
        # shift the offsets so that the minimum of all is zero
        offset_all_min = min([offset_min for (offset_min, offset_max) in offset_limits.values()])
        for data_name, (offset_min, offset_max) in offset_limits.items():
            offset_limits[data_name] = (offset_min - offset_all_min,
                                        offset_max - offset_all_min)
        with contextlib.ExitStack() as stack:
            containers = [stack.enter_context(ImageDataContainer(fname)) for fname in self._container_fnames]
            num_steps_per_traj = []
            num_steps_per_container = []
            num_trajs_per_container = []
            for container in containers:
                data_name_to_data_sizes = {}
                for data_name, (offset_min, offset_max) in offset_limits.items():
                    num_trajs, num_steps = container.get_data_shape(data_name)
                    data_name_to_data_sizes[data_name] = np.array([num_steps - offset_max] * num_trajs)
                data_sizes = np.array(list(data_name_to_data_sizes.values())).min(axis=0)
                num_steps_per_traj.extend(data_sizes)
                num_steps_per_container.append(data_sizes.sum())
                num_trajs_per_container.append(len(data_sizes))
        self._num_steps_per_traj = num_steps_per_traj
        self._num_steps_per_container = num_steps_per_container
        self._num_trajs_per_container = num_trajs_per_container
        self._num_steps_per_traj_cs = np.r_[0, np.cumsum(num_steps_per_traj)]
        self._num_steps_per_container_cs = np.r_[0, np.cumsum(num_steps_per_container)]
        self._num_trajs_per_container_cs = np.r_[0, np.cumsum(num_trajs_per_container)]
        assert self._num_steps_per_traj_cs[-1] == self._num_steps_per_container_cs[-1]
        self._excerpt_generator = self._get_excerpt_generator()

    @property
    def batch_size(self):
        if self._batch_size == 1 and self._squeeze:
            batch_size = 0
        else:
            batch_size = self._batch_size
        return batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        if batch_size == 0:
            self._batch_size = 1
            self._squeeze = True
        else:
            self._batch_size = batch_size
            self._squeeze = False

    @property
    def squeeze(self):
        return self._squeeze

    def _get_local_inds(self, all_ind):
        container_ind = np.searchsorted(self._num_steps_per_container_cs[1:], all_ind, side='right')
        all_traj_iter = np.searchsorted(self._num_steps_per_traj_cs[1:], all_ind, side='right')
        step_iter = all_ind - self._num_steps_per_traj_cs[all_traj_iter]
        traj_iter = all_traj_iter - self._num_trajs_per_container_cs[container_ind]
        return container_ind, traj_iter, step_iter

    def _get_excerpt_generator(self):
        indices = []
        continue_extending = True
        while True:
            if len(indices) < self._batch_size and continue_extending:
                if self.shuffle:
                    new_indices = np.random.permutation(self.size)
                else:
                    new_indices = np.arange(self.size)
                indices.extend(new_indices)
                if self.once:
                    continue_extending = False
            excerpt = np.asarray(indices[0:self._batch_size])
            del indices[0:self._batch_size]
            yield excerpt

    def __iter__(self):
        return self

    def __next__(self):
        with contextlib.ExitStack() as stack:
            containers = [stack.enter_context(ImageDataContainer(fname)) for fname in self._container_fnames]
            with self._lock:
                excerpt = next(self._excerpt_generator)
            if len(excerpt) == 0:
                raise StopIteration
            batch_data = []
            for data_name, offset in self._data_name_offset_pairs:
                transformer = self.transformers_dict.get(data_name, Transformer())
                datum = None  # initialize later to use dtype of first single_datum
                for i, all_ind in enumerate(excerpt):
                    container_ind, traj_iter, step_iter = self._get_local_inds(all_ind)
                    if isinstance(offset, int):
                        offsets = [offset]
                    elif isinstance(offset, slice):
                        offsets = np.arange(offset.start, offset.stop, offset.step)
                    single_datum_list = []
                    for int_offset in offsets:
                        single_datum = containers[container_ind].get_datum(traj_iter, step_iter + int_offset, data_name)
                        single_datum = np.asarray(transformer.preprocess(single_datum), dtype=self.dtype)
                        single_datum_list.append(single_datum)
                    single_datum = np.asarray(single_datum_list)
                    if isinstance(offset, int):
                        single_datum = np.squeeze(single_datum, axis=0)
                    if datum is None:
                        datum = np.empty(((len(excerpt),) + single_datum.shape), dtype=single_datum.dtype)
                    datum[i, ...] = single_datum
                batch_data.append(datum)
            if self.squeeze:
                batch_data = [np.squeeze(datum, axis=0) for datum in batch_data]
            return tuple(batch_data)

    def next(self):
        # python 2 compatible
        return self.__next__()

    @property
    def size(self):
        """
        Possible number of data points that can be returned (accounting for offsets).
        """
        return self._num_steps_per_traj_cs[-1]


def iterate_minibatches_generic(data, once=False, batch_size=0, shuffle=False):
    if batch_size == 0:
        non_zero_batch_size = 1
        squeeze = True
    else:
        non_zero_batch_size = batch_size
        squeeze = False
    size = len(data[0])
    assert all(len(datum) == size for datum in data)
    indices = []
    continue_extending = True
    while indices or continue_extending:
        if len(indices) < non_zero_batch_size and continue_extending:
            if shuffle:
                new_indices = np.random.permutation(size)
            else:
                new_indices = np.arange(size)
            indices.extend(new_indices)
            if once:
                continue_extending = False
        excerpt = np.asarray(indices[0:non_zero_batch_size])
        del indices[0:non_zero_batch_size]
        batch_data = [(datum[excerpt] if isinstance(datum, np.ndarray)
                       else [datum[ind] for ind in excerpt]) for datum in data]
        if squeeze:
            batch_data = [np.squeeze(batch_datum, axis=0) for batch_datum in batch_data]
        yield batch_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('container_fname', nargs='+', type=str)
    args = parser.parse_args()

    image_transformer = CompositionTransformer(
        [ImageTransformer(scale_size=0.125, crop_size=(32, 32)),
         OpsTransformer(scale=2.0 / 255.0, offset=-1.0, transpose=(2, 0, 1))])
    action_transformer = OpsTransformer(scale=0.1)
    transformers = {'image': image_transformer, 'action': action_transformer}

    data_name_offset_pairs = [('image', 0), ('action', 0), ('image', 1)]
    generator = DataGenerator(args.container_fname,
                              data_name_offset_pairs=data_name_offset_pairs,
                              transformers=transformers,
                              batch_size=32, shuffle=True, once=True)
    generator = ParallelGenerator(generator, nb_worker=4)
    time.sleep(1.0)
    start_time = time.time()
    for i, batch_data in zip(range(4), generator):
        print(batch_data[0].shape)
    print(time.time() - start_time)


if __name__ == "__main__":
    main()
