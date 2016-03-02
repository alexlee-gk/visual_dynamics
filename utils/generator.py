import contextlib
import threading
import queue
import time
import numpy as np
import utils


# copied from Keras library: https://github.com/fchollet/keras/blob/master/keras/models.py
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


class ParallelGenerator:
    def __init__(self, generator, max_q_size=10, wait_time=0.05, nb_worker=1):
        self.wait_time = wait_time
        self.data_gen_queue, self._data_stop, self.generator_threads = \
            generator_queue(generator, max_q_size=max_q_size, wait_time=wait_time, nb_worker=nb_worker)

    def __iter__(self):
        return self

    def __next__(self):
        while not self._data_stop.is_set() or not self.data_gen_queue.empty():
            if not self.data_gen_queue.empty():
                return self.data_gen_queue.get()
            else:
                time.sleep(self.wait_time)
        raise StopIteration

    def __del__(self):
        self._data_stop.set()
        for thread in self.generator_threads:
            thread.join()


class ImageVelDataGenerator:
    def __init__(self, *container_fnames, data_names, transformers_dict=None, once=False, batch_size=1, shuffle=False, dtype=None):
        """
        Iterate through all the data once or indefinitely. The data from contiguous files
        are treated as if they are contiguous. All of the returned minibatches
        contain batch_size data points. If shuffle=True, the data is iterated in a
        random order, and this order differs for each pass of the data.
        Note: this is not as efficient as it could be when shuffle=False since each
        data point is retrieved one by one regardless of the value of shuffle.
        """
        self._container_fnames = container_fnames
        self._image_name, self._vel_name = data_names
        self.transformers_dict = transformers_dict or dict()
        self.once = once
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dtype = dtype
        self._lock = threading.Lock()
        with contextlib.ExitStack() as stack:
            containers = [stack.enter_context(utils.container.ImageDataContainer(fname)) for fname in self._container_fnames]
            for container in containers:
                image_shape = container.get_data_shape(self._image_name)
                vel_shape = container.get_data_shape(self._vel_name)
                assert image_shape[:-1] == vel_shape[:-1]
                assert image_shape[-1] == (vel_shape[-1] + 1)
            data_sizes = [container.get_data_size(self._vel_name) for container in containers]  # use sizes of vel
            self._all_data_size = sum(data_sizes)
            self._data_sizes_cs = np.r_[0, np.cumsum(data_sizes)]
        self._excerpt_generator = self._get_excerpt_generator()

    def _get_datum(self, containers, all_ind, data_name, offset=0):  # pass in containers to prevent from opening every time this function is called
        local_ind = all_ind - self._data_sizes_cs
        container_ind = np.asscalar(np.where(local_ind >= 0)[0][-1])  # get index of last non-negative entry
        local_ind = local_ind[container_ind]
        local_inds = containers[container_ind]._get_datum_inds(local_ind, self._vel_name)  # use local_inds wrt vel indices
        if offset != 0:
            local_inds = list(local_inds)
            local_inds[-1] += offset
        return containers[container_ind].get_datum(*local_inds, data_name)

    def _get_excerpt_generator(self):
        indices = []
        continue_extending = True
        while True:
            if len(indices) < self.batch_size and continue_extending:
                if self.shuffle:
                    new_indices = np.random.permutation(self._all_data_size)
                else:
                    new_indices = np.arange(self._all_data_size)
                indices.extend(new_indices)
                if self.once:
                    continue_extending = False
            excerpt = np.asarray(indices[0:self.batch_size])
            del indices[0:self.batch_size]
            yield excerpt

    def __iter__(self):
        return self

    def __next__(self):
        with contextlib.ExitStack() as stack:
            containers = [stack.enter_context(utils.container.ImageDataContainer(fname)) for fname in self._container_fnames]
            with self._lock:
                excerpt = next(self._excerpt_generator)
            if len(excerpt) == 0:
                raise StopIteration
            batch_data = []
            for data_name, offset in [(self._image_name, 0), (self._vel_name, 0), (self._image_name, 1)]:
                transformer = self.transformers_dict.get(data_name, None) or utils.transformer.Transformer()
                datum = np.empty((len(excerpt), *transformer.preprocess_shape(containers[0].get_datum_shape(data_name))), dtype=self.dtype)
                for i, all_ind in enumerate(excerpt):
                    single_datum = self._get_datum(containers, all_ind, data_name, offset=offset)
                    datum[i, ...] = np.asarray(transformer.preprocess(single_datum), dtype=self.dtype)
                batch_data.append(datum)
            return tuple(batch_data)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('container_fname', nargs='+', type=str)
    args = parser.parse_args()

    image_transformer = utils.transformer.CompositionTransformer(
        [utils.transformer.ImageTransformer(scale_size=0.125, crop_size=(32, 32)),
         utils.transformer.ScaleOffsetTransposeTransformer(scale=2.0/255.0, offset=-1.0, transpose=(2, 0, 1))])
    vel_transformer = utils.transformer.ScaleOffsetTransposeTransformer(scale=0.1)
    transformers_dict = dict(image=image_transformer, vel=vel_transformer)
    generator = ImageVelDataGenerator(*args.container_fname, data_names=['image', 'vel'], transformers_dict=transformers_dict, batch_size=32, shuffle=True, once=True)
    generator = ParallelGenerator(generator, nb_worker=4)
    time.sleep(1.0)
    start_time = time.time()
    for i, batch_data in zip(range(4), generator):
        print(batch_data[0].shape)
    print(time.time() - start_time)


if __name__ == "__main__":
    main()
