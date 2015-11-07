from __future__ import division

import argparse
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    
    args = parser.parse_args()

    f = h5py.File(args.file, 'r+')
    inds = None
    for key, dataset in f.iteritems():
        if inds is None:
            inds = np.arange(dataset.shape[0])
            np.random.shuffle(inds)
        else:
            assert len(inds) == dataset.shape[0]
        f[key][:] = dataset[()][inds]

if __name__ == "__main__":
    main()