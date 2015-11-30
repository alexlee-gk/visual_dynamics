from __future__ import division

import argparse
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_hdf5_fname', type=str)
    parser.add_argument('dset_keys', nargs='*', type=str, default=[])

    args = parser.parse_args()

    f = h5py.File(args.results_hdf5_fname, 'r')
#     import IPython as ipy; ipy.embed()
    for group_key, group in f.items():
        print group_key
        if args.dset_keys:
            for dset_key in args.dset_keys:
                if dset_key in group:
                    print '\t' + dset_key + ':', group[dset_key][()]
        else:
            for dset_key, dset in group.items():
                print '\t' + dset_key + ':', dset[()]

if __name__ == "__main__":
    main()
