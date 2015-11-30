from __future__ import division

import argparse
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_hdf5_fname', type=str)
    parser.add_argument('group_or_dset_keys', nargs='*', type=str, default=[])

    args = parser.parse_args()

    f = h5py.File(args.results_hdf5_fname, 'r')

    def print_group_or_dset(group_or_dset_key, group_or_dset, level=1):
        if type(group_or_dset) == h5py.Group:
            group_key, group = group_or_dset_key, group_or_dset
            print '\t'*level + group_key + ':'
            for key, value in group.items():
                print_group_or_dset(key, value, level=level+1)
        elif type(group_or_dset) == h5py.Dataset:
            dset_key, dset = group_or_dset_key, group_or_dset
            print '\t'*level + dset_key + ':', dset[()]
        else:
            raise

    for group_key, group in f.items():
        print group_key +  ':'
        for group_or_dset_key, group_or_dset in group.items():
            if args.group_or_dset_keys and group_or_dset_key not in args.group_or_dset_keys:
                continue
            print_group_or_dset(group_or_dset_key, group_or_dset)

if __name__ == "__main__":
    main()
