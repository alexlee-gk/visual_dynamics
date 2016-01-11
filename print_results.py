from __future__ import division

import argparse
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_hdf5_fname', type=str)
    parser.add_argument('group_or_dset_keys', nargs='*', type=str, default=[])

    args, remaining = parser.parse_known_args()

    success_thresholds = dict()
    for key_or_threshold in remaining:
        if key_or_threshold.startswith('--'):
            key = key_or_threshold[2:]
            if key not in success_thresholds:
                success_thresholds[key] = []
        else:
            threshold = float(key_or_threshold)
            success_thresholds[key].append(threshold)

    f = h5py.File(args.results_hdf5_fname, 'r')

    def print_group_or_dset(group_or_dset_key, group_or_dset, success_thresholds=None, level=1):
        if type(group_or_dset) == h5py.Group:
            group_key, group = group_or_dset_key, group_or_dset
            print '\t'*level + group_key + ':'
            for key, value in group.items():
                print_group_or_dset(key, value, level=level+1)
        elif type(group_or_dset) == h5py.Dataset:
            dset_key, dset = group_or_dset_key, group_or_dset
            if success_thresholds and dset_key in success_thresholds:
                thresholds = success_thresholds[dset_key]
                for threshold in thresholds:
                    print '\t'*level + dset_key + ' < ' + str(threshold) + ':', (dset[()] < threshold).mean()
            else:
                print '\t'*level + dset_key + ':', dset[()]
        else:
            raise

    for group_key, group in f.items():
        print group_key +  ':'
        for group_or_dset_key, group_or_dset in group.items():
            all_keys = args.group_or_dset_keys + success_thresholds.keys()
            if all_keys and group_or_dset_key not in all_keys: # keys were specified yet current key wasn't
                continue
            print_group_or_dset(group_or_dset_key, group_or_dset, success_thresholds=success_thresholds)

if __name__ == "__main__":
    main()
