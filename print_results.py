from __future__ import division

import argparse
import numpy as np
import h5py
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_hdf5_fname', type=str)
    parser.add_argument('group_or_dset_keys', nargs='*', type=str, default=[])
    parser.add_argument('--print_with_tabs', '-t', action='store_true')

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

    def print_group_or_dset(group_or_dset_key, group_or_dset, table_data, success_thresholds=None, level=1):
        if type(group_or_dset) == h5py.Group:
            group_key, group = group_or_dset_key, group_or_dset
            print '\t'*level + group_key + ':'
            table_data[group_key] = OrderedDict()
            for key, value in group.items():
                print_group_or_dset(key, value, table_data[group_key], level=level+1)
        elif type(group_or_dset) == h5py.Dataset:
            dset_key, dset = group_or_dset_key, group_or_dset
            if success_thresholds and dset_key in success_thresholds:
                thresholds = success_thresholds[dset_key]
                for threshold in thresholds:
                    dset_key_str = dset_key + ' < ' + str(threshold)
                    print '\t'*level + dset_key_str + ':', (dset[()] < threshold).mean()
                    table_data[dset_key_str] = (dset[()] < threshold).mean()
            else:
                print '\t'*level + dset_key + ':', dset[()]
                table_data[dset_key] = np.asscalar(dset[()]) if dset[()].size == 1 else dset[()]
        else:
            raise

    table_data = OrderedDict()
    for group_key, group in f.items():
        print group_key +  ':'
        table_data[group_key] = OrderedDict()
        for group_or_dset_key, group_or_dset in group.items():
            all_keys = args.group_or_dset_keys + success_thresholds.keys()
            if all_keys and group_or_dset_key not in all_keys: # keys were specified yet current key wasn't
                continue
            print_group_or_dset(group_or_dset_key, group_or_dset, table_data[group_key], success_thresholds=success_thresholds)

    headers = table_data.values()[0].keys()
    if args.print_with_tabs:
        max_key_length = max([len(key) for key in table_data.keys()])
        row_format ="{:%d}"%(max_key_length) + "\t{}" * (len(headers))
        print row_format.format("", *headers)
        for group_key, group_table_data in table_data.items():
            print row_format.format(group_key, *group_table_data.values())
    else:
        from tabulate import tabulate
        print tabulate([[group_key] + group_table_data.values() for group_key, group_table_data in table_data.items()], headers=headers)

if __name__ == "__main__":
    main()
