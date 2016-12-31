from __future__ import division, print_function
import os
import argparse
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_fnames', nargs='+', type=str)

    args = parser.parse_args()

    for fname in args.result_fnames:
        print(fname)
        fname = os.path.join(fname, 'data.h5')
        num_trajs = 100
        num_steps = 100
        try:
            with h5py.File(fname, 'r') as f:
                reward = f['reward'][:].reshape((num_trajs, num_steps))
                # pan_angle_error = f['pan_angle'][:].reshape((num_trajs, num_steps+1))
                # tilt_angle_error = f['tilt_angle'][:].reshape((num_trajs, num_steps+1))
                print(np.sqrt((reward ** 2).mean()),
                      np.sqrt((reward[:, -1] ** 2).mean()))
                # print(np.sqrt((pan_angle_error ** 2).mean()),
                #       np.sqrt((pan_angle_error[:, -1] ** 2).mean()))
                # print(np.sqrt((tilt_angle_error ** 2).mean()),
                #       np.sqrt((tilt_angle_error[:, -1] ** 2).mean()))
        except IOError:
            continue


if __name__ == '__main__':
    main()
