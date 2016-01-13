from __future__ import division

import argparse
import cv2
import h5py
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_fname', type=str)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--reverse', action='store_true')

    args = parser.parse_args()

    with h5py.File(args.hdf5_fname, 'r') as hdf5_file:
        dsets = (hdf5_file['image_curr'], hdf5_file['vel'], hdf5_file['image_diff'])
        if args.reverse:
            dsets = tuple(dset[()][::-1] for dset in dsets)
        for image_curr, vel, image_diff in zip(*dsets):
            image_next = image_curr + image_diff
            vis_image, done = util.visualize_images_callback(image_curr, image_next, vis_scale=args.vis_scale, delay=0)
            if done:
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
