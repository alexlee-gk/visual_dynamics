from __future__ import division

import argparse
from collections import OrderedDict
import cv2
import h5py
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_fname', type=str)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--image_names', type=str, nargs='+', default=['image_curr', 'image_diff'])

    args = parser.parse_args()

    with h5py.File(args.hdf5_fname, 'r') as hdf5_file:
        dsets = tuple(hdf5_file[image_name] for image_name in args.image_names)
        if args.reverse:
            dsets = tuple(dset[()][::-1] for dset in dsets)
        for images in zip(*dsets):
            images = OrderedDict(zip(args.image_names, images))
            if 'image_diff' in images:
                images['image_next'] = images['image_curr'] + images.pop('image_diff')
            vis_image, done = util.visualize_images_callback(*images.values(), vis_scale=args.vis_scale, delay=0)
            if done:
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
