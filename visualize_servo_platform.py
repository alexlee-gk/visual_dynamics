from __future__ import division

import argparse
import numpy as np
import time
import itertools
import cv2
import h5py
import simulator
import controller
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--dof_min', type=float, nargs='+', default=None)
    parser.add_argument('--dof_max', type=float, nargs='+', default=None)
    parser.add_argument('--vel_min', type=float, nargs='+', default=None)
    parser.add_argument('--vel_max', type=float, nargs='+', default=None)
    parser.add_argument('--image_scale', '-f', type=float, default=0.15)
    parser.add_argument('--pwm_channels', '-c', nargs='+', type=int, default=(0, 1))
    parser.add_argument('--camera_id', '-i', type=str, default='0')

    args = parser.parse_args()
    args.dof_min = args.dof_min or (230, 220)
    args.dof_max = args.dof_max or (610, 560)
    args.vel_min = args.vel_min or (-50, -50)
    args.vel_max = args.vel_max or (50, 50)

    sim = simulator.ServoPlatform([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                  image_scale=args.image_scale, crop_size=args.image_size,
                                  pwm_channels=args.pwm_channels,
                                  camera_id=args.camera_id)

    # go to all combinations of dof limits
    for dof_values in itertools.product(*zip(args.dof_min, args.dof_max)):
        sim.reset(np.asarray(dof_values))
        if args.visualize:
            while True:
                image = sim.observe()
                vis_image, done, key = util.visualize_images_callback(image, vis_scale=args.vis_scale, ret_key=True)
                if done or key == 32:
                    break
            if done:
                break

    # apply max and min velocities
    sim.reset(np.asarray(args.dof_min))
    for vel_limit in [args.vel_max, args.vel_min]:
        action = np.asarray(vel_limit)
        while np.any(action):
            if args.visualize:
                while True:
                    image = sim.observe()
                    vis_image, done, key = util.visualize_images_callback(image, vis_scale=args.vis_scale, ret_key=True)
                    if done or key == 32:
                        break
                if done:
                    break
            action = sim.apply_action(action)
    
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
