from __future__ import division

import argparse
import cv2
import h5py
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--height', type=float, default=480)
    parser.add_argument('--draw_vel', type=int, default=1)
    
    args = parser.parse_args()

    f = h5py.File(args.file, 'r+')
    
    cv2.namedWindow("Image window", 1)

    for image_curr_data, vel_data, image_diff_data, image_next_data in zip(f['image_curr'], f['vel'], f['image_diff'], f['image_next']):
        try:
            vis_image = util.create_vis_image(image_curr_data, vel_data, image_diff_data, height=args.height, draw_vel=args.draw_vel)

            # # debug that image addition is done correctly
            # import numpy as np
            # image_next_std = image_next_data.T
            # image_next = util.destandarize(image_next_std).astype(np.uint8)
            # image_next = util.resize_from_height(image_next, args.height)
            # image_next = np.repeat(image_next[:, :, None], 3, axis=2) if image_next.ndim == 2 else image_next
            # vis_image = np.concatenate([vis_image, image_next], axis=1)
            
            cv2.imshow("Image window", vis_image)
            key = cv2.waitKey(0)
            key &= 255
            if key == 27 or key == ord('q'):
                break
        except KeyboardInterrupt:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
