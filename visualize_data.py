from __future__ import division

import argparse
import cv2
import h5py
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('--rescale_factor', type=int, default=10)
    parser.add_argument('--draw_vel', type=int, default=0)
    
    args = parser.parse_args()

    f = h5py.File(args.file, 'r+')
    
    cv2.namedWindow("Image window", 1)

    for image_curr_data, vel_data, image_diff_data, image_next_data in zip(f['image_curr'], f['vel'], f['image_diff'], f['image_next']):
        try:
            vis_image = util.create_vis_image(image_curr_data, vel_data, image_diff_data, rescale_factor=args.rescale_factor, draw_vel=args.draw_vel)
            
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
