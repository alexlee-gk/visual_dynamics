from __future__ import division

import argparse
import numpy as np
import cv2
import caffe
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proto_file', type=str, help='e.g. val.prototxt')
    parser.add_argument('model_file', type=str, help='e.g. *.caffemodel')
    parser.add_argument('--rescale_factor', type=int, default=10)
    parser.add_argument('--draw_vel', type=int, default=0)
   
    args = parser.parse_args()

    caffe.set_mode_cpu()
    net = caffe.Net(args.proto_file,
                    args.model_file,
                    caffe.TEST)

    cv2.namedWindow("Image window", 1)
    
    done = False
    while not done:
        blob_dict = net.forward(blobs=['image_curr', 'vel', 'image_diff', 'y_diff_pred'])
        assert blob_dict['loss'].ndim == 0
        print 'loss', float(blob_dict['loss'])
        # batch_size = net.blobs['image_curr'].data.shape[0]
        for image_curr_data, vel_data, image_diff_data, image_diff_pred_data in \
            zip(blob_dict['image_curr'], 
                blob_dict['vel'], 
                blob_dict['image_diff'], 
                blob_dict['y_diff_pred'].reshape(blob_dict['image_curr'].shape)):
            try:
                vis_image_gt = util.create_vis_image(image_curr_data, vel_data, image_diff_data, rescale_factor=args.rescale_factor, draw_vel=args.draw_vel)
                vis_image_pred = util.create_vis_image(image_curr_data, vel_data, image_diff_pred_data, rescale_factor=args.rescale_factor, draw_vel=args.draw_vel)
                vis_image = np.r_[vis_image_gt, vis_image_pred]
    
                cv2.imshow("Image window", vis_image)
                key = cv2.waitKey(0)
                key &= 255
                if key == 27 or key == ord('q'):
                    done = True
                    break
            except KeyboardInterrupt:
                done = True
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
