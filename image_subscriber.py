from __future__ import division

import argparse
import numpy as np
import rospy
import sensor_msgs
import cv2
from cv_bridge import CvBridge, CvBridgeError
import h5py
import util

class ImageSubscriberAndRandomController:
    def __init__(self, **kwargs):
        self.model_name = kwargs['model_name']
        self.num_steps = kwargs['num_steps']
        self.vel_max = np.asarray(kwargs['vel_max'])
        self.pos_min = np.asarray(kwargs['pos_min'])
        self.pos_max = np.asarray(kwargs['pos_max'])
        self.rescale_factor = kwargs['rescale_factor']
        self.visualize = kwargs['visualize']
        
        if kwargs['output'] is not None:
            self.f = h5py.File(kwargs['output'], "a")
        else:
            self.f = None
        self.step = 0

        if self.visualize:
            cv2.namedWindow("Image window", 1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.model_name + '/rgb/image_raw', sensor_msgs.msg.Image, self.callback)
        self.image = None
        self.image_prev = None
        self.vel = None
        self.vel_prev = None
    
    def callback(self, data):
        # get image
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError, e:
            print e
            self.image_sub.unregister()
            print "Error in receiving or converting the image, shutting down"
            rospy.signal_shutdown("Error, shutting down")
            return
        height, width = self.image.shape[:2]
        self.image = cv2.resize(self.image, (int(width/self.rescale_factor), int(height/self.rescale_factor)))

        if self.step != 0 and np.any(self.vel_prev) and np.allclose(self.image_prev, self.image): # non-zero vel and image hasn't changed
            return

        # generate and apply action
        self.vel = (2*np.random.random(3) - 1) * self.vel_max
        pose = util.get_model_pose(self.model_name)
        self.pos = np.asarray([pose.position.x, pose.position.y, pose.position.z])
        pos_next = np.clip(self.pos + self.vel, self.pos_min, self.pos_max)
        self.vel = pos_next - self.pos # recompute vel because of clipping
        pose.position.x, pose.position.y, pose.position.z = pos_next
        util.set_model_pose(self.model_name, pose)

        # visualization
        if self.visualize:
            cv2.imshow("Image window", self.image)
            cv2.waitKey(1)

        # save data
        if self.f is not None:
            image_std = util.standarize(self.image)
            data_keys = ["image_curr", "image_next", "image_diff", "vel", "pos"]
            image_shape = (self.num_steps, ) + image_std.T.shape
            data_shapes = [image_shape,  image_shape, image_shape, (self.num_steps,  (self.vel_max != 0).sum()), (self.num_steps,  3)]
            for data_key, data_shape in zip(data_keys, data_shapes):
                if data_key in self.f:
                    if self.f[data_key].shape != data_shape:
                        self.image_sub.unregister()
                        print "Error, file with different structure exists, shutting down"
                        rospy.signal_shutdown("Error, shutting down")
                        return
                else:
                    self.f.create_dataset(data_key, data_shape)
            if self.step != 0:
                image_prev_std = util.standarize(self.image_prev)
                self.f["image_next"][self.step-1] = image_std.T
                self.f["image_diff"][self.step-1] = image_std.T - image_prev_std.T
            if self.step != self.num_steps:
                self.f["image_curr"][self.step] = image_std.T
                self.f["vel"][self.step] = self.vel[self.vel_max != 0] # exclude axes with fixed position
                self.f["pos"][self.step] = self.pos
        
        if self.step == self.num_steps:
            self.image_sub.unregister()
            print "Collected all data, shutting down"
            rospy.signal_shutdown("Collected all data, shutting down")
            return
        
        self.image_prev = self.image
        self.vel_prev = self.vel
        self.step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='asus_camera')
    parser.add_argument('--num_steps', '-n', type=int, default=100)
    parser.add_argument('--vel_max', type=float, nargs=3, default=[.1, 0, .1], metavar=tuple([xyz + '_vel_max' for xyz in 'xyz']))
    parser.add_argument('--pos_min', type=float, nargs=3, default=[-5, 6, .5], metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=[5, 6, 5.5], metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--rescale_factor', '-r', type=float, default=8, metavar='r', help='rescale image by 1/r')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    
    args = parser.parse_args()
    
    rospy.init_node('image_subscriber', anonymous=True, log_level=rospy.INFO)

    pos0 = (np.asarray(args.pos_min) + np.asarray(args.pos_max))/2
    util.set_model_pose(args.model_name, util.create_pose(pos0, 0,0,np.pi/2))
    pose = util.get_model_pose(args.model_name)
    assert pose.position.x == pos0[0]
    assert pose.position.y == pos0[1]
    assert pose.position.z == pos0[2]
    
    ImageSubscriberAndRandomController(**vars(args))
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
