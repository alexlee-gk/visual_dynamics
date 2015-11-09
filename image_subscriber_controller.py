from __future__ import division

import argparse
import numpy as np
import rospy
import sensor_msgs
import cv2
from cv_bridge import CvBridge, CvBridgeError
import h5py
import util

class ImageSubscriberAndController(object):
    def __init__(self, **kwargs):
        self.model_name = kwargs['model_name']
        self.num_trajs = kwargs['num_trajs']
        self.num_steps = kwargs['num_steps']
        self.pos_min = np.asarray(kwargs['pos_min'])
        self.pos_max = np.asarray(kwargs['pos_max'])
        self.rescale_factor = kwargs['rescale_factor']
        
        if kwargs['output'] is not None:
            self.f = h5py.File(kwargs['output'], "a")
            self.data_iter = 0
        else:
            self.f = None

        self.traj_iter = 0
        self.step_iter = 0

        self.bridge = CvBridge()
        self.image_prev = None
        self.pos_prev = None
        self.vel_prev = None
        self.vel = np.zeros(3)
        self.pos0 = self.generate_initial_position()

        self.image_sub = rospy.Subscriber(self.model_name + '/rgb/image_raw', sensor_msgs.msg.Image, self.callback)
        self.done = False

    def shutdown(self, msg):
        self.image_sub.unregister()
        print msg
        rospy.signal_shutdown(msg)

    def generate_initial_position(self):
        pos0 = self.pos_min + np.random.random(3) * (self.pos_max - self.pos_min)
        return pos0

    def apply_velocity(self, vel):
        assert not np.any(self.vel) # can only apply velocity once per callback
        pos, quat = util.transform_from_pose(self.pose)
        pos_next = np.clip(pos + vel, self.pos_min, self.pos_max)
        self.vel = vel = pos_next - pos # recompute vel because of clipping
        util.set_model_pose(self.model_name, util.create_pose_from_transform((pos + vel, quat)))
        return vel

    def image_callback(self, image, pos, traj_iter, step_iter):
        pass
    
    def callback(self, data):
        # get position
        self.pose = util.get_model_pose(self.model_name)
        pos, quat = util.transform_from_pose(self.pose)

        # set initial position
        if self.step_iter == 0 and not np.all(pos == self.pos0):
            util.set_model_pose(self.model_name, util.create_pose_from_transform((self.pos0, quat)))
            self.skip_frames = 5
            return
        # skip frames to ensure that the pose has been set
        if self.step_iter == 0 and self.skip_frames > 0:
            self.skip_frames -= 1
            return

        # get image
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError, e:
            print e
            self.shutdown("Error in receiving or converting the image, shutting down")
            return
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width/self.rescale_factor), int(height/self.rescale_factor)))

        # check images and positions are consistent with previous velocity
        if self.step_iter > 0 and np.any(self.vel_prev) and np.all(self.image_prev == image): # non-zero vel and image hasn't changed
            return
        if self.step_iter > 0 and not np.all(pos == (self.pos_prev + self.vel_prev)):
            return

        self.image_callback(image, pos, self.traj_iter, self.step_iter)
        vel = self.vel
        self.vel = np.zeros(3)

        # save data
        if self.f is not None:
            image_std = util.standarize(image)
            data_keys = ["image_curr", "image_next", "image_diff", "vel", "pos"]
            num_data = self.num_trajs * self.num_steps
            image_shape = (num_data, ) + image_std.T.shape
            data_shapes = [image_shape,  image_shape, image_shape, (num_data,  len(vel)), (num_data, len(pos))]
            for data_key, data_shape in zip(data_keys, data_shapes):
                if data_key in self.f:
                    if self.f[data_key].shape != data_shape:
                        self.shutdown("Error, file with different structure exists, shutting down")
                        return
                else:
                    self.f.create_dataset(data_key, data_shape)
            assert self.data_iter == (self.traj_iter * self.num_steps + self.step_iter)
            if self.step_iter != 0:
                image_prev_std = util.standarize(self.image_prev)
                self.f["image_next"][self.data_iter-1] = image_std.T
                self.f["image_diff"][self.data_iter-1] = image_std.T - image_prev_std.T
            if self.step_iter != self.num_steps:
                self.f["image_curr"][self.data_iter] = image_std.T
                self.f["vel"][self.data_iter] = vel
                self.f["pos"][self.data_iter] = pos
                self.data_iter += 1
        
        self.image_prev = image
        self.pos_prev = pos
        self.vel_prev = vel
        self.step_iter += 1

        if self.step_iter == (self.num_steps + 1):
            self.traj_iter += 1
            self.step_iter = 0
            if self.traj_iter == self.num_trajs:
                print "Collected all data"
                self.image_sub.unregister()
                self.done = True
                return
            self.image_prev = None
            self.pos_prev = None
            self.vel_prev = None
            self.pos0 = self.generate_initial_position()

class ImageSubscriberAndRandomController(ImageSubscriberAndController):
    def __init__(self, **kwargs):
        super(ImageSubscriberAndRandomController, self).__init__(**kwargs)
        self.vel_max = np.asarray(kwargs['vel_max'])

        self.visualize = kwargs['visualize']
        if self.visualize:
            cv2.namedWindow("Image window", 1)

    def image_callback(self, image, pos, traj_iter, step_iter):
        # generate and apply action
        vel = (2*np.random.random(3) - 1) * self.vel_max
        vel = self.apply_velocity(vel)

        # visualization
        if self.visualize:
            vis_image = util.resize_from_scale(vis_image, self.rescale_factor)
            cv2.imshow("Image window", vis_image)
            key = cv2.waitKey(1)
            key &= 255
            if key == 27 or key == ord('q'):
                self.shutdown("Pressed ESC or q, shutting down")
                return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='asus_camera')
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--vel_max', type=float, nargs=3, default=[.1, 0, .1], metavar=tuple([xyz + '_vel_max' for xyz in 'xyz']))
    parser.add_argument('--pos_min', type=float, nargs=3, default=[-5, 6, .5], metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=[5, 6, 5.5], metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--rescale_factor', '-r', type=float, default=64, metavar='R', help='rescale image by 1/R')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    
    args = parser.parse_args()
    
    rospy.init_node('image_subscriber_controller', anonymous=True, log_level=rospy.INFO)

    image_sub_ctrl = ImageSubscriberAndRandomController(**vars(args))
    while not image_sub_ctrl.done:
        try:
            rospy.sleep(1)
        except KeyboardInterrupt:
            print "Shutting down"
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
