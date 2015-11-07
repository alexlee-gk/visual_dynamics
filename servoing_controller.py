from __future__ import division

import argparse
import numpy as np
import rospy
import sensor_msgs
import cv2
from cv_bridge import CvBridge, CvBridgeError
import util
import caffe

def compute_jacobian(net, output, input):
    assert output in net.outputs
    assert input in net.inputs
    input_data = net.blobs[input].data
    assert input_data.ndim == 2
    assert input_data.shape[0] == 1
    output_data = net.blobs[output].data
    assert output_data.ndim == 2
    assert output_data.shape[0] == 1
    doutput_dinput = np.array([net.backward(y_diff_pred=e[None,:])[input].flatten() for e in np.eye(output_data.shape[1])])
    return doutput_dinput

class ServoingController:
    def __init__(self, net, **kwargs):
        self.net = net
        self.model_name = kwargs['model_name']
        self.pos_min = np.asarray(kwargs['pos_min'])
        self.pos_max = np.asarray(kwargs['pos_max'])
        self.rescale_factor = kwargs['rescale_factor']
        self.visualize = kwargs['visualize']
        
        self.step = 0

        if self.visualize:
            cv2.namedWindow("Image window", 1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.model_name + '/rgb/image_raw', sensor_msgs.msg.Image, self.callback)
        self.image = None
        self.image_prev = None
        self.image_target = None
        self.y0 = None
        self.vel = None
        self.vel_prev = None

        self.alpha = 1.0
    
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
        self.image = util.resize_from_scale(self.image, 1.0/self.rescale_factor)
        if self.image_target is None:
            self.image_target = np.zeros(self.image.shape)
            self.y0 = self.image_target.flatten()

        if self.step != 0 and np.any(self.vel_prev) and np.allclose(self.image_prev, self.image): # non-zero vel and image hasn't changed
            return

        y = self.image.flatten()
        print np.linalg.norm(self.y0 - y)

        # use model to optimize for action       
        self.net.blobs['image_curr'].data[...] = y.reshape(self.net.blobs['image_curr'].data.shape)
        self.net.forward()
        # y_diff_pred = self.net.forward()['y_diff_pred'].flatten()
        J = compute_jacobian(self.net, 'y_diff_pred', 'vel')
        analytical_u_update = True
        if analytical_u_update:
            u = self.alpha * np.linalg.solve(J.T.dot(J), J.T.dot(self.y0 - y))
        else:
            gamma = 0.1
            u = np.zeros(self.net.blobs['vel'].data.flatten().shape)
            for _ in range(10):
                u -= gamma * J.T.dot(J.dot(u) - self.alpha * (self.y0 - y))
        self.vel = u
        # # apply velocity and get new image. replace this with simulation
        # vel_blob.data[...] = u[None, :]
        # y_diff_pred = net.forward()['y_diff_pred'].flatten()
        # y += y_diff_pred
        
        # apply action
        self.vel = np.array([u[0], 0.0, u[1]])
        pose = util.get_model_pose(self.model_name)
        self.pos = np.asarray([pose.position.x, pose.position.y, pose.position.z])
        pos_next = np.clip(self.pos + self.vel, self.pos_min, self.pos_max)
        self.vel = pos_next - self.pos # recompute vel because of clipping
        pose.position.x, pose.position.y, pose.position.z = pos_next
        util.set_model_pose(self.model_name, pose)

        # visualization
        key = None
        if self.visualize:
            cv2.imshow("Image window", util.resize_from_scale(self.image, self.rescale_factor))
            key = cv2.waitKey(1)
            key &= 255
        
        if key is not None and (key == 27 or key == ord('q')):
            self.image_sub.unregister()
            print "Pressed ESC or q, shutting down"
            rospy.signal_shutdown("Pressed ESC or q, shutting down")
            return
        
        self.image_prev = self.image
        self.vel_prev = self.vel
        self.step += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proto_file', type=str, help='e.g. deploy.prototxt')
    parser.add_argument('model_file', type=str, help='e.g. *.caffemodel')
    parser.add_argument('--model_name', type=str, default='asus_camera')
    parser.add_argument('--pos_min', type=float, nargs=3, default=[-5, 6, .5], metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=[5, 6, 5.5], metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--rescale_factor', '-r', type=float, default=64, metavar='R', help='rescale image by 1/R')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    
    args = parser.parse_args()
    
    rospy.init_node('servoing_controller', anonymous=True, log_level=rospy.INFO)

    pos0 = (np.asarray(args.pos_min) + np.asarray(args.pos_max))/2
    util.set_model_pose(args.model_name, util.create_pose(pos0, 0,0,np.pi/2))
    pose = util.get_model_pose(args.model_name)
    assert pose.position.x == pos0[0]
    assert pose.position.y == pos0[1]
    assert pose.position.z == pos0[2]
    
    caffe.set_mode_cpu()
    net = caffe.Net(args.proto_file,
                         args.model_file,
                         caffe.TEST)
   
    ServoingController(net, **vars(args))
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
