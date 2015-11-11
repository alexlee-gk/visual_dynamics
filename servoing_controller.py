from __future__ import division

import argparse
import numpy as np
import rospy
import tf
import cv2
import caffe
import util
from image_subscriber_controller import ImageSubscriberAndController

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

class ImageSubscriberAndServoingController(ImageSubscriberAndController):
    def __init__(self, image_targets, **kwargs):
        super(ImageSubscriberAndServoingController, self).__init__(**kwargs)
        self.image_targets = image_targets

        caffe.set_mode_cpu()
        self.net = caffe.Net(kwargs['proto_file'],
                             kwargs['model_file'],
                             caffe.TEST)

        self.alpha = 0.1

        self.visualize = kwargs['visualize']
        if self.visualize:
            cv2.namedWindow("Image window", 1)

    def generate_initial_transform(self):
        pos0 = (self.pos_min + self.pos_max)/2
        quat0 = tf.transformations.quaternion_from_euler(0, 0, np.pi/2)
        return (pos0, quat0)

    def image_callback(self, image, pos, traj_iter, step_iter):
        image_target = self.image_targets[traj_iter]
        y = util.standarize(image).flatten()
        y0 = util.standarize(image_target).flatten()
        # print np.linalg.norm(y0 - y)

        # use model to optimize for action       
        image_data = util.standarize(image).T
        self.net.blobs['image_curr'].data[...] = image_data
        self.net.forward()
        # y_diff_pred = self.net.forward()['y_diff_pred'].flatten()
        J = compute_jacobian(self.net, 'y_diff_pred', 'vel')
        analytical_u_update = True
        if analytical_u_update:
            u = self.alpha * np.linalg.solve(J.T.dot(J), J.T.dot(y0 - y))
        else:
            gamma = 0.1
            u = np.zeros(self.net.blobs['vel'].data.flatten().shape)
            for _ in range(10):
                u -= gamma * J.T.dot(J.dot(u) - self.alpha * (y0 - y))
        # # apply velocity and get new image. replace this with simulation
        # vel_blob.data[...] = u[None, :]
        # y_diff_pred = net.forward()['y_diff_pred'].flatten()
        # y += y_diff_pred
        
        # apply action
        vel = u
        vel = self.apply_velocity(vel)
        vel = np.clip(vel, -0.1, 0.1)

        # visualization
        if self.visualize:
            vis_image = np.concatenate([image, image_target], axis=1)
            vis_image = util.resize_from_scale(vis_image, self.rescale_factor)
            cv2.imshow("Image window", vis_image)
            key = cv2.waitKey(1)
            key &= 255
            if key == 27 or key == ord('q'):
                self.shutdown("Pressed ESC or q, shutting down")
                return

class ImagePositionCollector(ImageSubscriberAndController):
    def __init__(self, **kwargs):
        super(ImagePositionCollector, self).__init__(**kwargs)
        self.images = []
        self.positions = []

    def image_callback(self, image, pos, traj_iter, step_iter):
        self.images.append(image)
        self.positions.append(pos)

#         vis_image = util.resize_from_scale(image, self.rescale_factor)
#         cv2.imshow("Image window", vis_image)
#         key = cv2.waitKey(1)
#         key &= 255
#         if key == 27 or key == ord('q'):
#             self.shutdown("Pressed ESC or q, shutting down")
#             return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proto_file', type=str, help='e.g. deploy.prototxt')
    parser.add_argument('model_file', type=str, help='e.g. *.caffemodel')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='asus_camera')
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='number of trajectories')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--pos_min', type=float, nargs=3, default=[-5, 6, .5], metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=[5, 6, 5.5], metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--rescale_factor', '-r', type=float, default=64, metavar='R', help='rescale image by 1/R')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    
    args = parser.parse_args()
    
    rospy.init_node('servoing_controller', anonymous=True, log_level=rospy.INFO)
    
    image_pos_collector = ImagePositionCollector(**dict(vars(args).items() + [('num_steps', 1), ('visualize', 0)])) # override some args
    while not image_pos_collector.done:
        try:
            rospy.sleep(1)
        except KeyboardInterrupt:
            break

    image_targets = image_pos_collector.images
    servoing_controller = ImageSubscriberAndServoingController(image_targets, **vars(args))
    while not servoing_controller.done:
        try:
            rospy.sleep(1)
        except KeyboardInterrupt:
            print "Shutting down"
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
