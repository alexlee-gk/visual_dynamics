from __future__ import division

import argparse
import numpy as np
import cv2
import caffe
import util
from generate_data_square import ImageCollectorAndController, SquareSimulator

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

class ImageCollectorAndServoingController(ImageCollectorAndController):
    def __init__(self, image_targets, **kwargs):
        super(ImageCollectorAndServoingController, self).__init__(**kwargs)
        self.image_targets = image_targets

        caffe.set_mode_cpu()
        self.net = caffe.Net(kwargs['proto_file'],
                             kwargs['model_file'],
                             caffe.TEST)

        self.alpha = 0.1

        self.visualize = kwargs['visualize']
        if self.visualize:
            cv2.namedWindow("Image window", 1)
            self.vis_rescale_factor = kwargs['vis_rescale_factor']

    def generate_initial_position(self):
        pos0 = (self.sim.pos_min + self.sim.pos_max)/2
        return pos0

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
        
#         pos0 = self.sim.pos
#         image0 = util.standarize(self.sim.image).flatten()
#         assert np.allclose(image0, y)
#         diffs = []
#         for i in range(2):
#             self.sim.pos = pos0
#             velp = np.zeros(2)
#             velp[i] += 1
#             velp = self.sim.apply_velocity(velp)
#             imagep = util.standarize(self.sim.image).flatten()
#             self.sim.pos = pos0
#             velm = np.zeros(2)
#             velm[i] -= 1
#             velm = self.sim.apply_velocity(velm)
#             imagem = util.standarize(self.sim.image).flatten()
#             diffs.append((imagep - imagem) / (velp[i] - velm[i]))
#         self.sim.pos = pos0
#         J = np.asarray(diffs).T
        
        analytical_u_update = True
        if analytical_u_update:
            try:
                u = self.alpha * np.linalg.solve(J.T.dot(J), J.T.dot(y0 - y))
            except:
                u = np.zeros(2)
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
        vel = self.sim.apply_velocity(vel)
        vel = np.clip(vel, -1, 1)

        # visualization
        if self.visualize:
            vis_image = np.concatenate([image, image_target], axis=1)
            vis_image = util.resize_from_scale(vis_image, self.vis_rescale_factor)
            cv2.imshow("Image window", vis_image)
            key = cv2.waitKey(1)
            key &= 255
            if key == 27 or key == ord('q'):
                print "Pressed ESC or q, exiting"
                self.done = True
                return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('proto_file', type=str, help='e.g. deploy.prototxt')
    parser.add_argument('model_file', type=str, help='e.g. *.caffemodel')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='number of trajectories')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--image_size', '-s', type=int, nargs=2, default=[7, 10], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    parser.add_argument('--vis_rescale_factor', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    
    args = parser.parse_args()
    
    sim = SquareSimulator(args.image_size, args.square_length)

    image_targets = []
    for _ in range(args.num_trajs):
        pos0 = np.asarray(sim.pos_min + np.random.random(2) * (sim.pos_max - sim.pos_min), dtype=int)
        sim.pos = pos0
        image_targets.append(sim.image)

    image_collector_ctrl = ImageCollectorAndServoingController(image_targets, **dict(vars(args).items() + [('sim', sim)]))
    image_collector_ctrl.start_loop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
