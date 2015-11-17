from __future__ import division

import argparse
import numpy as np
import h5py
import cv2
import predictor
import util
from generate_data_square import ImageCollectorAndController, SquareSimulator

class ImageCollectorAndServoingController(ImageCollectorAndController):
    def __init__(self, feature_predictor, image_targets, alpha=0.1, vel_max=1.0, **kwargs):
        super(ImageCollectorAndServoingController, self).__init__(**kwargs)
        self.predictor = feature_predictor
        self.image_targets = image_targets
        self.alpha = alpha
        self.vel_max = vel_max

        self.visualize = kwargs['visualize']
        if self.visualize:
            cv2.namedWindow("Image window", 1)
            self.vis_rescale_factor = kwargs['vis_rescale_factor']

    def generate_initial_position(self):
        pos0 = (self.sim.pos_min + self.sim.pos_max)/2
        return pos0

    def image_callback(self, image, pos, traj_iter, step_iter):
        image_target = self.image_targets[traj_iter]

        x_target = np.expand_dims(image_target, axis=0).astype(np.float) / 255.0
        y_target = self.predictor.feature_from_input(x_target)
        x = np.expand_dims(image, axis=0).astype(np.float) / 255.0
        y = self.predictor.feature_from_input(x)

        # use model to optimize for action
        J = self.predictor.jacobian_control(x, None)
        try:
            u = self.alpha * np.linalg.solve(J.T.dot(J), J.T.dot(y_target - y))
        except:
            u = np.zeros(self.predictor.u_shape)

        # apply action
        vel = u
        vel = self.sim.apply_velocity(vel)
        vel = np.clip(vel, -self.vel_max, self.vel_max)

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
    parser.add_argument('--model_file', type=str, help='e.g. *.caffemodel')
    parser.add_argument('--train_hdf5_fname', type=str, default=None)
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--predictor', '-p', type=str, choices=['bilinear', 'bilinear_net', 'approx_bilinear_net'], default='bilinear')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='number of trajectories')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--image_size', '-s', type=int, nargs=2, default=[7, 10], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    parser.add_argument('--vis_rescale_factor', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--visualize', '-v', type=int, default=1)

    args = parser.parse_args()

    if args.predictor == 'bilinear':
        train_file = h5py.File(args.train_hdf5_fname, 'r+')
        X = train_file['image_curr'][:]
        U = train_file['vel'][:]
        feature_predictor = predictor.BilinearFeaturePredictor(X.shape[1:], U.shape[1:])
        if args.train:
            X_dot = train_file['image_diff'][:]
            Y_dot = feature_predictor.feature_from_input(X_dot)
            feature_predictor.train(X, U, Y_dot)
    else:
        if args.predictor == 'bilinear_net':
            feature_predictor = predictor.BilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname)
        elif args.predictor == 'approx_bilinear_net':
            feature_predictor = predictor.ApproxBilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname)
        if args.train:
            feature_predictor.train(args.train_hdf5_fname)

    sim = SquareSimulator(args.image_size, args.square_length)
    image_targets = []
    for _ in range(args.num_trajs):
        pos0 = np.asarray(sim.pos_min + np.random.random(2) * (sim.pos_max - sim.pos_min), dtype=int)
        sim.pos = pos0
        image_targets.append(sim.image)

    image_collector_ctrl = ImageCollectorAndServoingController(feature_predictor, image_targets, **dict(vars(args).items() + [('sim', sim)]))
    image_collector_ctrl.start_loop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
