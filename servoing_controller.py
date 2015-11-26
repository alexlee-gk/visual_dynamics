from __future__ import division

import argparse
import numpy as np
import h5py
import cv2
import caffe
from caffe.proto import caffe_pb2 as pb2
import predictor
import util
import simulator
import controller
import net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('--predictor', '-p', type=str, default='small_action_cond_encoder_net')
    parser.add_argument('--pretrained_fname', type=str, default=None)
    parser.add_argument('--solverstate_fname', type=str, default=None)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--vel_max', '-m', type=float, default=None)
    parser.add_argument('--image_size', type=int, nargs=2, default=[84, 84], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--simulator', '-s', type=str, default='square', choices=('square', 'ogre'))
    # square simulator
    parser.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    # ogre simulator
    parser.add_argument('--pos_min', type=float, nargs=3, default=[19, 2, -10], metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=[21, 6, -6], metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--image_scale', '-f', type=float, default=0.25)

    args = parser.parse_args()
    if args.simulator == 'square':
        if args.vel_max is None:
            args.vel_max = 1.0
    elif args.simulator == 'ogre':
        if args.vel_max is None:
            args.vel_max = 0.2
    else:
        raise
    args.pos_min = np.asarray(args.pos_min)
    args.pos_max = np.asarray(args.pos_max)

    if args.predictor == 'bilinear':
        train_file = h5py.File(args.train_hdf5_fname, 'r+')
        X = train_file['image_curr'][:]
        U = train_file['vel'][:]
        feature_predictor = predictor.BilinearFeaturePredictor(X.shape[1:], U.shape[1:])
        X_dot = train_file['image_diff'][:]
        Y_dot = feature_predictor.feature_from_input(X_dot)
        feature_predictor.train(X, U, Y_dot)
    else:
        caffe.set_mode_gpu();
        caffe.set_device(0);
        if args.predictor == 'bilinear_net':
            feature_predictor = predictor.BilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname,
                                                                      pretrained_file=args.pretrained_fname)
        elif args.predictor == 'approx_bilinear_net':
            feature_predictor = predictor.ApproxBilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname,
                                                                            pretrained_file=args.pretrained_fname)
        elif args.predictor == 'action_cond_encoder_net':
            feature_predictor = predictor.ActionCondEncoderNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname,
                                                                               pretrained_file=args.pretrained_fname)
        else:
            inputs = ['image_curr', 'vel']
            input_shapes = predictor.NetFeaturePredictor.infer_input_shapes(inputs, None, args.train_hdf5_fname)
            output = 'y_diff_pred'
            feature_predictor = predictor.NetFeaturePredictor(getattr(net, args.predictor), inputs, input_shapes, output,
                                                              pretrained_file=args.pretrained_fname, postfix=args.postfix)
        solver_param = pb2.SolverParameter(solver_type=pb2.SolverParameter.ADAM,
                                           base_lr=0.001, gamma=0.99,
                                           momentum=0.9, momentum2=0.999,
                                           max_iter=60000)
        feature_predictor.train(args.train_hdf5_fname,
                                solverstate_fname=args.solverstate_fname,
                                solver_param=solver_param,
                                batch_size=32)

    if args.simulator == 'square':
        sim = simulator.SquareSimulator(args.image_size, args.square_length, args.vel_max)
    elif args.simulator== 'ogre':
        sim = simulator.OgreSimulator(args.pos_min, args.pos_max, args.vel_max,
                                      image_scale=args.image_scale, crop_size=args.image_size)
    else:
        raise
    ctrl = controller.ServoingController(feature_predictor)

    done = False
    for traj_iter in range(args.num_trajs):
        try:
            pos_target = sim.sample_state()
            sim.reset(pos_target)
            image_target = sim.observe()
            ctrl.set_target_obs(image_target)

            pos_init = (sim.pos_min + sim.pos_max) / 2.0
            sim.reset(pos_init)
            for step_iter in range(args.num_steps):
                image = sim.observe()
                action = ctrl.step(image)
                action = sim.apply_action(action)

                # visualization
                if args.visualize:
                    vis_image = np.concatenate([image.transpose(1, 2, 0), image_target.transpose(1, 2, 0)], axis=1)
                    vis_image = util.resize_from_scale((vis_image * 255.0).astype(np.uint8), args.vis_scale)
                    cv2.imshow("Image window", vis_image)
                    key = cv2.waitKey(100)
                    key &= 255
                    if key == 27 or key == ord('q'):
                        print "Pressed ESC or q, exiting"
                        done = True
                        break
            if done:
                break
        except KeyboardInterrupt:
            break

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
