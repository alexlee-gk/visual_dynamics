from __future__ import division

import os
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
    parser.add_argument('--solverstate_fname', '--ss', type=str, default=None)
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--num_channel', type=int, help='net parameter')
    parser.add_argument('--y1_dim', type=int, help='net parameter')
    parser.add_argument('--y2_dim', type=int, help='net parameter')
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--output_hdf5_fname', '-o', type=str)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--output_image_dir', type=str)
    parser.add_argument('--vel_max', '-m', type=float, default=None)
    parser.add_argument('--image_size', type=int, nargs=2, default=None, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--simulator', '-s', type=str, default=None, choices=('square', 'ogre'))
    # square simulator
    parser.add_argument('--square_length', '-l', type=int, default=None, help='required to be odd')
    # ogre simulator
    parser.add_argument('--pos_min', type=float, nargs=3, default=None, metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=None, metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--image_scale', '-f', type=float, default=None)

    args = parser.parse_args()

    with h5py.File(args.train_hdf5_fname, 'r') as hdf5_file:
        if 'sim_args' in hdf5_file:
            for arg_name, arg in hdf5_file['sim_args'].items():
                if args.__dict__[arg_name] is None:
                    args.__dict__[arg_name] = arg[...] if arg.shape else np.asscalar(arg[...])
    args.pos_min = np.asarray(args.pos_min)
    args.pos_max = np.asarray(args.pos_max)

    if args.simulator == 'square':
        sim = simulator.SquareSimulator(args.image_size, args.square_length, args.vel_max)
    elif args.simulator== 'ogre':
        sim = simulator.OgreSimulator(args.pos_min, args.pos_max, args.vel_max,
                                      image_scale=args.image_scale, crop_size=args.image_size)
    else:
        raise

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
                                                                      pretrained_file=args.pretrained_fname,
                                                                      postfix=args.postfix)
        elif args.predictor == 'bilinear_constrained_net':
            feature_predictor = predictor.BilinearConstrainedNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname,
                                                                                 pretrained_file=args.pretrained_fname,
                                                                                 postfix=args.postfix)
        else:
            inputs = ['image_curr', 'vel']
            input_shapes = predictor.NetFeaturePredictor.infer_input_shapes(inputs, None, args.train_hdf5_fname)
            outputs = ['y_diff_pred', 'y', 'image_next_pred']
            net_kwargs = dict(num_channel=args.num_channel,
                              y1_dim=args.y1_dim,
                              y2_dim=args.y2_dim)
            net_func = getattr(net, args.predictor)
            net_func_with_kwargs = lambda *args, **kwargs: net_func(*args, **dict(net_kwargs.items() + kwargs.items()))
            feature_predictor = predictor.NetFeaturePredictor(net_func_with_kwargs, inputs, input_shapes, outputs,
                                                              pretrained_file=args.pretrained_fname, postfix=args.postfix)
        solver_param = pb2.SolverParameter(solver_type=pb2.SolverParameter.ADAM,
                                           base_lr=0.001, gamma=0.99,
                                           momentum=0.9, momentum2=0.999,
                                           max_iter=args.max_iter)
#         solver_param = pb2.SolverParameter(solver_type=pb2.SolverParameter.SGD,
#                                            base_lr=0.001, gamma=0.9, stepsize=1000,
#                                            momentum=0.9,
#                                            max_iter=args.max_iter)
        if args.solverstate_fname is not None and args.solverstate_fname == 'auto':
            args.solverstate_fname = feature_predictor.get_snapshot_fname() + '_iter_' + str(args.max_iter) + '.solverstate'
        feature_predictor.train(args.train_hdf5_fname,
                                solverstate_fname=args.solverstate_fname,
                                solver_param=solver_param,
                                batch_size=32)
        val_losses = {blob_name: np.asscalar(blob.data) for blob_name, blob in feature_predictor.val_net.blobs.items() if blob_name.endswith('loss')}
        print 'val_losses', val_losses

    ctrl = controller.ServoingController(feature_predictor)

    if args.output_hdf5_fname:
        output_hdf5_file = h5py.File(args.output_hdf5_fname, 'a')
        output_hdf5_group = output_hdf5_file.require_group(feature_predictor.net_name + feature_predictor.postfix)
        val_losses_group = output_hdf5_group.require_group('val_losses')
        for key, value in val_losses.items():
            dset = val_losses_group.require_dataset(key, (1,), type(value), exact=True)
            dset[...] = value

    np.random.seed(0)
    done = False
    image_pred_errors = []
    image_errors = []
    pos_errors = []
    iter_ = 0
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
                image_next_pred = feature_predictor.predict(image, action, prediction_name='image_next_pred')
                action = sim.apply_action(action)

                # visualization
                if args.visualize or args.output_image_dir:
                    vis_image = np.concatenate([image.transpose(1, 2, 0), image_next_pred.transpose(1, 2, 0), image_target.transpose(1, 2, 0)], axis=1)
                    vis_image = ((vis_image + 1.0) * 255.0/2.0).astype(np.uint8)
                    if args.output_image_dir:
                        if vis_image.ndim == 2:
                            output_image = np.concatenate([vis_image]*3, axis=2)
                        else:
                            output_image = vis_image
                        image_fname = feature_predictor.net_name + feature_predictor.postfix + '_%04d.png'%iter_
                        cv2.imwrite(os.path.join(args.output_image_dir, image_fname), output_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    vis_image = util.resize_from_scale(vis_image, args.vis_scale)
                    cv2.imshow("Image window", vis_image)
                    key = cv2.waitKey(1)
                    key &= 255
                    if key == 27 or key == ord('q'):
                        print "Pressed ESC or q, exiting"
                        done = True
                        break

            image = sim.observe()
            image_pred_error = np.linalg.norm(image_next_pred - image)
            image_pred_errors.append(image_pred_error)
            image_error = np.linalg.norm(image_target - image)
            image_errors.append(image_error)
            pos = sim.state
            pos_error = np.linalg.norm(pos_target - pos)
            pos_errors.append(pos_error)
            print 'image_pred_error:', image_pred_error
            print 'image_error:', image_error
            print 'pos_error:', pos_error
            if traj_iter == args.num_trajs-1:
                image_pred_error_mean = np.mean(image_pred_errors)
                image_error_mean = np.mean(image_errors)
                pos_error_mean = np.mean(pos_errors)
                print 'image_pred_error_mean:', image_pred_error_mean
                print 'image_error_mean:', image_error_mean
                print 'pos_error_mean:', pos_error_mean
            if args.output_hdf5_fname:
                for key, value in dict(image=image,
                                       image_pred_error=image_pred_error,
                                       image_target=image_target,
                                       image_error=image_error,
                                       state=pos,
                                       state_target=pos_target,
                                       pos_error=pos_error).items():
                    shape = (args.num_trajs, ) + value.shape
                    dset = output_hdf5_group.require_dataset(key, shape, value.dtype, exact=True)
                    dset[traj_iter] = value
                if traj_iter == args.num_trajs-1:
                    for key, value in dict(image_pred_error_mean=image_pred_error_mean,
                                           image_error_mean=image_error_mean,
                                           pos_error_mean=pos_error_mean).items():
                        dset = output_hdf5_group.require_dataset(key, (1,), type(value), exact=True)
                        dset[...] = value

            if done:
                break
        except KeyboardInterrupt:
            break
        iter_ += 1

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
