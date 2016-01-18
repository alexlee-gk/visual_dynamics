from __future__ import division

import os
import argparse
import numpy as np
import h5py
import cv2
from predictor import predictor
import simulator
import controller
import target_generator
import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_hdf5_fname', type=str)
    parser.add_argument('val_hdf5_fname', type=str, nargs='?', default=None)
    parser.add_argument('--predictor', '-p', type=str, default='small_action_cond_encoder_net')
    parser.add_argument('--pretrained_fname', '--pf', type=str, default=None)
    parser.add_argument('--solverstate_fname', '--sf', type=str, default=None)
    parser.add_argument('--train_batch_size', '--train_bs', type=int, default=32)
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--num_channel', type=int, help='net parameter')
    parser.add_argument('--y1_dim', type=int, help='net parameter')
    parser.add_argument('--y2_dim', type=int, help='net parameter')
    parser.add_argument('--constrained', type=int, default=1, help='net parameter')
    parser.add_argument('--levels', type=int, nargs='+', default=[3], help='net parameter')
    parser.add_argument('--x1_c_dim', '--x1cdim', type=int, default=16, help='net parameter')
    parser.add_argument('--num_downsample', '--numds', type=int, default=0, help='net parameter')
    parser.add_argument('--share_bilinear_weights', '--share', type=int, default=1, help='net parameter')
    parser.add_argument('--postfix', type=str, default=None)
    parser.add_argument('--output_hdf5_fname', '-o', type=str)
    parser.add_argument('--target_hdf5_fname', type=str, default=None)
    parser.add_argument('--train_target_hdf5_fnames', type=str, nargs=2, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--output_image_dir', type=str)
    parser.add_argument('--image_size', type=int, nargs=2, default=None, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--simulator', '-s', type=str, default=None, choices=('square', 'ogre', 'none'))
    # square simulator
    parser.add_argument('--abs_vel_max', type=float, default=None)
    parser.add_argument('--square_length', '-l', type=int, default=None, help='required to be odd')
    # ogre simulator
    parser.add_argument('--dof_min', type=float, nargs='+', default=None)
    parser.add_argument('--dof_max', type=float, nargs='+', default=None)
    parser.add_argument('--vel_min', type=float, nargs='+', default=None)
    parser.add_argument('--vel_max', type=float, nargs='+', default=None)
    parser.add_argument('--image_scale', '-f', type=float, default=None)
    parser.add_argument('--ogrehead', action='store_true')
    args = parser.parse_args()

    if args.val_hdf5_fname is None:
        args.val_hdf5_fname = args.train_hdf5_fname.replace('train', 'val')
    if args.postfix is None:
        args.postfix = os.path.basename(args.train_hdf5_fname).split('_')[0]

    # use sim args of the validation data
    with h5py.File(args.val_hdf5_fname, 'r') as hdf5_file:
        if 'sim_args' in hdf5_file:
            for arg_name, arg in hdf5_file['sim_args'].items():
                if args.__dict__[arg_name] is None:
                    args.__dict__[arg_name] = arg[...] if arg.shape else np.asscalar(arg[...])
    args.dof_min = np.asarray(args.dof_min)
    args.dof_max = np.asarray(args.dof_max)
    args.vel_min = np.asarray(args.vel_min)
    args.vel_max = np.asarray(args.vel_max)

    input_shapes = predictor.FeaturePredictor.infer_input_shapes(args.train_hdf5_fname)
    if args.predictor == 'bilinear':
        train_file = h5py.File(args.train_hdf5_fname, 'r+')
        X = train_file['image_curr'][:]
        U = train_file['vel'][:]
        feature_predictor = predictor.BilinearFeaturePredictor(X.shape[1:], U.shape[1:])
        X_dot = train_file['image_diff'][:]
        Y_dot = feature_predictor.feature_from_input(X_dot)
        if not args.no_train:
            feature_predictor.train(X, U, Y_dot)
    elif args.predictor.startswith('build_'):
        from predictor import predictor_theano, net_theano
        build_net = getattr(net_theano, args.predictor)
        feature_predictor = predictor_theano.TheanoNetFeaturePredictor(*build_net(input_shapes, levels=args.levels),
                                                                       postfix=args.postfix)
        if args.pretrained_fname is not None:
            feature_predictor.copy_from(args.pretrained_fname)
        if not args.no_train:
            feature_predictor.train(args.train_hdf5_fname, args.val_hdf5_fname,
                                    solver_type='ADAM',
                                    base_lr=0.001, gamma=0.99,
                                    momentum=0.9, momentum2=0.999,
                                    max_iter=args.max_iter)
    else:
        import caffe
        from caffe.proto import caffe_pb2 as pb2
        from predictor import predictor_caffe, net_caffe
        if args.pretrained_fname == 'auto':
            args.pretrained_fname = str(args.max_iter)
        if args.solverstate_fname == 'auto':
            args.solverstate_fname = str(args.max_iter)

        caffe.set_device(0)
        caffe.set_mode_gpu()
        if args.predictor == 'bilinear_net':
            feature_predictor = predictor_caffe.BilinearNetFeaturePredictor(hdf5_fname_hint=args.train_hdf5_fname,
                                                                            pretrained_file=args.pretrained_fname,
                                                                            postfix=args.postfix)
        else:
            net_kwargs = dict(num_channel=args.num_channel,
                              y1_dim=args.y1_dim,
                              y2_dim=args.y2_dim,
                              constrained=args.constrained,
                              levels=args.levels,
                              x1_c_dim=args.x1_c_dim,
                              num_downsample=args.num_downsample,
                              share_bilinear_weights=args.share_bilinear_weights)
            net_func = getattr(net_caffe, args.predictor)
            net_func_with_kwargs = lambda *args, **kwargs: net_func(*args, **dict(net_kwargs.items() + kwargs.items()))
            if args.predictor == 'fcn_action_cond_encoder_net':
                feature_predictor = predictor_caffe.FcnActionCondEncoderNetFeaturePredictor(net_func_with_kwargs,
                                                                                            input_shapes,
                                                                                            pretrained_file=args.pretrained_fname,
                                                                                            postfix=args.postfix)
            else:
                feature_predictor = predictor_caffe.CaffeNetFeaturePredictor(net_func_with_kwargs,
                                                                             input_shapes,
                                                                             pretrained_file=args.pretrained_fname,
                                                                             postfix=args.postfix)

        solver_param = pb2.SolverParameter(solver_type=pb2.SolverParameter.ADAM,
                                           base_lr=0.001, gamma=0.99,
                                           momentum=0.9, momentum2=0.999,
                                           max_iter=args.max_iter)
        if not args.no_train:
            feature_predictor.train(args.train_hdf5_fname,
                                    val_hdf5_fname=args.val_hdf5_fname,
                                    solverstate_fname=args.solverstate_fname,
                                    solver_param=solver_param,
                                    batch_size=args.train_batch_size)

            if feature_predictor.val_net is not None:
                val_losses = {blob_name: np.asscalar(blob.data) for blob_name, blob in feature_predictor.val_net.blobs.items() if blob_name.endswith('loss')}
                print 'val_losses', val_losses

    if args.simulator == 'square':
        sim = simulator.SquareSimulator(args.image_size, args.square_length, args.vel_max)
    elif args.simulator== 'ogre':
        sim = simulator.OgreSimulator([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                      image_scale=args.image_scale, crop_size=args.image_size,
                                      ogrehead=args.ogrehead)
    elif args.simulator == 'servo':
        sim = simulator.ServoPlatform([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                      image_scale=args.image_scale, crop_size=args.image_size)
    elif args.simulator == 'none':
        with h5py.File(args.val_hdf5_fname, 'r') as hdf5_file:
            for image_curr, vel, image_diff in zip(hdf5_file['image_curr'], hdf5_file['vel'], hdf5_file['image_diff']):
                image_next_pred = feature_predictor.predict(image_curr, vel, prediction_name='image_next_pred')
                if args.visualize:
                    image_next = image_curr + image_diff
                    image_curr = feature_predictor.preprocess_input(image_curr)
                    image_next = feature_predictor.preprocess_input(image_next)
                    image_pred_error = (image_next_pred - image_next)/2.0
                    vis_image, done = util.visualize_images_callback(image_curr, image_next_pred, image_next, image_pred_error, vis_scale=args.vis_scale, delay=0)
                    if done:
                        break
        return
    else:
        raise

    if args.target_hdf5_fname:
        target_gen = target_generator.Hdf5TargetGenerator(args.target_hdf5_fname)
        args.num_trajs = target_gen.num_images # override num_trajs to match the number of target images
    else:
        target_gen = target_generator.RandomTargetGenerator(sim)
    if args.train_target_hdf5_fnames:
        image_targets = [h5py.File(fname)['image_target'][()] for fname in args.train_target_hdf5_fnames]
        ctrl = controller.SpecializedServoingController(feature_predictor, *image_targets)
    else:
        ctrl = controller.ServoingController(feature_predictor)

    if args.output_hdf5_fname:
        output_hdf5_file = h5py.File(args.output_hdf5_fname, 'a')
        output_hdf5_group = output_hdf5_file.require_group(feature_predictor.net_name + '_' + feature_predictor.postfix)
        if feature_predictor.val_net is not None:
            val_losses_group = output_hdf5_group.require_group('val_losses')
            for key, value in val_losses.items():
                dset = val_losses_group.require_dataset(key, (1,), type(value), exact=True)
                dset[...] = value

    np.random.seed(0)
    done = False
    image_pred_errors = []
    image_errors = []
    pos_errors = []
    angle_errors = []
    iter_ = 0
    for traj_iter in range(args.num_trajs):
        try:
            image_target, dof_values_target = target_gen.get_target()
            ctrl.set_target_obs(image_target)

            dof_values_init = np.mean(sim.dof_limits, axis=0)
            sim.reset(dof_values_init)
            for step_iter in range(args.num_steps):
                image = sim.observe()
                action = ctrl.step(image)
                image_next_pred = feature_predictor.predict(image, action, prediction_name='image_next_pred')
                action = sim.apply_action(action)

                # visualization
                if args.visualize or args.output_image_dir:
                    vis_image, done = util.visualize_images_callback(feature_predictor.preprocess_input(image),
                                                                     image_next_pred,
                                                                     feature_predictor.preprocess_input(image_target),
                                                                     vis_scale=args.vis_scale, delay=100)
                    if args.output_image_dir:
                        if vis_image.ndim == 2:
                            output_image = np.concatenate([vis_image]*3, axis=2)
                        else:
                            output_image = vis_image
                        image_fname = feature_predictor.net_name + feature_predictor.postfix + '_%04d.png'%iter_
                        iter_ += 1
                        cv2.imwrite(os.path.join(args.output_image_dir, image_fname), output_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    if done:
                        break
            image = sim.observe()
            image_pred_error = np.linalg.norm(image_next_pred - feature_predictor.preprocess_input(image))
            image_pred_errors.append(image_pred_error)
            image_error = np.linalg.norm(feature_predictor.preprocess_input(image_target) - feature_predictor.preprocess_input(image))
            image_errors.append(image_error)
            dof_values = sim.state
            pos_error = np.linalg.norm(dof_values_target[:3] - dof_values[:3])
            pos_errors.append(pos_error)
            angle_error = dof_values_target[3:] - dof_values[3:]
            angle_errors.append(angle_error)
            print 'image_pred_error:', image_pred_error
            print 'image_error:', image_error
            print 'pos_error:', pos_error
            print 'angle_error:', angle_error, 'deg:', np.rad2deg(angle_error)
            if traj_iter == args.num_trajs-1:
                image_pred_error_mean = np.mean(image_pred_errors)
                image_error_mean = np.mean(image_errors)
                pos_error_mean = np.mean(pos_errors)
                angle_error_mean = np.mean(angle_errors, axis=0)
                print 'image_pred_error_mean:', image_pred_error_mean
                print 'image_error_mean:', image_error_mean
                print 'pos_error_mean:', pos_error_mean
                print 'angle_error_mean:', angle_error_mean, 'deg:', np.rad2deg(angle_error_mean)
            if args.output_hdf5_fname:
                for key, value in dict(image=image,
                                       image_pred_error=image_pred_error,
                                       image_target=image_target,
                                       image_error=image_error,
                                       state=dof_values,
                                       state_target=dof_values_target,
                                       pos_error=pos_error,
                                       angle_error=angle_error).items():
                    shape = (args.num_trajs, ) + value.shape
                    dset = output_hdf5_group.require_dataset(key, shape, value.dtype, exact=True)
                    dset[traj_iter] = value
                if traj_iter == args.num_trajs-1:
                    for key, value in dict(image_pred_error_mean=image_pred_error_mean,
                                           image_error_mean=image_error_mean,
                                           pos_error_mean=pos_error_mean,
                                           angle_error_mean=angle_error_mean).items():
                        if np.isscalar(value):
                            dset = output_hdf5_group.require_dataset(key, (1,), type(value), exact=True)
                        else:
                            dset = output_hdf5_group.require_dataset(key, value.shape, value.dtype, exact=True)
                        dset[...] = value

            if done:
                break
        except KeyboardInterrupt:
            break

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
