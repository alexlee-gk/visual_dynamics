from __future__ import division

import argparse
import numpy as np
import cv2
import data_container
import util
import simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_container_fnames', type=str, nargs='+')
    parser.add_argument('--traj_container', type=str, default='ImageTrajectoryDataContainer')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--image_scale', '-f', type=float, default=0.125)
    parser.add_argument('--crop_size', type=int, nargs=2, default=[32, 32], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--crop_offset', type=int, nargs=2, default=[0, 0], metavar=('HEIGHT_OFFSET', 'WIDTH_OFFSET'))
    args = parser.parse_args()

    TrajectoryDataContainer = getattr(data_container, args.traj_container)
    if not issubclass(TrajectoryDataContainer, data_container.TrajectoryDataContainer):
        raise ValueError('trajectory data container %s'%args.traj_data_container)
    num_trajs_total = 0
    num_steps_all = None # all traj containers should have same num_steps
    sim_args = None # sim_args of first traj container is used
    traj_containers = []
    for traj_container_fname in args.traj_container_fnames:
        traj_container = TrajectoryDataContainer(traj_container_fname)
        num_trajs = traj_container.num_trajs
        num_steps = traj_container.num_steps-1
        num_trajs_total += num_trajs
        if num_steps_all is None:
            num_steps_all = num_steps
        else:
            if num_steps_all != num_steps:
                raise ValueError('num_steps should be the same for all trajectories')
        if sim_args is None:
            sim_args = traj_container.get_group('sim_args')
        traj_containers.append(traj_container)

    image_transformer_args = dict(image_scale=args.image_scale,
                                  crop_size=args.crop_size,
                                  crop_offset=args.crop_offset)
    image_transformer = simulator.ImageTransformer(**image_transformer_args)

    assert np.allclose(-sim_args['vel_min'], sim_args['vel_max']) # assume vel limits are symmetric, otherwise also keep track of bias
    vel_scale = sim_args['vel_max']
    vel_inv_scale = 1. / vel_scale
    if args.output:
        output_traj_container = data_container.TrajectoryDataContainer(args.output, num_trajs_total, num_steps_all, write=True)
        sim_args['vel_scale'] = vel_scale
        output_traj_container.add_group('sim_args', sim_args)
        output_traj_container.add_group('image_transformer_args', image_transformer_args)
    else:
        output_traj_container = None

    done = False
    traj_iter_total = 0
    try:
        for traj_container in traj_containers:
            for traj_iter in range(traj_container.num_trajs):
                print 'traj_iter', traj_iter
                for step_iter in range(num_steps_all):
                    image, dof_val, vel = traj_container.get_datum(traj_iter, step_iter, ['image', 'dof_val', 'vel']).values()
                    image_next, = traj_container.get_datum(traj_iter, step_iter+1, ['image']).values()
                    # transform images
                    image = image_transformer.transform(image)
                    image_next = image_transformer.transform(image_next)
                    image_diff = image_next - image
                    if output_traj_container:
                        output_traj_container.add_datum(traj_iter_total + traj_iter, step_iter, dict(image_curr=image,
                                                                                                     image_diff=image_diff,
                                                                                                     dof_val=dof_val,
                                                                                                     vel=vel_inv_scale * vel))
                    if args.visualize:
                        vis_image, done = util.visualize_images_callback(image, image_next, image_diff/2, vis_scale=args.vis_scale, delay=0)
                    if done:
                        break
                if done:
                    break
            if done:
                break
            traj_iter_total += traj_container.num_trajs
    except KeyboardInterrupt:
        pass

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
