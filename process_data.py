from __future__ import division

import argparse
import cv2
import data_container
import util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traj_container_fname', type=str)
    parser.add_argument('--traj_container', type=str, default='ImageTrajectoryDataContainer')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    args = parser.parse_args()

    TrajectoryDataContainer = getattr(data_container, args.traj_container)
    if not issubclass(TrajectoryDataContainer, data_container.TrajectoryDataContainer):
        raise ValueError('trajectory data container %s'%args.traj_data_container)
    traj_container = TrajectoryDataContainer(args.traj_container_fname)
    num_trajs = traj_container.num_trajs
    num_steps = traj_container.num_steps-1
    sim_args = traj_container.get_group('sim_args')

    if args.output:
        output_traj_container = data_container.TrajectoryDataContainer(args.output, num_trajs, num_steps, write=True)
        output_traj_container.add_group('sim_args', sim_args)
    else:
        output_traj_container = None

    done = False
    for traj_iter in range(num_trajs):
        print 'traj_iter', traj_iter
        try:
            for step_iter in range(num_steps):
                image, dof_val, vel = traj_container.get_datum(traj_iter, step_iter, ['image', 'dof_val', 'vel']).values()
                image_next, = traj_container.get_datum(traj_iter, step_iter+1, ['image']).values()
                if output_traj_container:
                    output_traj_container.add_datum(traj_iter, step_iter, dict(image_curr=image,
                                                                               image_diff=image_next - image,
                                                                               dof_val=dof_val,
                                                                               vel=vel))
                if args.visualize:
                    vis_image, done = util.visualize_images_callback(image, vis_scale=args.vis_scale)
            if done:
                break
        except KeyboardInterrupt:
            break

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
