import argparse
import cv2
import data_container
import util_parser
import target_generator
from utils import util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--target_generator', type=str, default='InteractiveTargetGenerator')
    parser.add_argument('--traj_container', type=str, default='ImageTrajectoryDataContainer')
    parser.add_argument('--num_images', '-n', type=int, default=10)
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=1, metavar='S', help='rescale image by S for visualization')
    util_parser.add_simulator_subparsers(parser)
    args = parser.parse_args()

    sim = args.create_simulator(args)

    if args.output:
        TrajectoryDataContainer = getattr(data_container, args.traj_container)
        if not issubclass(TrajectoryDataContainer, data_container.TrajectoryDataContainer):
            raise ValueError('trajectory data container %s'%args.traj_container)
        traj_container = TrajectoryDataContainer(args.output, args.num_images, 1, write=True)
        sim_args = args.get_sim_args(args)
        traj_container.add_group('sim_args', sim_args)
    else:
        traj_container = None

    if args.target_generator == 'InteractiveTargetGenerator':
        generator = target_generator.InteractiveTargetGenerator(sim, args.num_images, vis_scale=args.vis_scale)
    elif args.target_generator == 'RandomTargetGenerator':
        generator = target_generator.RandomTargetGenerator(sim)
    else:
        raise ValueError('generator %s'%args.target_generator)

    done = False
    for image_iter in range(args.num_images):
        print('image_iter', image_iter)
        try:
            image_target, dof_values_target = generator.get_target()
            if traj_container:
                traj_container.add_datum(image_iter, 0, dict(image_target=image_target, pos=dof_values_target))
            if args.visualize:
                vis_image, done = util.visualize_images_callback(image_target, vis_scale=args.vis_scale)
            if done:
                break
        except KeyboardInterrupt:
            break
    sim.stop()
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
