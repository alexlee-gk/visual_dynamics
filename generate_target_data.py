from __future__ import division

import argparse
import cv2
import target_generator
import data_container
import util
import util_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--target_generator', type=str, default='InteractiveTargetGenerator')
    parser.add_argument('--data_container', type=str, default='ImageDataContainer')
    parser.add_argument('--num_images', '-n', type=int, default=10)
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    util_parser.add_simulator_subparsers(parser)
    args = parser.parse_args()

    sim = args.create_simulator(args)

    if args.output:
        DataContainer = getattr(data_container, args.data_container)
        container = DataContainer(args.output, args.num_images)
        sim_args = args.get_sim_args(args)
        container.add_group('sim_args', sim_args)
    else:
        container = None

    if args.target_generator == 'InteractiveTargetGenerator':
        generator = target_generator.InteractiveTargetGenerator(sim, vis_scale=args.vis_scale)
    elif args.target_generator == 'RandomTargetGenerator':
        generator = target_generator.RandomTargetGenerator(sim)
    else:
        raise ValueError('generator %s'%args.target_generator)

    done = False
    for image_iter in range(args.num_images):
        print 'image_iter', image_iter
        try:
            image_target, dof_values_target = generator.get_target()
            if container:
                container.add_datum(image_iter, dict(image_target=image_target, pos=dof_values_target))
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
