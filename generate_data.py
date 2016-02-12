import argparse
import numpy as np
import cv2
import controller
import data_container
import util
import util_parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--traj_container', type=str, default='ImageTrajectoryDataContainer')
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=1, metavar='S', help='rescale image by S for visualization')
    subparsers = util_parser.add_simulator_subparsers(parser)
    parser_servo = subparsers.choices['servo']
    parser_servo.add_argument('--background_window', '-b', action='store_true')
    parser_servo.add_argument('--background_window_size', type=int, nargs=2, default=[5, 8], metavar=('HEIGHT', 'WIDTH'))
    args = parser.parse_args()

    sim = args.create_simulator(args)

    if args.output:
        TrajectoryDataContainer = getattr(data_container, args.traj_container)
        if not issubclass(TrajectoryDataContainer, data_container.TrajectoryDataContainer):
            raise ValueError('trajectory data container %s'%args.traj_container)
        traj_container = TrajectoryDataContainer(args.output, args.num_trajs, args.num_steps+1, write=True)
        sim_args = args.get_sim_args(args)
        traj_container.add_group('sim_args', sim_args)
    else:
        traj_container = None

    if args.simulator == 'servo' and args.background_window:
        cv2.namedWindow("Background window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Background window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(100)

    ctrl = controller.RandomController(*sim.action_bounds)
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            if args.simulator == 'servo' and args.background_window:
                background_shape = (np.random.randint(max(0, args.background_window_size[0]+1-3), args.background_window_size[0]+1),
                                    np.random.randint(max(0, args.background_window_size[0]+1-3), args.background_window_size[1]+1))
                cv2.imshow("Background window", (np.ones(background_shape)[..., None] * np.random.random(3)[None, None, :]))
                key = cv2.waitKey(100)
                key &= 255
                if key == 27 or key == ord('q'):
                    print("Pressed ESC or q, exiting")
                    done = True
                    break

            dof_val_init = sim.sample_state()
            sim.reset(dof_val_init)
            for step_iter in range(args.num_steps):
                dof_val = sim.dof_values
                image = sim.observe()
                action = ctrl.step(image)
                action = sim.apply_action(action)
                if traj_container:
                    traj_container.add_datum(traj_iter, step_iter, dict(image=image,
                                                                        dof_val=dof_val,
                                                                        vel=action))
                    if step_iter == (args.num_steps-1):
                        image_next = sim.observe()
                        traj_container.add_datum(traj_iter, step_iter+1, dict(image=image_next,
                                                                              dof_val=sim.dof_values))
                if args.visualize:
                    vis_image, done = util.visualize_images_callback(image, vis_scale=args.vis_scale, delay=0)
                if done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    sim.stop()
    if args.visualize or (args.simulator == 'servo' and args.background_window):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
