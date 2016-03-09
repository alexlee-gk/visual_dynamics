import argparse
import cv2
import controller
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_config', type=str, help='config file with simulator arguments')
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=1, metavar='S', help='rescale image by S for visualization')
    args = parser.parse_args()

    with open(args.sim_config) as yaml_string:
        sim = utils.config.from_yaml(yaml_string)

    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['image', 'dof_val'], (args.num_trajs, args.num_steps+1))
        container.reserve('vel', (args.num_trajs, args.num_steps))
        container.add_info(simulator_config=sim.get_config())
    else:
        container = None

    ctrl = controller.RandomController(*sim.action_bounds)
    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            dof_val_init = sim.sample_state()
            sim.reset(dof_val_init)
            for step_iter in range(args.num_steps):
                dof_val = sim.dof_values
                image = sim.observe()
                action = ctrl.step(image)
                action = sim.apply_action(action)
                if container:
                    container.add_datum(traj_iter, step_iter, image=image, dof_val=dof_val, vel=action)
                    if step_iter == (args.num_steps-1):
                        image_next = sim.observe()
                        container.add_datum(traj_iter, step_iter+1, image=image_next, dof_val=sim.dof_values)
                if args.visualize:
                    vis_image, done = utils.visualization.visualize_images_callback(image, vis_scale=args.vis_scale, delay=0)
                if done:
                    break
            if done:
                break
        except KeyboardInterrupt:
            break
    sim.stop()
    if args.visualize:
        cv2.destroyAllWindows()
    if container:
        container.close()


if __name__ == "__main__":
    main()
