import argparse
import cv2
import numpy as np
import controller
import target_generator
import utils


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('predictor_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-s', type=int, default=10, metavar='S', help='rescale image by S for visualization')

    parser.add_argument('--alpha', type=float, default=1.0, help='controller parameter')
    parser.add_argument('--lambda_', '--lambda', type=float, default=0.0, help='controller parameter')
    parser.add_argument('--dof_limit_factor', type=float, default=1.0, help='experiment parameter')
    args = parser.parse_args()

    with open(args.predictor_fname) as predictor_file:
        predictor = utils.from_yaml(predictor_file)

    simulator_config = predictor.simulator_config
    sim = utils.config.from_config(simulator_config)

    ctrl = controller.ServoingController(predictor, alpha=args.alpha, lambda_=args.lambda_)
    target_gen = target_generator.RandomTargetGenerator(sim, args.num_trajs)

    if args.output_dir:
        container = utils.container.ImageDataContainer(args.output_dir, 'x')
        container.reserve(['image', 'dof_val', 'image_x0'], (args.num_trajs, args.num_steps + 1))
        container.reserve(['vel', 'image_x0_next_pred'], (args.num_trajs, args.num_steps))
        container.reserve(['image_target', 'dof_val_target', 'image_x0_target'], args.num_trajs)
        container.add_info(simulator_config=sim.get_config())
        container.add_info(predictor_config=predictor.get_config())
    else:
        container = None

    done = False
    for traj_iter in range(args.num_trajs):
        print('traj_iter', traj_iter)
        try:
            # generate target image
            image_target, dof_val_target = target_gen.get_target()
            if container:
                image_x0_target = predictor.transformers[0].transformers[-1].deprocess(predictor.preprocess(image_target)[0])
                container.add_datum(traj_iter, image_target=image_target, dof_val_target=dof_val_target,
                                    image_x0_target=image_x0_target)
            ctrl.set_target_obs(image_target)
            image_target, = predictor.preprocess(image_target)

            # generate initial state
            sim.reset(dof_val_target)
            reset_action = args.dof_limit_factor * (sim.dof_vel_limits[0] + np.random.random_sample(sim.dof_vel_limits[0].shape) * (sim.dof_vel_limits[1] - sim.dof_vel_limits[0]))
            dof_val_init = sim.dof_values + reset_action
            sim.reset(dof_val_init)
            for step_iter in range(args.num_steps):
                dof_val = sim.dof_values
                image = sim.observe()
                action = ctrl.step(image)
                action = sim.apply_action(action)
                if container:
                    image_x0 = predictor.transformers[0].transformers[-1].deprocess(predictor.preprocess(image)[0])
                    image_next_pred = predictor.predict('x0_next_pred', image, action)
                    np.clip(image_next_pred, -1.0, 1.0, out=image_next_pred)
                    image_x0_next_pred = predictor.transformers[0].transformers[-1].deprocess(image_next_pred)
                    container.add_datum(traj_iter, step_iter, image=image, dof_val=dof_val, vel=action,
                                        image_x0=image_x0, image_x0_next_pred=image_x0_next_pred)
                    if step_iter == (args.num_steps - 1):
                        image_next = sim.observe()
                        image_x0_next = predictor.transformers[0].transformers[-1].deprocess(predictor.preprocess(image)[0])
                        container.add_datum(traj_iter, step_iter + 1, image=image_next, dof_val=sim.dof_values,
                                            image_x0=image_x0_next)
                if args.visualize == 1:
                    image_next_pred = predictor.predict('x0_next_pred', image, action)
                    vis_image, done = utils.visualization.visualize_images_callback(*predictor.preprocess(image),
                                                                                    image_next_pred,
                                                                                    image_target,
                                                                                    image_transformer=predictor.transformers[0].transformers[-1],
                                                                                    vis_scale=args.vis_scale,
                                                                                    delay=100)
                elif args.visualize > 1:
                    predictor.plot(image, action, sim.observe())
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
