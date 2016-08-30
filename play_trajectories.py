import argparse
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_fname', type=str, help='file name of data container')
    parser.add_argument('--use_actions', '-a', action='store_true')
    args = parser.parse_args()

    with utils.container.ImageDataContainer(args.data_fname) as container:
        environment_config = container.get_info('environment_config')
        env = utils.config.from_config(environment_config)

        num_trajs, num_steps = container.get_data_shape('action')
        assert container.get_data_shape('state') == (num_trajs, num_steps + 1)
        if args.use_actions:
            for traj_iter in range(num_trajs):
                state = container.get_datum(traj_iter, 0, 'state')
                env.reset(state)
                env.render()
                for step_iter in range(num_steps):
                    action = container.get_datum(traj_iter, step_iter, 'action')
                    env.step(action)
                    env.render()
        else:
            for traj_iter in range(num_trajs):
                for step_iter in range(num_steps + 1):
                    state = container.get_datum(traj_iter, step_iter, 'state')
                    env.reset(state)
                    env.render()
        env.close()


if __name__ == "__main__":
    main()
