import argparse
import yaml
import cv2
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--image_transformer_config', default='config/image_transformer_32.yaml')
    parser.add_argument('--vis_scale', '-s', type=int, default=1, metavar='S', help='rescale image by S for visualization')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--image_names', type=str, nargs='+', default=['image_curr', 'image_diff'])

    args = parser.parse_args()

    try:
        with open(args.image_transformer_config) as config_file:
            image_transformer = yaml.load(config_file)
    except FileNotFoundError:
        image_transformer = utils.transformer.Transformer()

    with utils.container.ImageDataContainer(args.data_dir) as container:
        num_trajs, num_steps = container.get_data_shape('image')
        for traj_iter in range(num_trajs):
            for step_iter in range(num_steps-1):
                image_curr = container.get_datum(traj_iter, step_iter, 'image')
                image_next = container.get_datum(traj_iter, step_iter+1, 'image')
                vis_image, done = utils.visualization.visualize_images_callback(
                    image_transformer.preprocess(image_curr),
                    image_transformer.preprocess(image_next),
                    vis_scale=args.vis_scale, delay=0)
                if done:
                    break
            if done:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
