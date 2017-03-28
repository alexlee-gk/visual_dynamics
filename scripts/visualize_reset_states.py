import argparse
import os

import cv2

from visual_dynamics.utils.config import from_config, from_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reset_states_fname', type=str)
    parser.add_argument('--output_dir', '-o', type=str)
    args = parser.parse_args()

    with open(args.reset_states_fname, 'r') as reset_state_file:
        reset_state_config = from_yaml(reset_state_file)
    env = from_config(reset_state_config['environment_config'])
    reset_states = reset_state_config['reset_states']

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    for traj_iter, reset_state in enumerate(reset_states):
        print(traj_iter)
        obs = env.reset(reset_state)
        image = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
        if args.output_dir:
            image_fname = os.path.join(args.output_dir, 'image_%03d.jpg' % traj_iter)
            cv2.imwrite(image_fname, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imshow("image", image)
        key = cv2.waitKey(0)
        key &= 255
        if key == 27 or key == ord('q'):
            print("Pressed ESC or q, exiting")
            break


if __name__ == '__main__':
    main()
