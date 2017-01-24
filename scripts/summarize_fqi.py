from __future__ import division, print_function

import csv

import argparse
import numpy as np
import yaml

from visual_dynamics import policies
from visual_dynamics.utils.rl_util import do_rollouts, discount_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm_fnames', nargs='+', type=str)
    args = parser.parse_args()

    for algorithm_fname in args.algorithm_fnames:
        with open(algorithm_fname) as algorithm_file:
            algorithm_config = yaml.load(algorithm_file)
        mean_discounted_returns = algorithm_config['mean_discounted_returns']
        row_values = [algorithm_fname, max(mean_discounted_returns)] + mean_discounted_returns
        print('\t'.join([str(value) for value in row_values]))

    return

    header_format = '{:>15}\t{:>15}\t{:>15}\t{:>15}'
    row_format = '{:>15.2f}\t{:>15.2f}\t{:>15.4f}\t{:>15.4f}'
    print(header_format.format('w_init', 'lambda_init', 'mean', 'standard error'))
    print('=' * 60)

    if args.output_fname:
        with open(args.output_fname, 'ab') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['w_init', 'lambda_init', 'mean', 'standard error'])

    for w_init in args.w_inits:
        for lambda_init in args.lambda_inits:
            pol = policies.ServoingPolicy(predictor, alpha=1.0, lambda_=lambda_init, w=w_init)
            _, _, _, rewards = do_rollouts(env, pol, args.num_trajs, args.num_steps,
                                                 target_distance=args.target_distance,
                                                 image_visualizer=image_visualizer,
                                                 gamma=args.gamma,
                                                 seeds=np.arange(args.num_trajs))
            discounted_returns = discount_returns(rewards, args.gamma)
            row_values = [w_init, lambda_init, np.mean(discounted_returns), np.std(discounted_returns) / np.sqrt(len(discounted_returns))]
            print(row_format.format(*row_values))

            if args.output_fname:
                with open(args.output_fname, 'ab') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([str(value) for value in (row_values + list(discounted_returns))])


if __name__ == "__main__":
    main()
