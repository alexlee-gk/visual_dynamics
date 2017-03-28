import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', choices=['main', 'trpo'], default='main')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--infix', type=str, help='e.g. unseen')
    parser.add_argument('--usetex', '--use_tex', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.usetex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    fig, ax = plt.subplots(tight_layout=True)
    bar_width = 0.35
    opacity = 0.8
    error_config = {'ecolor': '0.3'}

    if args.experiment_name == 'main':
        feature_dynamics_names = ['fc_pixel', 'local_pixel', 'local_level1', 'local_level2', 'local_level3', 'local_level4', 'local_level5']
        feature_dynamics_labels = ['pixel,\nfully\nconnected', 'pixel,\nlocally\nconnected', 'VGG\nconv1_2', 'VGG\nconv2_2', 'VGG\nconv3_3', 'VGG\nconv4_3', 'VGG\nconv5_3']
        if args.usetex:
            feature_dynamics_labels = [label.replace('_', '\hspace{-0.1em}\_\hspace{0.1em}') for label in feature_dynamics_labels]
        algorithm_names = ['unoptimized', 'unweighted', 'cem', 'trpo_iter_2', 'trpo', 'fqi']
        algorithm_labels = ['Unweighted (0)', 'Unweighted Features (1500)', 'CEM (3250)', 'TRPO ($\geq$ 80)', 'TRPO ($\geq$ 800)', 'FQI, ours (20)']

        for i, algorithm_name in enumerate(algorithm_names):
            mean_returns = []
            std_returns = []
            for feature_dynamics_name in feature_dynamics_names:
                conditions = [algorithm_name, feature_dynamics_name]
                if args.infix:
                    conditions.append(args.infix)
                result_fname = os.path.join(args.results_dir, '_'.join(conditions + ['test.csv']))
                if not os.path.exists(result_fname):
                    break
                with open(result_fname, 'r') as result_file:
                    reader = csv.reader(result_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    mean_return, std_return = list(reader)[-1]
                    mean_returns.append(-float(mean_return))
                    std_returns.append(float(std_return))

            index = np.arange(len(mean_returns)) * (len(algorithm_names) + 2) * bar_width
            plt.bar(index + i * bar_width, mean_returns, bar_width,
                    alpha=opacity,
                    color='C%d' % i,
                    yerr=std_returns,
                    error_kw=error_config,
                    label=algorithm_labels[i])

        plt.xlabel('Feature Dynamics')
        plt.ylabel('Average Cost')
        if args.infix == 'unseen':
            plt.title('Costs when following novel cars')
        else:
            plt.title('Costs when following cars seen during training')
        plt.xticks(index + bar_width * (len(algorithm_names) - 1) / 2, feature_dynamics_labels)
        plt.legend()

        ax.set_ylim((0, 14))
    else:
        observation_type_names = ['trpo_pos', 'trpo_pixel', 'trpo_level1', 'trpo_level2', 'trpo_level3']
        observation_type_labels = ['ground truth\ncar position', 'raw pixel-intensity\nimages', 'VGG conv1_2\nfeatures', 'VGG conv2_2\nfeatures', 'VGG conv3_3\nfeatures']
        if args.usetex:
            observation_type_labels = [label.replace('_', '\hspace{-0.1em}\_\hspace{0.1em}') for label in observation_type_labels]

        mean_returns = []
        std_returns = []
        for observation_type_name in observation_type_names:
            conditions = [observation_type_name]
            if args.infix:
                conditions.append(args.infix)
            result_fname = os.path.join(args.results_dir, '_'.join(conditions + ['test.csv']))
            if not os.path.exists(result_fname):
                break
            with open(result_fname, 'r') as result_file:
                reader = csv.reader(result_file, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                mean_return, std_return = list(reader)[-1]
                mean_returns.append(-float(mean_return))
                std_returns.append(float(std_return))

        index = np.arange(len(mean_returns)) * 2 * bar_width
        plt.bar(index, mean_returns, bar_width,
                alpha=opacity,
                color='C0',
                yerr=std_returns,
                error_kw=error_config)

        plt.axvline(x=bar_width, color='k', linestyle='--')
        plt.text(index[0], 12, 'ground truth\nobservation', horizontalalignment='center', verticalalignment='center')
        plt.text(index[1], 12, 'image-based\nobservations', horizontalalignment='center', verticalalignment='center')

        plt.xlabel('Observation Type')
        plt.ylabel('Average Cost')
        if args.infix == 'unseen':
            plt.title('Costs when following novel cars')
        else:
            plt.title('Costs when following cars seen during training')
        plt.xticks(index, observation_type_labels)
        plt.legend()

        ax.set_ylim((0, 15))

    if args.save:
        plt.show(block=False)
        fname = args.experiment_name
        if args.infix:
            fname += '_' + args.infix
        fname += '_results'
        plt.savefig('/home/alex/Dropbox/visual_servoing/20160322/%s.pdf' % fname, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()
