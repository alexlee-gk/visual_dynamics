import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', choices=['fqi', 'all'], default='fqi')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--infix', type=str, help='e.g. unseen')
    parser.add_argument('--usetex', '--use_tex', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.usetex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    if args.experiment_name != 'fqi':
        figsize = (9, 6)
    else:
        figsize = None
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
    bar_width = 0.35
    opacity = 0.8
    error_config = {'ecolor': '0.3'}

    title_fontsize = 18
    fontsize = 14

    if args.experiment_name == 'fqi':
        feature_dynamics_names = ['fc_pixel', 'local_pixel', 'local_level1', 'local_level2', 'local_level3', 'local_level4', 'local_level5']
        feature_dynamics_labels = ['pixel,\nfully\nconnected', 'pixel,\nlocally\nconnected', 'VGG\nconv1_2', 'VGG\nconv2_2', 'VGG\nconv3_3', 'VGG\nconv4_3', 'VGG\nconv5_3']
        if args.usetex:
            feature_dynamics_labels = [label.replace('_', '\hspace{-0.1em}\_\hspace{0.1em}') for label in feature_dynamics_labels]

        algorithm_name = 'fqi'
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

        index = np.arange(len(mean_returns)) * 2 * bar_width
        color_palette = sns.color_palette('Set2', 10)
        color = [color_palette[i] for i in [3, 5, 4, 6, 7, 9, 8]]
        plt.bar(index, mean_returns, bar_width,
                alpha=opacity,
                color=color,
                yerr=std_returns,
                error_kw=error_config)

        plt.xlabel('Feature Dynamics', fontsize=title_fontsize)
        plt.ylabel('Average Cost', fontsize=title_fontsize)
        if args.infix == 'unseen':
            plt.title('Costs of Executions when Following Novel Cars', fontsize=title_fontsize)
        else:
            plt.title('Costs of Executions when Following Cars Seen During Training', fontsize=title_fontsize)
        plt.xticks(index, feature_dynamics_labels, fontsize=fontsize)
        plt.yticks(np.arange(10), fontsize=fontsize)
        plt.legend()

        ax.set_ylim((0, 10))
    else:
        method_names = ['orb_nofilter', 'ccot', 'trpo_pixel',
                        'unweighted_local_level4',
                        'trpo_iter_2_local_level4', 'trpo_iter_50_local_level4', 'fqi_local_level4']
        method_labels = ['ORB\nfeature\npoints\nIBVS',
                         'C-COT\nvisual\ntracker\nIBVS',
                         'CNN\n+TRPO\n($\geq$ 20000)',
                         'unweighted\nfeature\ndynamics\n+CEM\n(1500)',
                         'feature\ndynamics\n+TRPO\n($\geq$ 80)',
                         'feature\ndynamics\n+TRPO\n($\geq$ 2000)',
                         r'$\textbf{ours,}$' '\n'
                         r'$\textbf{feature}$' '\n'
                         r'$\textbf{dynamics}$' '\n'
                         r'$\textbf{+FQI}$' '\n'
                         r'$\textbf{(20)}$']
        if args.usetex:
            method_labels = [label.replace('_', '\hspace{-0.1em}\_\hspace{0.1em}') for label in method_labels]

        mean_returns = []
        std_returns = []
        for method_name in method_names:
            conditions = [method_name]
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
        color_palette = sns.color_palette('Set2', 10)
        color = [color_palette[2]] * 3 + [color_palette[1]] * 5
        plt.bar(index, mean_returns, bar_width,
                alpha=opacity,
                color=color,
                yerr=std_returns,
                error_kw=error_config)

        plt.axvline(x=(index[2] + index[3]) / 2., color='k', linestyle='--')

        plt.text((index[1] + index[2]) / 2., 4.5, 'prior methods that\ndo not use learned\nfeature dynamics',
                 horizontalalignment='center', verticalalignment='center', fontsize=title_fontsize)
        text = 'methods that use VGG conv4_3\nfeatures and their learned\n  locally connected feature dynamics'
        if args.usetex:
            text = text.replace('_', '\hspace{-0.1em}\_\hspace{0.1em}')
        plt.text((index[4] + index[5]) / 2., 4.5, text, horizontalalignment='center', verticalalignment='center', fontsize=title_fontsize)

        plt.xlabel('Feature Representation and Optimization Method', fontsize=title_fontsize)
        plt.ylabel('Average Cost', fontsize=title_fontsize)
        # if args.infix == 'unseen':
        #     plt.title('Costs of Executions when Following Novel Cars', fontsize=title_fontsize)
        # else:
        #     plt.title('Costs of Executions when Following Cars Seen During Training', fontsize=title_fontsize)
        plt.xticks(index, method_labels, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend()

        ax.set_ylim((0, 5.5))

    if args.save:
        fname = args.experiment_name
        if args.infix:
            fname += '_' + args.infix
        fname += '_results'
        plt.savefig('/home/alex/Dropbox/visual_servoing/20160322/%s.pdf' % fname, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()
