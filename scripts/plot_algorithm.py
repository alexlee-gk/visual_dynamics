import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from visual_dynamics.utils.iter_util import flatten_tree, unflatten_tree


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm_fnames', nargs='+', type=str)
    parser.add_argument('--progress_csv_paths', nargs='+', type=str)
    parser.add_argument('--usetex', '--use_tex', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    if args.usetex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

    title_fontsize = 18
    fontsize = 14

    if args.algorithm_fnames is None:
        args.algorithm_fnames = ['algorithm_learning/fqi_nooptfitbias_l2reg0.1_fc_pixel.yaml',
                                 'algorithm_learning/fqi_nooptfitbias_l2reg0.1_local_pixel.yaml',
                                 'algorithm_learning/fqi_nooptfitbias_l2reg0.1_local_level1.yaml',
                                 'algorithm_learning/fqi_nooptfitbias_l2reg0.1_local_level2.yaml',
                                 'algorithm_learning/fqi_nooptfitbias_l2reg0.1_local_level3.yaml',
                                 'algorithm_learning/fqi_nooptfitbias_l2reg0.1_local_level4.yaml',
                                 'algorithm_learning/fqi_nooptfitbias_l2reg0.1_local_level5.yaml']
    if args.progress_csv_paths is None:
        args.progress_csv_paths = [
            ['/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_20_20_05_35_0001/progress.csv',
             '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_27_08_11_23_0001/progress.csv'],
            '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_19_14_10_04_0001/progress.csv',
            '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_19_14_09_56_0001/progress.csv',
            ['/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_19_14_10_03_0001/progress.csv',
             '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_23_21_32_15_0001/progress.csv'],
            '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_19_14_10_17_0001/progress.csv',
            '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_20_14_31_29_0001/progress.csv',
            '/home/alex/rll/rllab/data/local/experiment/experiment_2017_03_20_20_05_03_0001/progress.csv']
    assert len(args.algorithm_fnames) == 7
    assert len(args.progress_csv_paths) == 7

    fqi_n_iters = 10
    fqi_mean_returns = []
    fqi_std_returns = []
    for algorithm_fname in args.algorithm_fnames:
        with open(algorithm_fname) as algorithm_file:
            algorithm_config = yaml.load(algorithm_file)
            fqi_mean_returns_ = -np.asarray(algorithm_config['mean_returns'][-(fqi_n_iters + 1):])
            fqi_std_returns_ = np.asarray(algorithm_config['std_returns'][-(fqi_n_iters + 1):])
            fqi_mean_returns.append(fqi_mean_returns_)
            fqi_std_returns.append(fqi_std_returns_)

    trpo_n_iters = 50
    trpo_iterations = []
    trpo_mean_returns = []
    trpo_std_returns = []
    for progress_csv_path in flatten_tree(args.progress_csv_paths):
        with open(progress_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            # divide by sqrt(10) to convert from standard deviation to standard error
            trpo_iterations_, trpo_mean_returns_, trpo_std_returns_ = \
                zip(*[(int(row['Iteration']), -float(row['AverageValReturn']), float(row['StdValReturn']) / np.sqrt(10)) for row in reader][:(trpo_n_iters + 1)])
            trpo_iterations.append(trpo_iterations_)
            trpo_mean_returns.append(np.array(trpo_mean_returns_))
            trpo_std_returns.append(np.array(trpo_std_returns_))
    trpo_iterations = unflatten_tree(args.progress_csv_paths, trpo_iterations)
    trpo_mean_returns = unflatten_tree(args.progress_csv_paths, trpo_mean_returns)
    trpo_std_returns = unflatten_tree(args.progress_csv_paths, trpo_std_returns)
    for i, (trpo_iterations_, trpo_mean_returns_, trpo_std_returns_) in enumerate(zip(trpo_iterations, trpo_mean_returns, trpo_std_returns)):
        trpo_iterations_flat = flatten_tree(trpo_iterations_, base_type=int)
        if trpo_iterations_flat != list(trpo_iterations_):
            assert trpo_iterations_flat == list(range(len(trpo_iterations_flat)))
            trpo_iterations[i] = tuple(trpo_iterations_flat[:(trpo_n_iters + 1)])
            trpo_mean_returns[i] = np.append(*trpo_mean_returns_)[:(trpo_n_iters + 1)]
            trpo_std_returns[i] = np.append(*trpo_std_returns_)[:(trpo_n_iters + 1)]

    color_palette = sns.color_palette('Set2', 10)
    colors = [color_palette[i] for i in [3, 5, 4, 6, 7, 9, 8]]
    labels = ['pixel, fully connected', 'pixel, locally connected', 'VGG conv1_2', 'VGG conv2_2', 'VGG conv3_3', 'VGG conv4_3', 'VGG conv5_3']
    if args.usetex:
        labels = [label.replace('_', '\hspace{-0.1em}\_\hspace{0.1em}') for label in labels]

    fig, (fqi_ax, trpo_ax) = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    # FQI
    fqi_batch_size = 10 * 100  # 10 trajectories, 100 time steps per trajectory
    fqi_num_samples = fqi_batch_size * np.arange(fqi_n_iters + 1)
    for i, (mean_return, std_return, label) in enumerate(zip(fqi_mean_returns, fqi_std_returns, labels)):
        fqi_ax.plot(fqi_num_samples, mean_return, label=label, color=colors[i])
        fqi_ax.fill_between(fqi_num_samples,
                            mean_return + std_return / 2.0,
                            mean_return - std_return / 2.0,
                            color=colors[i],
                            alpha=0.3)
    fqi_ax.set_xlabel('Number of Training Samples', fontsize=title_fontsize)
    fqi_ax.set_ylabel('Average Costs', fontsize=title_fontsize)
    fqi_ax.set_yscale('log')
    fqi_ax.set_xlim(0, fqi_num_samples[-1])
    fqi_ax.xaxis.set_tick_params(labelsize=fontsize)
    fqi_ax.yaxis.set_tick_params(labelsize=fontsize)
    # plt.axhline(y=min(map(min, fqi_mean_returns + trpo_mean_returns)), color='k', linestyle='--')

    fqi_ax.legend(bbox_to_anchor=(1.08, -0.1), loc='upper center', ncol=4, fontsize=fontsize)

    fqi_ax2 = fqi_ax.twiny()
    fqi_ax2.set_xlabel("FQI Sampling Iteration", fontsize=title_fontsize)
    fqi_ax2.set_xlim(fqi_ax.get_xlim())
    fqi_ax2.set_xticks(fqi_batch_size * np.arange(fqi_n_iters + 1))
    fqi_ax2.set_xticklabels(np.arange(fqi_n_iters + 1), fontsize=fontsize)

    # TRPO
    trpo_batch_size = 4000
    trpo_num_samples = trpo_batch_size * np.arange(trpo_n_iters + 1)
    for i, (mean_return, std_return, label) in enumerate(zip(trpo_mean_returns, trpo_std_returns, labels)):
        trpo_ax.plot(trpo_num_samples[:len(mean_return)], mean_return, label=label, color=colors[i])
        trpo_ax.fill_between(trpo_num_samples[:len(mean_return)],
                             mean_return + std_return / 2.0,
                             mean_return - std_return / 2.0,
                             color=colors[i],
                             alpha=0.3)
    trpo_ax.set_xlabel('Number of Training Samples', fontsize=title_fontsize)
    # trpo_ax.set_ylabel('Average Costs')
    trpo_ax.set_yscale('log')
    trpo_ax.set_xlim(0, trpo_num_samples[-1])
    trpo_ax.xaxis.set_tick_params(labelsize=fontsize)
    trpo_ax.yaxis.set_tick_params(labelsize=fontsize)
    trpo_ax.set_xticks(trpo_batch_size * np.arange(0, trpo_n_iters + 1, 10))
    trpo_ax.set_xticklabels(trpo_batch_size * np.arange(0, trpo_n_iters + 1, 10), fontsize=fontsize)
    # plt.axhline(y=min(map(min, fqi_mean_returns + trpo_mean_returns)), color='k', linestyle='--')

    trpo_ax2 = trpo_ax.twiny()
    trpo_ax2.set_xlabel("TRPO Sampling Iteration", fontsize=title_fontsize)
    trpo_ax2.set_xlim(trpo_ax.get_xlim())
    trpo_ax2.set_xticks(trpo_batch_size * np.arange(0, trpo_n_iters + 1, 5))
    trpo_ax2.set_xticklabels(np.arange(0, trpo_n_iters + 1, 5), fontsize=fontsize)

    ymins, ymaxs = zip(fqi_ax.get_ylim(), trpo_ax.get_ylim())
    ylim = (min(ymins), max(ymaxs))
    fqi_ax.set_ylim(ylim)
    trpo_ax.set_ylim(ylim)
    fqi_ax.grid(b=True, which='both', axis='y')
    trpo_ax.grid(b=True, which='both', axis='y')

    if args.save:
        plt.savefig('/home/alex/Dropbox/visual_servoing/20160322/fqi_trpo_learning_val_trajs.pdf', bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main()
