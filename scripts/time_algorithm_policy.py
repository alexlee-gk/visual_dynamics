import argparse

import yaml

from visual_dynamics.utils.config import Python2to3Loader, from_config
from visual_dynamics.utils.time_util import tic, toc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm_fname', type=str)
    args = parser.parse_args()

    with open(args.algorithm_fname) as algorithm_file:
        algorithm_config = yaml.load(algorithm_file, Loader=Python2to3Loader)

    alg = from_config(algorithm_config)
    env = alg.env
    servoing_pol = alg.servoing_pol

    obs = env.reset()
    servoing_pol.pi2([obs])

    T = 1000
    tic()
    for _ in range(T):
        pi = servoing_pol.pi2([obs])
    print(toc() / T)


if __name__ == '__main__':
    main()
