from __future__ import print_function

import time


start_time = None


def tic():
    global start_time
    start_time = time.time()


def toc(name=None):
    duration = time.time() - start_time
    if name:
        print(name, duration)
    else:
        print(duration)
    return duration
