from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import os
import cPickle

import numpy as np

from data import draw_sky, load_data, freqcount

# ===========================================================================
# Loading preprocessed data from pickle file
# ===========================================================================
train, test = load_data()
# We only inspect the train data, do not touch the test data,
# test data is for evaluation
while True:
    _ = raw_input(
        "Please specify your choice:\n"
        "1) name of sky (e.g. Sky1)\n"
        "2) 'rand' for randomly visual 9 skies\n"
        "3) 'dist' distribution of the number of halos\n"
        "4) 'exit' to stop\n"
        "Your input:")
    if _.lower() == "exit":
        break
    elif _ == 'dist':
        plt.figure(figsize=(6, 4), dpi=180)
        n = freqcount([int(i[-1][0]) for i in train.itervalues()])
        x = range(1, len(n) + 1)
        plt.bar(x, [n[i] for i in x])
        plt.xticks(x)
        plt.show(block=True)
    elif _ not in train and _ != 'rand':
        print('Cannot find sky with name: "%s"' % _)
    else:
        plt.figure(figsize=(12, 12), dpi=120)
        if _ == 'rand': # multiple skies
            skies = np.random.choice(train.keys(), size=9, replace=False)
            for i, s in enumerate(skies):
                ax = plt.subplot(3, 3, i + 1)
                draw_sky(train[s][0], train[s][1], ax)
                plt.title(s)
            plt.tight_layout()
        else:
            draw_sky(train[_][0], train[_][1])
        plt.show(block=True)
