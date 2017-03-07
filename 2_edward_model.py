from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import os
import numpy as np
from odin import utils
import cPickle

# ===========================================================================
# Loading preprocessed data from pickle file
# Format of the data:
# "Sky1" -> [array_of_galaxies, halos_position]
# For exmple: to get all galaxies in "Sky1"
# >>> train['Sky1'][0]
# To get all halos in "Sky1"
# >>> train['Sky1'][1]
# ===========================================================================
train = cPickle.load(open("data/train.dat", 'rb'))
test = cPickle.load(open("data/test.dat", 'rb'))

from data import draw_sky
draw_sky(train['Sky1'][0])
plt.show(block=True)
