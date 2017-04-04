import os
# remove any visible GPU, force tensorflow to run only on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from multiprocessing import cpu_count
from six.moves import cPickle
from numbers import Number
from collections import defaultdict, OrderedDict

from .visualization import *

_path = os.path.dirname(os.path.realpath(__file__))


_SESSION = tf.Session(config=tf.ConfigProto(**{
    'intra_op_parallelism_threads': cpu_count() - 1,
    'allow_soft_placement': True,
    'log_device_placement': False,
}))


def get_value(x):
    """evaluate any tensorflow variable to get it real value"""
    if isinstance(x, (tuple, list)):
        return _SESSION.run(x)
    return x.eval(session=_SESSION)


def load_data():
    """
    Loading preprocessed data from pickle file
    Format of the data: "Sky1" -> [array_of_galaxies, halos_position]
    For exmple: to get all galaxies in "Sky1"
    >>> train['Sky1'][0]
    To get all halos in "Sky1"
    >>> train['Sky1'][1]

    Note
    ----
    Each Galaxy position contain: [x, y, e1, e2]
    Each Halos position contain: [nb_halo, refX, refY, x1, y1, x2, y2, x3, y3]

    Return
    ------
    train_data, test_data
    """
    train_path = os.path.join(_path, "train.dat")
    test_path = os.path.join(_path, "test.dat")
    if not os.path.exists(train_path):
        raise Exception("Cannot find train data at path:" + train_path)
    if not os.path.exists(test_path):
        raise Exception("Cannot find test data at path:" + test_path)
    train = cPickle.load(open(train_path, 'rb'))
    test = cPickle.load(open(test_path, 'rb'))
    return train, test


def freqcount(x, key=None, count=1, normalize=False, sort=False):
    """ x: list, iterable

    Parameters
    ----------
    key: callable
        extract the key from each item in the list
    count: callable, int
        extract the count from each item in the list
    normalize: bool
        if normalize, all the values are normalized from 0. to 1. (
        which sum up to 1. in total).
    sort: boolean
        if True, the list will be sorted in ascent order.

    Return
    ------
    dict: x(obj) -> freq(int)
    """
    freq = defaultdict(int)
    if key is None:
        key = lambda x: x
    if count is None:
        count = 1
    if isinstance(count, Number):
        _ = int(count)
        count = lambda x: _
    for i in x:
        c = count(i)
        i = key(i)
        freq[i] += c
    # always return the same order
    s = float(sum(v for v in freq.values()))
    freq = OrderedDict([(k, freq[k] / s if normalize else freq[k])
                        for k in sorted(freq.keys())])
    if sort:
        freq = OrderedDict(sorted(freq.items(), key=lambda x: x[1]))
    return freq
