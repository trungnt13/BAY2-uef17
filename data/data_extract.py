# -*- coding: utf-8 -*-
# This code had been used for extracting and spliting the data
from __future__ import print_function, division, absolute_import

import os
import numpy as np
from six.moves import cPickle

TRAIN_SKIES_PATH = "/Users/trungnt13/tmp/Train_Skies/"
TRAIN_HALOS_PATH = "/Users/trungnt13/tmp/Train_Skies/halos.csv"


def get_all_files(path, filter_func=None):
    ''' Recurrsively get all files in the given path '''
    file_list = []
    if os.access(path, os.R_OK):
        for p in os.listdir(path):
            p = os.path.join(path, p)
            if os.path.isdir(p):
                file_list += get_all_files(p, filter_func)
            else:
                if filter_func is not None and not filter_func(p):
                    continue
                # remove dump files of Mac
                if '.DS_Store' in p or '.DS_STORE' in p or \
                    '._' == os.path.basename(p)[:2]:
                    continue
                file_list.append(p)
    return file_list

files = get_all_files(TRAIN_SKIES_PATH, lambda x: "_Sky" in x and ".csv" in x)
skies = {}
for f in files:
    name = os.path.basename(f).replace(".csv", "")
    name = name.split("_")[-1]
    f = np.genfromtxt(f, dtype=str, delimiter=",", skip_header=1)
    skies[name] = f[:, 1:].astype('float64')
halos = np.genfromtxt(TRAIN_HALOS_PATH, dtype=str, delimiter=',', skip_header=1)
halos = {X[0]: X[1:].astype('float64') for X in halos}

data = {n: (skies[n], h) for n, h in halos.iteritems()}.items()
np.random.shuffle(data); np.random.shuffle(data)
train = dict(data[:int(0.8 * len(data))])
test = dict(data[int(0.8 * len(data)):])
for _ in range(100):
    a, b = np.random.choice(range(len(data)), size=2, replace=False)
    a = data[a]
    b = data[b]
    if a[1][0].shape == b[1][0].shape:
        assert np.sum(a[1][0] - b[1][0]) != 0
cPickle.dump(train, open("train.dat", 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(test, open("test.dat", 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
