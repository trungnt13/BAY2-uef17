# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from data import get_value, draw_sky

import numpy as np

import tensorflow as tf
from edward.models import Normal, Uniform

M = 300 # number of galaxies
N = 3 # number of Dark Halos

# (x, y) ~ Uniform([0, 0], [4200, 4200])
galaxies_pos = Uniform(
    a=np.full(shape=(M, 2), fill_value=0., dtype='float32'),
    b=np.full(shape=(M, 2), fill_value=4200., dtype='float32'))

# (X, Y) ~ Uniform([0, 0], [4200, 4200])
halos_pos = Uniform(
    a=np.full(shape=(N, 2), fill_value=0., dtype='float32'),
    b=np.full(shape=(N, 2), fill_value=4200., dtype='float32'))

# mass of the dark halos
halos_mass = Uniform(
    a=np.full(shape=(N,), fill_value=40, dtype='float32'),
    b=np.full(shape=(N,), fill_value=180, dtype='float32')
)

# ====== calculate the distance from galaxies to halos ====== #
# tricky to calculate euclidean distance between 2 matrices but
# this broadcast trick would do the job.
euclidean_distance = tf.square(
    tf.subtract(
        galaxies_pos, # shape=(M, 2)
        tf.expand_dims(halos_pos, axis=1) # shape=(N, 1, 2)
    ) # shape=(N, M, 2)
)
distance_factor = tf.divide(1., euclidean_distance) # shape=(N, M, 2)
# multiply with the log of mass
mass_factor = tf.log(tf.reshape(halos_mass, (N, 1, 1))) # shape=(N, 1, 1)
mean = tf.reduce_sum(
    distance_factor * mass_factor, # shape=(N, M, 2)
    axis=(0,)
)  # shape=(M, 2)
# ====== ellipticity of the galaxies ====== #
# e ~ Normal(∑ 1 / distance * log(mass), sigma)
# I don't know what is the value of sigma, so why not give it
# an Uniform distribution, we give each galaxy a different sigma :D
sigma = Uniform(
    a=np.full(shape=(M, 2), fill_value=0.12, dtype='float32'),
    b=np.full(shape=(M, 2), fill_value=0.33, dtype='float32')
)
galaxies_elp = Normal(
    mu=mean,
    sigma=sigma,
)

# ====== happy sampling ====== #
galXY, halXY, halMAS, galE, sigma = get_value(
    [galaxies_pos.value(), halos_pos.value(),
     halos_mass.value(), galaxies_elp.value(), sigma.value()]
)
print("Galaxies position:", galXY.shape)
print("Galaxies ellipticity:", galE.shape)
print("Halos position:", halXY.shape)
print("Halos mass:", halMAS.shape)
print("Sigma:", sigma.shape)

# ====== visualize the generated sky ====== #
plt.figure(figsize=(8, 8), dpi=180)
draw_sky(galaxies=np.concatenate([galXY, galE], axis=-1),
         halos=[N] + halXY.ravel().tolist())
plt.show(block=True)
