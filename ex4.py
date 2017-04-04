from __future__ import print_function, division, absolute_import

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from data import load_data
import edward as ed
from edward.models import (Categorical, InverseGamma, Mixture,
    MultivariateNormalDiag, Normal, Uniform)

# ===========================================================================
# Load and prepare data
# ===========================================================================
nb_components = 3
nb_datapoints = 300
nb_features = 2

train, test = load_data()
Galaxy_Pos = []
Galaxy_E = []
Halos_Pos = []
for name in sorted(train.keys(), key=lambda x: int(x.replace('Sky', ''))):
    (gal, hal) = train[name]
    Galaxy_Pos.append(gal[:nb_datapoints, :2])
    Galaxy_E.append(gal[:nb_datapoints, 2:])
    Halos_Pos.append(hal[3:3 + nb_components * 2].reshape(nb_components, 2))
print("Galaxy (X, Y):", len(Galaxy_Pos), Galaxy_Pos[0].shape)
print("Galaxy (E1, E2):", len(Galaxy_E), Galaxy_E[0].shape)
print("Halos (X, Y):", len(Halos_Pos), Halos_Pos[0].shape)

POS_STD = 0.2


# ===========================================================================
# Create the model
# ===========================================================================
def calculte_mean_from_distance_factor(gpos, hpos):
    # gpos: galaxies pos
    # hpos: one Halo pos
    # M: nb_datapoints
    # N: nb_components
    hpos = tf.reshape(hpos, (1, 2))
    euclidean_distance = tf.square(
        tf.subtract(
            gpos, # shape=(M, 2)
            tf.expand_dims(hpos, axis=1) # shape=(N, 1, 2)
        ) # shape=(N, M, 2)
    )
    distance_factor = tf.divide(1., euclidean_distance) # shape=(N, M, 2)
    mean = tf.reduce_sum(distance_factor, axis=(0,))  # shape=(M, 2)
    return mean

# (x, y) ~ Normal([0.5, 0.5], [0.5, 0.5])
galaxies_pos = Normal(
    mu=tf.fill([nb_datapoints, nb_features], 0.5),
    sigma=tf.fill([nb_datapoints, nb_features], POS_STD))

# latent variable z
mu = Normal(mu=tf.fill([nb_components, nb_features], 0.5),
            sigma=tf.fill([nb_components, nb_features], POS_STD))
sigma = InverseGamma(alpha=tf.ones([nb_components, nb_features]),
                     beta=tf.ones([nb_components, nb_features]))
cat = Categorical(logits=tf.zeros([nb_datapoints, nb_components]))
components = [
    MultivariateNormalDiag(mu=calculte_mean_from_distance_factor(galaxies_pos, mu[k]),
                           diag_stdev=tf.ones([nb_datapoints, 1]) * sigma[k])
    for k in range(nb_components)]
x = Mixture(cat=cat, components=components)

# ====== inference ====== #
qmu = Normal(
    mu=tf.Variable(tf.random_normal([nb_components, nb_features], mean=0., stddev=0.5)),
    sigma=tf.nn.softplus(tf.Variable(tf.zeros([nb_components, nb_features]))))
qsigma = InverseGamma(
    alpha=tf.nn.softplus(tf.Variable(tf.random_normal([nb_components, nb_features]))),
    beta=tf.nn.softplus(tf.Variable(tf.random_normal([nb_components, nb_features]))))

# fitting data
print("BUilding inference model ...")
# You must normalize Galaxy_Pos to (0, 1)
inference = ed.KLqp({mu: qmu, sigma: qsigma},
    data={x: Galaxy_E[0], galaxies_pos: Galaxy_Pos[0] / 4200})
inference.initialize(n_samples=20, n_iter=4000)
sess = ed.get_session()
tf.global_variables_initializer().run()

print("Start training ...")
for _ in range(inference.n_iter):
    info_dict = inference.update()
    inference.print_progress(info_dict)
    t = info_dict['t']
    if t % inference.n_print == 0:
        print("\nInferred cluster means:")
        print(sess.run(qmu.mean()))
