from __future__ import print_function, division, absolute_import

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from data import load_data, draw_sky
import edward as ed
from edward.models import (Categorical, InverseGamma, Mixture,
    MultivariateNormalDiag, Normal, Uniform)

# ===========================================================================
# Load and prepare data
# ===========================================================================
nb_components = 3
nb_datapoints = 300

train, test = load_data()
Galaxy_Pos = []
Galaxy_E = []
Halos_Pos = []
for name in sorted(train.keys(), key=lambda x: int(x.replace('Sky', ''))):
    (gal, hal) = train[name]
    Galaxy_Pos.append(gal[:nb_datapoints, :2] / 4200)
    Galaxy_E.append(gal[:nb_datapoints, 2:])
    Halos_Pos.append(hal[3:3 + nb_components * 2] / 4200)
print("Galaxy (X, Y):", len(Galaxy_Pos), Galaxy_Pos[0].shape)
print("Galaxy (E1, E2):", len(Galaxy_E), Galaxy_E[0].shape)
print("Halos (X, Y):", len(Halos_Pos), Halos_Pos[0].shape)


# ===========================================================================
# Create the model
# ===========================================================================
galPos_placeholder = tf.placeholder(dtype='float32', shape=(nb_datapoints, 2),
                                    name='galPos')
galE_placeholder = tf.placeholder(dtype='float32', shape=(nb_datapoints, 2),
                                  name='galE')
# (X, Y) of each Halos
halPos_placeholder = tf.placeholder(dtype='float32', shape=(2 * nb_components,), name='halPos')


def neural_network(galPos, galE):
    """
    Input
    -----
    galPos:  (nb_datapoints, 2)
    galE:  (nb_components, 2)

    Output
    ------
    mu, sigma, logits"""
    # 2 hidden layers with 15 hidden units
    X = tf.concat(values=[galPos, galE], axis=1)
    hidden1 = slim.fully_connected(X, 15)

    hidden3 = tf.transpose(hidden1, (1, 0))
    # positioin of 3 halos
    hidden3 = slim.fully_connected(hidden3, 2 * nb_components)
    hidden3 = tf.transpose(hidden3, (1, 0))

    mus = slim.fully_connected(hidden3, nb_components, activation_fn=None)
    sigmas = slim.fully_connected(hidden3, nb_components, activation_fn=tf.exp)
    logits = slim.fully_connected(hidden3, nb_components, activation_fn=None)
    return mus, sigmas, logits

mus, sigmas, logits = neural_network(galPos_placeholder, galE_placeholder)

# ====== create the mixture ====== #
z = Categorical(logits=logits)
components = [Normal(mu=mu, sigma=sigma) for mu, sigma
              in zip(tf.unstack(tf.transpose(mus)),
                     tf.unstack(tf.transpose(sigmas)))]
y = Mixture(cat=z, components=components, value=tf.zeros(shape=(2 * nb_components,)))

# ====== perform inference ====== #
# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
inference = ed.MAP(data={y: halPos_placeholder})
inference.initialize(var_list=tf.trainable_variables())
sess = ed.get_session()
tf.global_variables_initializer().run()

# ====== fitting ====== #
n_epoch = 30
early_stop = [10e8]
for i in range(n_epoch):
    avg_loss = 0
    n = 0
    # shuffle the trianign data
    idx = np.random.permutation(len(Galaxy_Pos))
    galpos = [Galaxy_Pos[i] for i in idx]
    gale = [Galaxy_E[i] for i in idx]
    halpos = [Halos_Pos[i] for i in idx]
    for gPos, gE, hPos in zip(galpos, gale, halpos):
        info_dict = inference.update(feed_dict=
            {galPos_placeholder: gPos, galE_placeholder: gE,
             halPos_placeholder: hPos})
        avg_loss += info_dict['loss']
        n += 1
    print("Loss:", avg_loss / n)
    # early stopping
    if avg_loss / n > min(early_stop):
        break
    early_stop.append(avg_loss / n)

# ===========================================================================
# Visualize on test data
# ===========================================================================
gal, hal_true = test['Sky89']
hal_pred = sess.run(y, feed_dict={galPos_placeholder: gal[:nb_datapoints, :2] / 4200,
                                  galE_placeholder: gal[:nb_datapoints, 2:]})
print(hal_true[3:], hal_pred)

gal, hal_true = test['Sky164']
hal_pred = sess.run(y, feed_dict={galPos_placeholder: gal[:nb_datapoints, :2] / 4200,
                                  galE_placeholder: gal[:nb_datapoints, 2:]})
print(hal_true[3:], hal_pred)
