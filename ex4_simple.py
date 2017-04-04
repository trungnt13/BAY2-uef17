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


# ===========================================================================
# Create the model
# ===========================================================================
# latent variable z
mu = Normal(mu=tf.zeros([nb_components, nb_features]),
            sigma=tf.ones([nb_components, nb_features]))
sigma = InverseGamma(alpha=tf.ones([nb_components, nb_features]),
                     beta=tf.ones([nb_components, nb_features]))
cat = Categorical(logits=tf.zeros([nb_datapoints, nb_components]))
components = [
    MultivariateNormalDiag(mu=tf.ones([nb_datapoints, 1]) * mu[k],
                           diag_stdev=tf.ones([nb_datapoints, 1]) * sigma[k])
    for k in range(nb_components)]
x = Mixture(cat=cat, components=components)

# ====== inference ====== #
qmu = Normal(
    mu=tf.Variable(tf.random_normal([nb_components, nb_features])),
    sigma=tf.nn.softplus(tf.Variable(tf.zeros([nb_components, nb_features]))))
qsigma = InverseGamma(
    alpha=tf.nn.softplus(tf.Variable(tf.random_normal([nb_components, nb_features]))),
    beta=tf.nn.softplus(tf.Variable(tf.random_normal([nb_components, nb_features]))))

# fitting data
print("BUilding inference model ...")
# You must normalize Galaxy_Pos to (0, 1)
X = Galaxy_Pos[0]
X = (X - X.mean()) / X.std()
inference = ed.KLqp({mu: qmu, sigma: qsigma}, data={x: X})
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
