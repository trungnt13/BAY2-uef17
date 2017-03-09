from __future__ import print_function, division, absolute_import

from data import get_value
import tensorflow as tf
import edward as ed
from edward.models import Beta, Bernoulli


theta = Beta(a=1.0, b=1.0)
# 100-dimensional Bernoulli
x = Bernoulli(p=tf.ones(12) * theta)

# ====== sampling from each marginal variables
theta_sample = theta.sample()
x_sample = x.sample()
print("Marginal theta samples:", get_value(theta_sample))
print("Marginal X samples:", get_value(x_sample))

# ====== sampling from the joint distribution
samples = get_value([x.value(), theta.value()])
print("From joint distribution:")
print("- X:", samples[0])
print("- theta:", samples[1])
