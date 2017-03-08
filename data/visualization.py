from __future__ import print_function, division, absolute_import

import os

import numpy as np


def draw_sky(galaxies, halos, ax=None):
    """adapted from Vishal Goklani
    Original code:
    https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter5_LossFunctions/draw_sky2.py
    Author:
        Max Margenot (https://github.com/mmargenot)
        Cameron Davidson-Pilon (https://github.com/CamDavidsonPilon)
    Modified work:
        TrungNT (https://github.com/trungnt13)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    size_multiplier = 45
    if ax is None:
        ax = plt.gca()
    ax.set_aspect('equal')
    n = galaxies.shape[0]
    for i in range(n):
        _g = galaxies[i, :]
        x, y = _g[0], _g[1]
        d = np.sqrt(_g[2]**2 + _g[3]**2)
        a = 1.0 / (1 - d)
        b = 1.0 / (1 + d)
        theta = np.degrees(np.arctan2(_g[3], _g[2]) * 0.5)
        ax.add_patch(
            Ellipse(xy=(x, y), width=size_multiplier * a, height=size_multiplier * b, angle=theta))
    ax.autoscale_view(tight=True)
    # ====== draw the halos ====== #
    for i in range(int(halos[0])):
        x = halos[1 + i * 2]
        y = halos[2 + i * 2]
        ax.scatter(x, y, c='black', alpha=0.5, s=60, linewidth=0)
    plt.xticks([], [])
    plt.yticks([], [])
    return ax
