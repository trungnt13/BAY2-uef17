# Observing Dark Worlds

There is more to the Universe than meets the eye. Out in the cosmos exists a form of matter that outnumbers the stuff we can see by almost 7 to 1, and we don't know what it is. What we do know is that it does not emit or absorb light, so we call it Dark Matter.

In fact we observe that this stuff aggregates and forms massive structures called Dark Matter Halos.

Because of enormous mass, it warps and bends spacetime such that any light from a background galaxy which passes close to the Dark Matter will have its path altered and changed. This bending causes the galaxy to appear as an ellipse in the sky.

Since there are many galaxies behind a Dark Matter halo, their shapes will correlate with its position.

Detecting these Dark Matter halos is hard, but possible using this data. If we can accurately estimate the positions of these halos, we can then understand the function they play in the Universe.

# Training Data

The training data consists of 300 simulated skies. Each sky contains between 300 and 740 galaxies. Each galaxy is represent by an ellipse on the sky, this ellipse is specified by _x_ and _y_ position ranging from _0_ to _4200_ (units are pixels), and a measure of ellipticity: _e1_ and _e2_.

Training galaxy data is provided in a series of 300 files, one file for each Sky (e.g, _Training_Sky27.csv_ or _Training_Sky123.csv_). These files have 4 columns:

* galaxy id
* x-coordinate
* y-coordinate
* e1
* e2

Dark matter halo locations in each sky are provided in the file Training_halos.csv. This file contains 10 columns, namely:

* Sky Id
* number of halos (1, 2 or 3)
* reference x-coordinate (used for evalulation metric)
* reference y-coordinate (used for evalulation metric)
* x-coordinate halo 1
* y-coordinate halo 1
* x-coordinate halo 2
* y-coordinate halo 2
* x-coordinate halo 3
* y-coordinate halo 3

In the case of only two or one halo present there will be zeros in the column.

# Test Data

The test data is in a similar format to the training data. There are 120 simulated skies (see final panel in Figure 2 of the description). Each sky contains 300 to 740 galaxies. Each galaxy will have an _x_ and _y_ position ranging from 0 to 4200, _e1_ and _e2_ values (totalling 4 columns per galaxy in the sky).

In each sky there are either 1, 2 or 3 dark matter halos. The halo counts in each sky are provided in the file _Test_haloCounts.csv_.

The challenge is to predict the center of each dark matter halo in each test sky based on the galaxy information provided.

# Notable knowledge

Each sky contain 1 to 3 Dark Matter Halos, probably 1 large Halos that governs most of the galaxies.

The ellipticity of each galaxy in the sky is dependent on:

* The position of the halos.
* The distance between the galaxy and halo.
* The mass of the halos.

The target variables inclues:

* The number of halos in each sky.
* _(x, y)_ for each halos.
