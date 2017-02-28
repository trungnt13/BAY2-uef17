# Install _miniconda_ for python package management

Download [miniconda](https://conda.io/miniconda.html) for **python 2.7**

From terminal:

```bash
chmod +x Miniconda2-latest-[MacOSX|Linux]-[x86|x86_64].sh
./Miniconda2-latest-[MacOSX|Linux]-[x86|x86_64].sh
conda update conda
```

# Create environment for probabilistic programming

Download the [text file (here)](https://gist.githubusercontent.com/trungnt13/042f3cafb545faad417b5694f5604a09/raw/099f20a31bf7842fa0c571674439312d4e532ba8/bay2-environment.yml), which contains all the necessary package.

First we create an environment with all specified packages, run following commands in terminal:

```bash
conda env create -f=[/path/to/environment.yml]
```

Activate the environment, override system default _python_:

```bash
source activate [yourenvname (bay2 by default)]
```

# Install _tensorflow_ for scientific computation

First you must activate the environment, then, _tensorflow_ can be installed via _pip_ using following command:

```bash
# Ubuntu/Linux 64-bit, CPU only, Python 2.7
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 8.0 and CuDNN v5. For other versions, see "Installing from sources" below.
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp27-none-linux_x86_64.whl

# Mac OS X, CPU only, Python 2.7:
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.0.0-py2-none-any.whl

# Mac OS X, GPU enabled, Python 2.7:
pip install --upgrade https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.0.0-py2-none-any.whl
```

It is suggested to install CPU version, since it doesn't require external libraries, and CPU is enough for our models (Note: significant speed up can be achieved with GPU version).

For Windows users, you have 2 options:

* Figure out instruction for [Windows here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
* Or install your own environment on _cs1.uef.fi_ server.

# Install _edward_, probabilistic programming library

Start from terminal, we clone the newest version of _edward_ from _github_:

```bash
git clone https://github.com/blei-lab/edward.git
```

The library will be download to _edward_ folder in the same path, now edit your **~/.bashrc** to add the library to your python path:

```bash
nano ~/.bashrc
# at the end of the file
export PYTHONPATH=$PYTHONPATH:/path/to/edward
```

Use $ctrl + O$ to save the file, and restart your bash.

Run a few tests to validate our installations:

```bash
source activate bay2
cd /path/to/edward/examples
python tf_bernoulli.py
```

And the output:

```bash
Iteration   1 [  1%]: Loss = -0.018
Iteration  10 [ 10%]: Loss = 0.033
Iteration  20 [ 20%]: Loss = 0.038
Iteration  30 [ 30%]: Loss = -0.000
Iteration  40 [ 40%]: Loss = -0.008
Iteration  50 [ 50%]: Loss = -0.001
Iteration  60 [ 60%]: Loss = -0.000
Iteration  70 [ 70%]: Loss = -0.001
Iteration  80 [ 80%]: Loss = -0.001
Iteration  90 [ 90%]: Loss = 0.001
Iteration 100 [100%]: Loss = -0.000
```