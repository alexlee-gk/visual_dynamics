# visual_dynamics
Algorithms used in the paper <a href="https://arxiv.org/abs/1703.11000" target="_blank">Learning Visual Servoing with Deep Features and Fitted Q-Iteration</a>.

These are test executions of our policy based on VGG conv4_3 feature dynamics. The executions on the left use the cars seen during training and the ones on the right use novel cars. Our policy was trained with the fitted Q-iteration algorithm that we propose.
To see executions of other methods, check out the <a href="http://rll.berkeley.edu/visual_servoing/" target="_blank">paper's website</a>.

![Alt Text](http://rll.berkeley.edu/visual_dynamics/fqi_local_level4_test.gif)
![Alt Text](http://rll.berkeley.edu/visual_dynamics/fqi_local_level4_novel_test.gif)

## Installation instructions

### Install bleeding-edge version of Theano and apply patch
```
git clone git://github.com/Theano/Theano.git
cd Theano
git apply patches/theano_matrix_inverse.patch
python setup.py develop --prefix=~/.local
```

### Install bleeding-edge version of Lasagne and apply patch
```
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
git apply patches/lasagne_dilation.patch
pip install -r requirements.txt
pip install --editable . --user
```

### Install OpenCV
```
sudo apt-get install python-opencv
```

### Install CitySim3D and its dependencies
Follow the instructions from the [CitySim3D](https://github.com/alexlee-gk/citysim3d) site.

### Install visual_dynamics and its dependencies
```
git clone git@github.com:alexlee-gk/visual_dynamics.git
cd visual_dynamics
pip install -r requirements.txt
```

## Advanced installation instructions: Use pyenv and install dependencies from source

### Set up a new python environment using pyenv

Install desired version of python 3 (e.g. 3.5.2). Make sure to use the `--enable-shared` flag to generate python shared libraries, which will later be linked to.
```
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.5.2
```

### Install Theano
```
git clone git://github.com/Theano/Theano.git
cd Theano
pyenv local 3.5.2
python setup.py develop
```

### Install Lasagne
```
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
pyenv local 3.5.2
pip install -r requirements.txt
pip install --editable .
```

### Install OpenCV
Make sure python-dev is installed for the python version being used, e.g.
```
sudo apt-get install python3.5-dev
```
```
git clone git@github.com:opencv/opencv.git
mkdir opencv_build
cd opencv_build
pyenv local 3.5.2
cmake \
-DWITH_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DPYTHON3_EXECUTABLE=~/.pyenv/versions/3.5.2/bin/python3.5 \
-DPYTHON3_INCLUDE_DIR=~/.pyenv/versions/3.5.2/include/python3.5m \
-DPYTHON3_INCLUDE_DIR2=~/.pyenv/versions/3.5.2/include/python3.5m \
-DPYTHON_INCLUDE_DIRS=~/.pyenv/versions/3.5.2/include/ \
-DPYTHON3_LIBRARY=~/.pyenv/versions/3.5.2/lib/libpython3.so \
-DPYTHON3_NUMPY_INCLUDE_DIRS=~/.pyenv/versions/3.5.2/lib/python3.5/site-packages/numpy/core/include \
-DPYTHON3_PACKAGES_PATH=~/.pyenv/versions/3.5.2/lib/python3.5/site-packages \
-DINSTALL_PYTHON_EXAMPLES=ON \
-DINSTALL_C_EXAMPLES=OFF \
-DBUILD_EXAMPLES=ON \
-DBUILD_opencv_python3=ON \
../opencv
make -j4
sudo make install
ln -s /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so ~/.pyenv/versions/3.5.2/lib/python3.5/site-packages/cv2.so
```
For python 2, the `cmake` command is the following:
```
cmake \
-DWITH_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DPYTHON2_EXECUTABLE=~/.pyenv/versions/2.7.12/bin/python2.7 \
-DPYTHON2_INCLUDE_DIR=~/.pyenv/versions/2.7.12/include/python2.7 \
-DPYTHON2_INCLUDE_DIR2=~/.pyenv/versions/2.7.12/include/python2.7 \
-DPYTHON_INCLUDE_DIRS=~/.pyenv/versions/2.7.12/include/ \
-DPYTHON2_LIBRARY=~/.pyenv/versions/2.7.12/lib/libpython2.7.so \
-DPYTHON2_NUMPY_INCLUDE_DIRS=~/.pyenv/versions/2.7.12/lib/python2.7/site-packages/numpy/core/include \
-DPYTHON2_PACKAGES_PATH=~/.pyenv/versions/2.7.12/lib/python2.7/site-packages \
-DINSTALL_PYTHON_EXAMPLES=ON \
-DINSTALL_C_EXAMPLES=OFF \
-DBUILD_EXAMPLES=ON \
-DBUILD_opencv_python2=ON \
../opencv
```

The library can be installed only for the local user by specifying a local install prefix, e.g. `-DCMAKE_INSTALL_PREFIX=~/.local`, in which case `make install` should be run without root priviledges and the last symbolic linking step might not needed.

#### Common installation problems
-  After running `cmake`, the python2 OpenCV module appears next to 'Unavailable' instead of 'To be built'. Omit the flags that define `PYTHON2_EXECUTABLE` and `PYTHON2_LIBRARY` in the `cmake` command and then fix them with `ccmake` afterwards.
- The file `Python.h` is not found even though it is in the specified `PYTHON3_INCLUDE_DIR`, `fatal error: Python.h: No such file or directory`. Explicitly expanding the home  directory `~` to `${HOME}` might solve this.
- Installation for python 2 causes the compilation error `error: invalid conversion from ‘const char*’ to ‘Py_ssize_t {aka long int}’`. In this case, disable python 2 support with the option `-DBUILD_opencv_python2=OFF`.
- Importing `cv2` gives the error `ImportError: dynamic module does not define module export function (PyInit_cv2)` because it is using the wrong `cv2` library. Make sure the path for the newly built `cv2` package appears first in the `PYTHONPATH`, `export PYTHONPATH=~/.pyenv/versions/3.5.2/lib/python3.5/site-packages:$PYTHONPATH`.

In Mac OS X, replace the cmake command with this one:
```
cmake \
-DWITH_CUDA=OFF \
-DCMAKE_BUILD_TYPE=RELEASE \
-DPYTHON3_EXECUTABLE=~/.pyenv/versions/3.5.2/bin/python3.5 \
-DPYTHON3_INCLUDE_DIR=~/.pyenv/versions/3.5.2/include/python3.5m \
-DPYTHON3_INCLUDE_DIR2=~/.pyenv/versions/3.5.2/include/python3.5m \
-DPYTHON3_LIBRARY=~/.pyenv/versions/3.5.2/lib/libpython3m.dylib \
-DPYTHON3_LIBRARY_DEBUG=~/.pyenv/versions/3.5.2/lib/libpython3m.dylib \
-DPYTHON3_NUMPY_INCLUDE_DIRS=~/.pyenv/versions/3.5.2/lib/python3.5/site-packages/numpy/core/include \
-DPYTHON3_PACKAGES_PATH=~/.pyenv/versions/3.5.2/lib/python3.5/site-packages \
../opencv
```
The option `WITH_CUDA=OFF` might be necessary if Caffe is used. See [this issue](https://github.com/BVLC/caffe/issues/2256) for more information.

### Links
1. https://gist.github.com/pohmelie/cf4eda5df24303325b16
2. http://stackoverflow.com/questions/33250375/compiling-opencv3-with-pyenv-using-python-3-5-0-on-osx


## Example usage

### Generate training and validation data
```
mkdir -p data
python scripts/generate_data.py config/environment/simplequad.yaml config/policy/random_quad_back.yaml -n100 -t100 -o data/simplequad_train_data
python scripts/generate_data.py config/environment/simplequad.yaml config/policy/random_quad_back.yaml -n10 -t100 -o data/simplequad_val_data
```

### Train multiscale bilinear dynamics for a particular feature representation
```
python scripts/train.py config/predictor/multiscale_dilated_vgg_local_level1_scales012.yaml config/transformer/transformer_128.yaml config/solver/adam_gamma0.9_level1scales012.yaml config/data/simplequad.yaml
```

### Learn a weighting of the servoing features using fitted Q-iteration reinforcement learning
```
python scripts/learn_visual_servoing.py models/theano/multiscale_dilated_vgg_local_level1_scales012/transformer_128/adam_gamma0.9_level1scales012/simplequad/_iter_10000_model.yaml config/algorithm/fqi_nooptfitbias.yaml
```
