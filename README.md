# visual_dynamics

## Install dependencies from source and link to specific python installation

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
-DPYTHON3_NUMPY_INCLUDE_DIRS=/home/alex/.pyenv/versions/3.5.2/lib/python3.5/site-packages/numpy/core/include \ -DPYTHON3_PACKAGES_PATH=lib/python3.5/site-packages \
-DINSTALL_PYTHON_EXAMPLES=ON \
-DINSTALL_C_EXAMPLES=OFF \
-DBUILD_EXAMPLES=ON \
-DBUILD_opencv_python3=ON \
../opencv
make -j4
sudo make install
ln -s /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-x86_64-linux-gnu.so ~/.pyenv/versions/3.5.2/lib/python3.5/site-packages/cv2.so
```

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


## Optional dependencies

### Servos controlled through FT232H breakout board (for non-Jetson machines)

Install libftdi and its dependencies:
```
sudo apt-get update
sudo apt-get install build-essential libusb-1.0-0-dev swig cmake python-dev libconfuse-dev libboost-all-dev
wget http://www.intra2net.com/en/developer/libftdi/download/libftdi1-1.2.tar.bz2
tar xvf libftdi1-1.2.tar.bz2
cd libftdi1-1.2
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="/usr/" ../
make
sudo make install
```

Install Adafruit GPIO library:
```
cd /path/to/visual_dynamics
cd ext/adafruit/
sudo python setup.py install
```

In order to use the device with root access, put the following in the file `/etc/udev/rules.d/99-libftdi.rules`:
```
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", GROUP="dialout", MODE="0660"
```
Make sure to unplug and plug in the device for this rule to take effect. If root access is still required, the user might need to be added to the dialout group:
```
sudo usermod -a -G dialout $USER
```
Make sure to log out and log in for this change to take effect.
