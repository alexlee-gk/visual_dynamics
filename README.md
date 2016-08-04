# visual_dynamics

## Use custom python, Theano and Lasagne installations

### Setting up a new python environment using pyenv

Install desired version of python 3 (e.g. 3.5.2). Make sure to use the `--enable-shared` flag to generate python shared libraries, which will later be linked to.
```
env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.5.2
```

### Install local version of Theano
```
git clone git://github.com/Theano/Theano.git
cd Theano
pyenv local 3.5.2
python setup.py develop
```

### Install local version of Lasagne
```
git clone https://github.com/Lasagne/Lasagne.git
cd Lasagne
pyenv local 3.5.2
pip install -r requirements.txt
pip install --editable .
```

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
