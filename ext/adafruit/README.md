Adafruit's Raspberry-Pi Python Code Library
============
  Here is a growing collection of libraries and example python scripts
  for controlling a variety of Adafruit electronics with a Raspberry Pi
  
  In progress!

  Adafruit invests time and resources providing this open source code,
  please support Adafruit and open-source hardware by purchasing
  products from Adafruit!

  Written by Limor Fried, Kevin Townsend and Mikey Sklar for Adafruit Industries.
  BSD license, all text above and below must be included in any redistribution
  
  To download, we suggest logging into your Pi with Internet accessibility and typing:
  
```bash
git clone https://github.com/adafruit/Adafruit-Raspberry-Pi-Python-Code.git
```

============
Copyright (c) 2012-2013 Limor Fried, Kevin Townsend and Mikey Sklar for Adafruit Industries.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Adafruit Python GPIO Library
============================

Library to provide a cross-platform GPIO interface on the Raspberry Pi and Beaglebone Black using the [RPi.GPIO](https://pypi.python.org/pypi/RPi.GPIO) and [Adafruit_BBIO](https://pypi.python.org/pypi/Adafruit_BBIO) libraries.

The library is currently in an early stage, but you can see how its used in the [Adafruit Nokia LCD library](https://github.com/adafruit/Adafruit_Nokia_LCD) to write Python code that is easily portable between the Raspberry Pi and Beaglebone Black.

Note that you typically don't need to install this library directly as other libraries will depend on it in their setup and automatically install it.  However if you do need to manually install do so by running these commands:

- On a Debian-based Linux like Raspbian, Ubuntu, etc. in a terminal execute:
  
  ```
  sudo apt-get update
  sudo apt-get install build-essential python-pip python-dev python-smbus git
  git clone https://github.com/adafruit/Adafruit_Python_GPIO.git
  cd Adafruit_Python_GPIO
  sudo python setup.py install
  ```

- On Mac OSX, first install PIP by [downloading the python script here](https://bootstrap.pypa.io/get-pip.py) and execute it with `python get-pip.py` in a terminal, then install the [git source control system](http://git-scm.com/downloads).  Then in a terminal execute:
  
  ```
  git clone https://github.com/adafruit/Adafruit_Python_GPIO.git
  cd Adafruit_Python_GPIO
  sudo python setup.py install
  ```

- On Windows, first install the [latest Python 2.7 version](https://www.python.org/downloads/windows/), then install PIP by [downloading the python script here](https://bootstrap.pypa.io/get-pip.py) and execute it with `python get-pip.py` in a terminal, and finally install the [git source control system](http://git-scm.com/downloads).  Then in a git bash prompt execute:
  
  ```
  git clone https://github.com/adafruit/Adafruit_Python_GPIO.git
  cd Adafruit_Python_GPIO
  python setup.py install
  ```

Contributing
------------

For information on contributing, such as how to run tests, etc. please see the [project wiki](https://github.com/adafruit/Adafruit_Python_GPIO/wiki/Running-Tests) on GitHub.
