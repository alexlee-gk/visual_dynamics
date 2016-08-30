#!/usr/bin/env python

import numpy as np
import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from ext.adafruit.Adafruit_PWM_Servo_Driver.Adafruit_PWM_Servo_Driver import PWM

class PwmListener(object):
    def __init__(self, pwm_address=0x40, pwm_freq=60):
        self.pwm = PWM(pwm_address)
        self.pwm.setPWMFreq(pwm_freq)
        rospy.Subscriber("/pwm_channels_values", numpy_msg(Floats), self.callback)

    def callback(self, data):
        channels_values = data.data.astype(np.int)
        num_channels = len(channels_values) // 2
        print "==="
        for channel, dof_value in zip(channels_values[:num_channels], channels_values[num_channels:]):
            print channel, dof_value
            self.pwm.setPWM(channel, 0, dof_value)


if __name__ == '__main__':
    rospy.init_node('pwm_node')
    pwm_listener = PwmListener()
    rospy.spin()
