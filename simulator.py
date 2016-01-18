from __future__ import division

import time
import numpy as np
import threading
import cv2

def axis2quat(axis, angle):
    axis = np.asarray(axis)
    axis = 1.0*axis/axis.sum();
    return np.append(np.cos(angle/2.0), axis*np.sin(angle/2.0))

def quaternion_multiply(*qs):
    if len(qs) == 2:
        q0, q1 = qs
        return np.array([-q1[1]*q0[1] - q1[2]*q0[2] - q1[3]*q0[3] + q1[0]*q0[0],
                          q1[1]*q0[0] + q1[2]*q0[3] - q1[3]*q0[2] + q1[0]*q0[1],
                         -q1[1]*q0[3] + q1[2]*q0[0] + q1[3]*q0[1] + q1[0]*q0[2],
                          q1[1]*q0[2] - q1[2]*q0[1] + q1[3]*q0[0] + q1[0]*q0[3]])
    else:
        return quaternion_multiply(qs[0], quaternion_multiply(*qs[1:]))

class Simulator(object):
    def apply_action(self, action):
        raise NotImplementedError

    def observe(self):
        raise NotImplementedError

    def reset(self, state):
        raise NotImplementedError

    @property
    def action_bounds(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError

    def sample_state(self):
        raise NotImplementedError

    @property
    def action_dim(self):
        raise NotImplementedError

    @property
    def state_dim(self):
        raise NotImplementedError


class SquareSimulator(object):
    def __init__(self, image_size, square_length, vel_max, pos_init=None):
        assert len(image_size) == 2
        assert square_length%2 != 0
        self._image = np.zeros(image_size, dtype=np.uint8)
        self.square_length = square_length
        self.pos_min = np.asarray([square_length//2]*2)
        self.pos_max = np.asarray([l - 1 - square_length//2 for l in image_size])
        self._pos = None
        if pos_init is None:
            self.pos = (self.pos_min + self.pos_max)/2
        else:
            self.pos = pos_init
        self.vel_max = vel_max

    @property
    def pos(self):
        return self._pos.copy()

    @pos.setter
    def pos(self, next_pos):
        self._image *= 0
        self._pos = np.clip(next_pos, self.pos_min, self.pos_max)
        ij0 = self._pos - self.square_length//2
        ij1 = self._pos + self.square_length//2 + 1
        self._image[ij0[0]:ij1[0], ij0[1]:ij1[1]] = 255

    def apply_action(self, vel):
        pos_prev = self.pos.copy()
        self.pos += vel
        vel = self.pos - pos_prev # recompute vel because of clipping
        return vel

    def observe(self):
        import cv2
        return (cv2.GaussianBlur(self._image, (5,5), -1).astype(float)[None, :] / 255.0) * 2.0 - 1.0

    def reset(self, pos):
        self.pos = pos

    @property
    def action_bounds(self):
        action_min = -self.vel_max * np.ones(self.action_dim)
        action_max = self.vel_max * np.ones(self.action_dim)
        return action_min, action_max

    @property
    def state(self):
        return self.pos

    def sample_state(self):
        pos = self.pos_min + np.random.random_sample(self.pos_min.shape) * (self.pos_max - self.pos_min)
        return pos

    @property
    def action_dim(self):
        dim, = self._pos.shape
        return dim

    @property
    def state_dim(self):
        dim, = self._pos.shape
        return dim


class DiscreteVelocitySimulator(Simulator):
    def __init__(self, dof_limits, dof_vel_limits, dtype=None):
        dof_min, dof_max = dof_limits
        vel_min, vel_max = dof_vel_limits
        assert len(dof_min) == len(dof_max)
        assert len(vel_min) == len(vel_max)
        self.dof_limits = [np.asarray(dof_limit, dtype=dtype) for dof_limit in dof_limits]
        self.dof_vel_limits = [np.asarray(dof_vel_limit, dtype=dtype) for dof_vel_limit in dof_vel_limits]
        self._dof_values = np.mean(self.dof_limits, axis=0)

    @property
    def dof_values(self):
        return self._dof_values.copy()

    @dof_values.setter
    def dof_values(self, next_dof_values):
        assert self._dof_values is None or self._dof_values.shape == next_dof_values.shape
        self._dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])

    def apply_action(self, vel):
        dof_values_prev = self.dof_values.copy()
        self.dof_values += vel
        vel = self.dof_values - dof_values_prev # recompute vel because of clipping
        return vel

    def reset(self, dof_values):
        self.dof_values = dof_values

    @property
    def action_bounds(self):
        return self.dof_vel_limits

    @property
    def state(self):
        return self.dof_values

    def sample_state(self):
        dof_values = self.dof_limits[0] + np.random.random_sample(self.dof_limits[0].shape) * (self.dof_limits[1] - self.dof_limits[0])
        return dof_values

    @property
    def action_dim(self):
        dim, = self._dof_values.shape
        return dim

    @property
    def state_dim(self):
        dim, = self._dof_values.shape
        return dim


class ScaleCropImageSimulator(Simulator):
    def __init__(self, image_scale=None, crop_size=None):
        self.image_scale = image_scale
        self.crop_size = crop_size

    def _scale_crop(self, image):
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = (image.astype(float) / 255.0) * 2.0 - 1.0
        if self.image_scale is not None:
            image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)
        if self.crop_size is not None:
            h, w = image.shape[:2]
            crop_h, crop_w = self.crop_size
            image = image[h/2-crop_h/2:h/2-crop_h/2+crop_h, w/2-crop_w/2:w/2-crop_w/2+crop_w, ...]
        if image.ndim == 2:
            image = image[None, :, :]
        else:
            image = image.transpose(2, 0, 1)
        return image


class OgreSimulator(DiscreteVelocitySimulator, ScaleCropImageSimulator):
    def __init__(self, dof_limits, dof_vel_limits, image_scale=None, crop_size=None, ogrehead=False):
        """
        DOFs are x, y, z, angle_x, angle_y, angle_z
        """
        DiscreteVelocitySimulator.__init__(self, dof_limits, dof_vel_limits)
        ScaleCropImageSimulator.__init__(self, image_scale=image_scale, crop_size=crop_size)
        self._q0 = axis2quat(np.array([0, 1, 0]), np.pi/2)

        import pygre
        self.ogre = pygre.Pygre()
        self.ogre.init()
        self.ogre.addNode("node1", "house.mesh", 0, 0, 0)
        if ogrehead:
            self.ogre.addNode("node2", "ogrehead.mesh", 10, 5, -5) #([far, close], [down, up], [right, left])
            self.ogre.addNode("node3", "ogrehead.mesh", 10, 0, -5)
            self.ogre.addNode("node4", "ogrehead.mesh", 10, 5, -10)
            self.ogre.addNode("node5", "ogrehead.mesh", 10, 0, -10)
        self.ogre.setCameraOrientation(self._q0)

    @DiscreteVelocitySimulator.dof_values.setter
    def dof_values(self, next_dof_values):
        assert self._dof_values is None or self._dof_values.shape == next_dof_values.shape
        self._dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])
        pos_angle = np.zeros(6)
        pos_angle[:min(6, self.state_dim)] += self._dof_values[:min(6, self.state_dim)]
        pos, angle = pos_angle[:3], pos_angle[3:]
        quat = quaternion_multiply(*[axis2quat(axis, theta) for axis, theta in zip(np.eye(3), angle)] + [self._q0])
        self.ogre.setCameraPosition(pos)
        self.ogre.setCameraOrientation(quat)

    def observe(self):
        image = self.ogre.getScreenshot()
        return self._scale_crop(image)


class VideoCaptureThread(threading.Thread):
    def __init__(self, camera_id, warmup_frames=0):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(camera_id)
        self.warmup_frames = warmup_frames
        self.image = None
        self._done = False

    def run(self):
        while not self._done and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                raise RuntimeError("Could not read the image from the capture device")
            if self.warmup_frames > 0:
                self.warmup_frames -= 1
            self.image = image
        self._done = True

    def stop(self):
        self.cap.release()
        self._done = True

    def ready(self):
        return self.image is not None and self.warmup_frames == 0


class ServoPlatform(DiscreteVelocitySimulator, ScaleCropImageSimulator):
    def __init__(self, dof_limits, dof_vel_limits,
                 image_scale=None, crop_size=None,
                 pwm_address=0x40, pwm_freq=60, pwm_channels=(0, 1), camera_id=0, warmup_frames=25):
        """
        DOFs are pan, tilt
        """
        DiscreteVelocitySimulator.__init__(self, dof_limits, dof_vel_limits, dtype=np.int)
        ScaleCropImageSimulator.__init__(self, image_scale=image_scale, crop_size=crop_size)
        # camera initialization
        self.cap_thread = VideoCaptureThread(camera_id, warmup_frames=warmup_frames)
        self.cap_thread.start()
        while not self.cap_thread.ready():
            print "Capture device not ready, waiting..."
            time.sleep(.1)
        # servos initialization
        try:
            from ext.adafruit.Adafruit_PWM_Servo_Driver.Adafruit_PWM_Servo_Driver import PWM
            self.pwm = PWM(pwm_address)
            self.pwm.setPWMFreq(pwm_freq)
            self.pwm_channels = pwm_channels
            self.use_pwm = True
        except:
            self.use_pwm = False
            print "Exception when using pwm. Disabling it."

    def __del__(self):
        self.cap_thread.stop()
        self.cap_thread.join()

    @DiscreteVelocitySimulator.dof_values.setter
    def dof_values(self, next_dof_values):
        assert self._dof_values is None or self._dof_values.shape == next_dof_values.shape
        next_dof_values = np.round(next_dof_values).astype(np.int)
        self._dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])
        if self.use_pwm:
            for channel, dof_value in zip(self.pwm_channels, self._dof_values):
                self.pwm.setPWM(channel, 0, dof_value)
        time.sleep(.5)

    def apply_action(self, vel):
        return super(ServoPlatform, self).apply_action(np.round(vel).astype(np.int))

    def observe(self):
        return self._scale_crop(self.cap_thread.image)

class PR2HeadSimulator(Simulator):
    def __init__(self, robot, vel_max):
        self.robot = robot
        self.vel_max = vel_max
        self.angle_inds = [self.robot.GetJointIndex('head_pan_joint'),
                           self.robot.GetJointIndex('head_tilt_joint')]

    @property
    def angle(self):
        return self.robot.GetDOFValues(self.angle_inds)

    @angle.setter
    def angle(self, next_angle):
        self.robot.SetDOFValues(next_angle, self.angle_inds)

    def apply_action(self, vel):
        angle_prev = self.angle.copy()
        self.angle += vel
        vel = self.angle - angle_prev
        return vel

    def observe(self):
        rgb = (np.random.random((480, 640)) * 255).astype(np.uint8)
        return rgb

    def reset(self, angle):
        self.angle = angle

    @property
    def action_bounds(self):
        action_min = -self.vel_max * np.ones(self.action_dim)
        action_max = self.vel_max * np.ones(self.action_dim)
        return action_min, action_max

    @property
    def state(self):
        return self.angle

    @property
    def action_dim(self):
        dim, = self.angle.shape
        return dim


class PR2Head(PR2HeadSimulator):
    def __init__(self, robot, pr2, vel_max):
        super(PR2Head, self).__init__(robot, vel_max)
        self.pr2 = pr2

        import cloudprocpy
        self.grabber = cloudprocpy.CloudGrabber()
        self.grabber.startRGBD()

    @property
    def angle(self):
        return super(PR2Head, self).angle

    @angle.setter
    def angle(self, next_angle):
        self.robot.SetDOFValues(next_angle, self.angle_inds)
        self.pr2.head.set_pan_tilt(*self.angle)

    def observe(self):
        rgb, _ = self.grabber.getRGBD()
        return rgb
