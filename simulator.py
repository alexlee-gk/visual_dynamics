import time
import numpy as np
import cv2
import video
import util

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

    def stop(self):
        return


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
    def __init__(self, dof_limits, dof_vel_limits, dof_vel_scale=None, dtype=None):
        dof_min, dof_max = dof_limits
        vel_min, vel_max = dof_vel_limits
        assert len(dof_min) == len(dof_max)
        assert len(vel_min) == len(vel_max)
        self.dof_limits = [np.asarray(dof_limit, dtype=dtype) for dof_limit in dof_limits]
        self.dof_vel_limits = [np.asarray(dof_vel_limit, dtype=dtype) for dof_vel_limit in dof_vel_limits]
        self._dof_values = np.mean(self.dof_limits, axis=0)
        if dof_vel_scale is None:
            self.dof_vel_scale = np.ones(self.state_dim)
        else:
            self.dof_vel_scale = dof_vel_scale

    @property
    def dof_values(self):
        return self._dof_values.copy()

    @dof_values.setter
    def dof_values(self, next_dof_values):
        assert self._dof_values is None or self._dof_values.shape == next_dof_values.shape
        self._dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])

    def apply_action(self, vel):
        dof_values_prev = self.dof_values.copy()
        self.dof_values += (self.dof_vel_scale * vel).astype(self.dof_values.dtype)
        vel = (self.dof_values - dof_values_prev) / self.dof_vel_scale # recompute vel because of clipping
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


class ImageTransformer(object):
    def __init__(self, image_scale=None, crop_size=None, crop_offset=None):
        self.image_scale = image_scale
        self.crop_size = np.asarray(crop_size) if crop_size is not None else None
        self.crop_offset = np.asarray(crop_offset) if crop_offset is not None else None

    def transform(self, image):
        need_swap_channels = (image.ndim == 3 and image.shape[0] == 3)
        if need_swap_channels:
            image = image.transpose(1, 2, 0)
        if self.image_scale is not None and self.image_scale != 1.0:
            image = cv2.resize(image, (0, 0), fx=self.image_scale, fy=self.image_scale)
        if self.crop_size is not None and tuple(self.crop_size) != image.shape[:2]:
            h, w = image_shape = np.asarray(image.shape[:2])
            crop_h, crop_w = self.crop_size
            if crop_h > h:
                raise ValueError('crop height %d is larger than image height %d (after scaling)'%(crop_h, h))
            if crop_w > w:
                raise ValueError('crop width %d is larger than image width %d (after scaling)'%(crop_w, w))
            crop_origin = image_shape/2
            if self.crop_offset is not None:
                crop_origin += self.crop_offset
            crop_corner = crop_origin - self.crop_size/2
            if not (np.all(np.zeros(2) <= crop_corner) and np.all(crop_corner + self.crop_size <= image_shape)):
                raise IndexError('crop indices out of range')
            image = image[crop_corner[0]:crop_corner[0] + crop_h,
                          crop_corner[1]:crop_corner[1] + crop_w,
                          ...]
        if need_swap_channels:
            image = image.transpose(2, 0, 1)
        return image


class NodeTrajectoryManager(object):
    def __init__(self, ogre, node_name, start, end, num_steps):
        self.ogre = ogre
        self.node_name = node_name
        self.start = start
        self.end = end
        self.num_steps = num_steps
        self.t = 0.0
        self.ogre.setNodePosition(self.node_name, self._calculate_position())
        self.dir = 1.0

    def step(self):
        self.t += self.dir * (1. / self.num_steps)
        if not (0 <= self.t <= 1.):
            self.dir *= -1.
            self.t += 2. * self.dir * (1. / self.num_steps)
        self.ogre.setNodePosition(self.node_name, self._calculate_position())

    def _calculate_position(self):
        assert 0 <= self.t <= 1
        return (1.0 - self.t) * self.start + self.t * self.end


class OgreSimulator(DiscreteVelocitySimulator):
    def __init__(self, dof_limits, dof_vel_limits, dof_vel_scale=None, background_color=None, ogrehead=False, random_background_color=False, random_ogrehead=0):
        """
        DOFs are x, y, z, angle_x, angle_y, angle_z
        """
        DiscreteVelocitySimulator.__init__(self, dof_limits, dof_vel_limits, dof_vel_scale=dof_vel_scale)
        self._q0 = axis2quat(np.array([0, 1, 0]), np.pi/2)

        import pygre
        self.ogre = pygre.Pygre()
        self.ogre.init()
        if background_color is not None:
            self.ogre.setBackgroundColor(np.asarray(background_color))
        self.ogre.addNode(b'house', b'house.mesh', 0, 0, 0)
        self.traj_managers = []
        if ogrehead:
            start = np.array([12, 2.5, -2])
            end = np.array([12, 2.5, -14])
            num_steps = 30
            self.ogre.addNode(b'ogrehead', b'ogrehead.mesh', *start) #([far, close], [down, up], [right, left])
            self.ogre.setNodeScale(b'ogrehead', np.array([.03]*3))
            self.traj_managers.append(NodeTrajectoryManager(self.ogre, b'ogrehead', start, end, num_steps))
        self.random_background_color = random_background_color
        self.random_ogrehead = random_ogrehead
        for ogrehead_iter in range(self.random_ogrehead):
            self.ogre.addNode(b'ogrehead%d'%ogrehead_iter, b'ogrehead.mesh', 0, 0, 0)
            self.ogre.setNodeScale(b'ogrehead%d'%ogrehead_iter, np.array([.03]*3))
        self.ogre.setCameraOrientation(self._q0)

    @DiscreteVelocitySimulator.dof_values.setter
    def dof_values(self, next_dof_values):
        assert self._dof_values is None or self._dof_values.shape == next_dof_values.shape
        self._dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])
        pos_angle = np.zeros(6)
        pos_angle[:min(6, self.state_dim)] += self._dof_values[:min(6, self.state_dim)]
        pos = pos_angle[:3]
        self.ogre.setCameraPosition(pos)
        if self.state_dim > 3:
            angle = pos_angle[3:]
            quat = quaternion_multiply(*[axis2quat(axis, theta) for axis, theta in zip(np.eye(3), angle)] + [self._q0])
            self.ogre.setCameraOrientation(quat)

    def apply_action(self, vel):
        vel = super(OgreSimulator, self).apply_action(vel)
        if self.traj_managers:
            for manager in self.traj_managers:
                manager.step()
        return vel

    def reset(self, dof_values):
        super(OgreSimulator, self).reset(dof_values)
        if self.random_background_color:
            self.ogre.setBackgroundColor(np.random.random(3))
        for ogrehead_iter in range(self.random_ogrehead):
            ogrehead_pos_min = np.array([12, 0, -15])
            ogrehead_pos_max = np.array([12, 5, 0])
            ogrehead_pos = ogrehead_pos_min + np.random.random(3) * (ogrehead_pos_max - ogrehead_pos_min)
            self.ogre.setNodePosition(b'ogrehead%d'%ogrehead_iter, ogrehead_pos)

    def observe(self):
        image = self.ogre.getScreenshot()
        return util.obs_from_image(image)


class CarNodeTrajectoryManager(object):
    def __init__(self, ogre, node_name, dof_values_init, dof_vel_init, dof_limits, dof_vel_limits, dof_acc_limits, max_travel_distance=np.inf):
        self.ogre = ogre
        self.node_name = node_name
        state_dim = len(dof_values_init)
        for limit in dof_limits + dof_vel_limits + dof_acc_limits:
            assert len(limit) == state_dim
        self.dof_limits = [np.asarray(dof_limit) for dof_limit in dof_limits]
        self.dof_vel_limits = [np.asarray(dof_vel_limit, dtype=float) for dof_vel_limit in dof_vel_limits]
        self.dof_acc_limits = [np.asarray(dof_acc_limit, dtype=float) for dof_acc_limit in dof_acc_limits]
        self.reset(dof_values_init, dof_vel_init)
        self.max_travel_distance = max_travel_distance

    def reset(self, dof_values, dof_vel=None):
        state_dim = len(dof_values)
        if dof_vel is not None:
            self._dof_vel = np.asarray(dof_vel, dtype=float)
        self._dof_acc = np.zeros(state_dim)
        self.dof_values = np.asarray(dof_values, dtype=float)
        self._dof_values_reset = np.asarray(dof_values, dtype=float).copy()

    @property
    def distance_traveled(self):
        return abs(self._dof_values[2] - self._dof_values_reset[2])

    @property
    def dof_values(self):
        return self._dof_values.copy()

    @dof_values.setter
    def dof_values(self, next_dof_values):
        self._dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])
        self.ogre.setNodePosition(self.node_name, self._dof_values)

    @property
    def dof_vel(self):
        return self._dof_vel.copy()

    @dof_vel.setter
    def dof_vel(self, next_dof_vel):
        self._dof_vel = np.clip(next_dof_vel, self.dof_vel_limits[0], self.dof_vel_limits[1])

    @property
    def dof_acc(self):
        return self._dof_acc.copy()

    @dof_acc.setter
    def dof_acc(self, next_dof_acc):
        self._dof_acc = np.clip(next_dof_acc, self.dof_acc_limits[0], self.dof_acc_limits[1])

    def apply_action(self, next_acc):
        dof_values_prev = self.dof_values.copy()
        dof_vel_prev = self.dof_vel.copy()
        self.dof_acc = next_acc
        self.dof_vel += self.dof_acc
        self.dof_values += self.dof_vel
        # update with actual values
        self.dof_vel = self.dof_values - dof_values_prev
        self.dof_acc = self.dof_vel - dof_vel_prev

    def step(self):
        if self.distance_traveled < self.max_travel_distance:
            dof_acc_min, dof_acc_max = self.dof_acc_limits
            acc = dof_acc_min + np.random.random_sample(dof_acc_min.shape) * (dof_acc_max - dof_acc_min)
            self.apply_action(acc)


class CityOgreSimulator(OgreSimulator):
    def __init__(self, dof_limits, dof_vel_limits, dof_vel_scale=None, dof_vel_offset=None, static_car=False):
        DiscreteVelocitySimulator.__init__(self, dof_limits, dof_vel_limits, dof_vel_scale=dof_vel_scale)
        self._q0 = np.array([1., 0., 0., 0.])

        import pygre
        self.ogre = pygre.Pygre()
        self.ogre.init()
        self.ogre.addNode(b'city', b'_urban-level-02-medium-3ds_3DS.mesh', 0, 0, 0)
        self.traj_managers = []
        node_name = b'car'
        self.ogre.addNode(node_name, b'camaro2_3ds.mesh', -51, 10.7, 225)
        self.ogre.setNodeScale(node_name, np.array([0.3]*3))
        self.ogre.setNodeOrientation(node_name, axis2quat(np.array((0,1,0)), np.deg2rad(180)))
        car_dof_values_init = [-51, 10.7, 225]
        car_dof_limits = [[-51-6, 10.7, -275], [-51+6, 10.7, 225]]
        if static_car:
            car_dof_vel_init = np.zeros(3)
            car_dof_vel_limits = [np.zeros(3), np.zeros(3)]
            car_dof_acc_limits = [np.zeros(3), np.zeros(3)]
        else:
            car_dof_vel_init = [0, 0, -1]
            car_dof_vel_limits = [[-1, 0, -10], [1, 0, -1]]
            car_dof_acc_limits = [[-.1, 0, 0], [.1, 0, 0]]
        self.car_traj_manager = CarNodeTrajectoryManager(self.ogre, node_name, car_dof_values_init, car_dof_vel_init, car_dof_limits, car_dof_vel_limits, car_dof_acc_limits)
        self.traj_managers.append(self.car_traj_manager)
        self.ogre.setCameraOrientation(quaternion_multiply(axis2quat(np.array((1,0,0)), -np.pi/4), self._q0)) # look diagonally downwards at a 45 deg angle

    def reset(self, dof_values):
        DiscreteVelocitySimulator.reset(self, dof_values)

    def observe(self):
        image = self.ogre.getScreenshot()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return util.obs_from_image(image)

    def sample_state(self):
        car_dof_values = util.sample_interval(*self.car_traj_manager.dof_limits)
        self.car_traj_manager.reset(car_dof_values)
        # constrain sampled state to be in the 45 deg line of sight
        val = 5 + np.random.random_sample(1) * 50
        dof_values = np.array([-51, 10.7 + val, car_dof_values[2] + val, -np.pi/4, 0])
        return dof_values[:self.state_dim]

    @staticmethod
    def look_at(target_pos, camera_pos):
        ax = target_pos - camera_pos
        pan = np.arctan2(-ax[0], -ax[2])
        tilt = np.arcsin(ax[1] / np.linalg.norm(ax))
        dof_values = np.concatenate([camera_pos, np.array([tilt, pan])])
        return dof_values

class ServoPlatform(DiscreteVelocitySimulator):
    def __init__(self, dof_limits, dof_vel_limits, dof_vel_scale=None,
                 pwm_address=0x40, pwm_freq=60, pwm_channels=(0, 1), pwm_extra_delay=.5,
                 camera_id=0,
                 delay=True):
        """
        DOFs are pan, tilt
        """
        DiscreteVelocitySimulator.__init__(self, dof_limits, dof_vel_limits, dof_vel_scale=dof_vel_scale, dtype=np.int)
        self.delay = delay
        # camera initialization
        if isinstance(camera_id, str):
            if camera_id.isdigit():
                camera_id = int(camera_id)
            else:
                camera_id = util.device_id_from_camera_id(camera_id)
        self.cap = video.VideoCapture(device=camera_id)
        # servos initialization
        self.last_time = -np.inf
        try:
            from ext.adafruit.Adafruit_PWM_Servo_Driver.Adafruit_PWM_Servo_Driver import PWM
            self.pwm = PWM(pwm_address)
            self.pwm.setPWMFreq(pwm_freq)
            self.pwm_channels = pwm_channels
            self.pwm_extra_delay = pwm_extra_delay
            self.use_pwm = True
            self.dof_values = self._dof_values
            duration = self.duration_dof_vel(np.diff(self.dof_limits, axis=0).max())
            if self.delay:
                time.sleep(duration + self.pwm_extra_delay)
            self.last_time = self.cap.get_time()
        except Exception as e:
            self.use_pwm = False
            print("Exception when using pwm: %s. Disabling it."%e)

    def stop(self):
        self.cap.release()

    @DiscreteVelocitySimulator.dof_values.setter
    def dof_values(self, next_dof_values):
        assert self._dof_values is None or self._dof_values.shape == next_dof_values.shape
        next_dof_values = np.round(next_dof_values).astype(np.int)
        next_dof_values = np.clip(next_dof_values, self.dof_limits[0], self.dof_limits[1])
        dof_changes = np.abs(next_dof_values - self._dof_values)
        self._dof_values = next_dof_values
        if self.use_pwm:
            finish_times = []
            for channel, dof_value, dof_change in zip(self.pwm_channels, self._dof_values, dof_changes):
                self.pwm.setPWM(channel, 0, dof_value)
                finish_time = time.time() + self.duration_dof_vel(dof_change)
                finish_times.append(finish_time)
            duration = time.time() - np.max(finish_times)
            if self.delay:
                time.sleep(max(0, duration + self.pwm_extra_delay))
            self.last_time = self.cap.get_time()

    def reset(self, dof_values):
        self.dof_values = dof_values
        if self.delay:
            time.sleep(2.5)

    def observe(self):
        while True:
            image, image_time = self.cap.get()
            if image_time > self.last_time:
                break
        return util.obs_from_image(image)

    def duration_dof_vel(self, dof_vel):
        """
        Time duration the servo (HS-5645) takes to change its dof by dof_vel.
        """
        deg_vel = 90.0 * dof_vel / 250.0
        duration = 0.23 * deg_vel / 60.0
        return abs(duration)


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
