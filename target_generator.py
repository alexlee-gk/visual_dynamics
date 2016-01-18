from __future__ import division

import numpy as np
import h5py
import util

class TargetGenerator(object):
    def get_target(self):
        raise NotImplementedError


class RandomTargetGenerator(TargetGenerator):
    def __init__(self, sim):
        self.sim = sim

    def get_target(self):
        """
        Changes the state of the sim as a side effect
        """
        dof_values_target = self.sim.sample_state()
        self.sim.reset(dof_values_target)
        image_target = self.sim.observe()
        return image_target, dof_values_target


class Hdf5TargetGenerator(TargetGenerator):
    def __init__(self, hdf5_fname):
        self.hdf5_fname = hdf5_fname
        with h5py.File(self.hdf5_fname, 'r') as hdf5_file:
            self.num_images = len(hdf5_file['image_target'])
        self.image_iter = 0

    def get_target(self):
        with h5py.File(self.hdf5_fname, 'r') as hdf5_file:
            image_target = hdf5_file['image_target'][self.image_iter][()]
            dof_values_target = hdf5_file['pos'][self.image_iter][()]
            self.image_iter += 1
        return image_target, dof_values_target


class InteractiveTargetGenerator(TargetGenerator):
    def __init__(self, sim, vis_scale=1):
        self.sim = sim
        self.vis_scale = vis_scale

    def get_target(self):
        dof_vel_min, dof_vel_max = self.sim.dof_vel_limits
        while True:
            image = self.sim.observe()
            _, exit_request, key = util.visualize_images_callback(image, window_name="Interactive target window", vis_scale=self.vis_scale, delay=100, ret_key=True)
            if exit_request:
                raise KeyboardInterrupt # something else should be raised
            vel = np.zeros(self.sim.state_dim)
            if key == 81: # left arrow
                vel[0] = dof_vel_min[0]
            elif key == 82: # up arrow
                vel[1] = dof_vel_max[1]
            elif key == 83: # right arrow
                vel[0] = dof_vel_max[0]
            elif key == 84: # down arrow
                vel[1] = dof_vel_min[1]
            elif key == 32: # space
                break
            self.sim.apply_action(vel)
        dof_values_target = self.sim.state
        image_target = self.sim.observe()
        return image_target, dof_values_target
