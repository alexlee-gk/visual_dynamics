from __future__ import division

import argparse
import numpy as np
import cv2
import h5py
import util

class ImageCollectorAndController(object):
    def __init__(self, sim, **kwargs):
        self.num_trajs = kwargs['num_trajs']
        self.num_steps = kwargs['num_steps']
        
        if kwargs['output'] is not None:
            self.f = h5py.File(kwargs['output'], "a")
        else:
            self.f = None

        self.sim = sim
        
        self.done = False

    def generate_initial_position(self):
        pos0 = np.asarray(self.sim.pos_min + np.random.random(2) * (self.sim.pos_max - self.sim.pos_min), dtype=int)
        return pos0

    def image_callback(self, image, pos, traj_iter, step_iter):
        vel = np.zeros(2)
        return vel
    
    def start_loop(self):
        if self.f is not None:
            data_iter = 0
            image_prev = None
        
        for traj_iter in range(self.num_trajs):
            try:
                pos0 = self.generate_initial_position()
                self.sim.pos = pos0
                for step_iter in range(self.num_steps):
                    image = self.sim.image
                    pos = self.sim.pos
                    vel = self.image_callback(image, pos, traj_iter, step_iter)
            
                    # save data
                    if self.f is not None:
                        image_std = util.standarize(image)
                        data_keys = ["image_curr", "image_next", "image_diff", "vel", "pos"]
                        num_data = self.num_trajs * self.num_steps
                        image_shape = (num_data, 1) + image_std.shape
                        data_shapes = [image_shape,  image_shape, image_shape, (num_data,  len(vel)), (num_data, len(pos))]
                        for data_key, data_shape in zip(data_keys, data_shapes):
                            if data_key in self.f:
                                if self.f[data_key].shape != data_shape:
                                    print "Error, file with different structure exists, shutting down"
                                    self.done
                                    return
                            else:
                                self.f.create_dataset(data_key, data_shape)
                        assert data_iter == (traj_iter * self.num_steps + step_iter)
                        if step_iter != 0:
                            image_prev_std = util.standarize(image_prev)
                            self.f["image_next"][data_iter-1] = np.expand_dims(image_std, axis=0)
                            self.f["image_diff"][data_iter-1] = np.expand_dims(image_std - image_prev_std, axis=0)
                        if step_iter != self.num_steps:
                            self.f["image_curr"][data_iter] = np.expand_dims(image_std, axis=0)
                            self.f["vel"][data_iter] = vel
                            self.f["pos"][data_iter] = pos
                            data_iter += 1
                    
                    image_prev = image.copy()

                    if self.done:
                        break
            except KeyboardInterrupt:
                self.done = True
            if self.done:
                break

class ImageCollectorAndRandomController(ImageCollectorAndController):
    def __init__(self, **kwargs):
        super(ImageCollectorAndRandomController, self).__init__(**kwargs)
        self.vel_max = np.asarray(kwargs['vel_max'])

        self.visualize = kwargs['visualize']
        if self.visualize:
            cv2.namedWindow("Image window", 1)
            self.vis_rescale_factor = kwargs['vis_rescale_factor']

    def image_callback(self, image, pos, traj_iter, step_iter):
        # generate and apply action
        vel = np.asarray([np.random.random_integers(-veli_max, veli_max) for veli_max in self.vel_max], dtype=int)
        vel = self.sim.apply_velocity(vel)

        # visualization
        if self.visualize:
            vis_image = util.resize_from_scale(image, self.vis_rescale_factor)
            cv2.imshow("Image window", vis_image)
            key = cv2.waitKey(1)
            key &= 255
            if key == 27 or key == ord('q'):
                print "Pressed ESC or q, exiting"
                self.done = True
                return

        return vel

class SquareSimulator(object):
    def __init__(self, image_size, square_length):
        assert len(image_size) == 2
        assert square_length%2 != 0
        self._image = np.zeros(image_size, dtype=np.uint8)
        self.square_length = square_length
        self.pos_min = np.asarray([square_length//2]*2, dtype=int)
        self.pos_max = np.asarray([l - 1 - square_length//2 for l in image_size], dtype=int)
        self._pos = None
        self.pos = (self.pos_min + self.pos_max)//2

    @property
    def image(self):
#         return self._image.copy()
        return cv2.GaussianBlur(self._image, (5, 5), 0)

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
    
    def apply_velocity(self, vel):
        pos_prev = self.pos.copy()
        self.pos += vel
        vel = self.pos - pos_prev # recompute vel because of clipping
        return vel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--image_size', '-s', type=int, nargs=2, default=[7, 10], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    parser.add_argument('--vel_max', type=int, nargs=2, default=[1, 1], metavar=tuple([ij + '_vel_max' for ij in 'ij']))
    parser.add_argument('--vis_rescale_factor', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    
    args = parser.parse_args()
    
    sim = SquareSimulator(args.image_size, args.square_length)

    image_collector_ctrl = ImageCollectorAndRandomController(**dict(vars(args).items() + [('sim', sim)]))
    image_collector_ctrl.start_loop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
