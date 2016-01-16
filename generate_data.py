from __future__ import division

import argparse
import numpy as np
import cv2
import h5py
import simulator
import controller
import util

class DataCollector(object):
    def __init__(self, fname, num_datum, auto_shuffle=True, sim_args=None):
        self.fname = fname
        self.num_datum = num_datum
        self.auto_shuffle = auto_shuffle
        self.file = h5py.File(self.fname, 'a')
        self.datum_iter = 0
        if sim_args is not None:
            g = self.file.require_group('sim_args')
            for arg_name, arg in sim_args.items():
                if arg_name in g:
                    g[arg_name][...] = arg
                else:
                    g[arg_name] = arg

    def add(self, **datum):
        if self.datum_iter < self.num_datum:
            for key, value in datum.items():
                shape = (self.num_datum, ) + value.shape
                dset = self.file.require_dataset(key, shape, value.dtype, exact=True)
                dset[self.datum_iter] = value
            if self.datum_iter == (self.num_datum - 1):
                if self.auto_shuffle:
                    self.shuffle()
                self.file.close()
        else:
            raise RuntimeError("Tried to add more data than specified (%d)"%self.num_datum)
        self.datum_iter += 1

    def shuffle(self):
        inds = None
        for key, dataset in self.file.iteritems():
            if type(dataset) != h5py.Dataset:
                continue
            if inds is None:
                inds = np.arange(dataset.shape[0])
                np.random.shuffle(inds)
            else:
                assert len(inds) == dataset.shape[0]
            self.file[key][:] = dataset[()][inds]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=0)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--background_window', '-b', action='store_true')
    parser.add_argument('--background_window_size', type=int, nargs='+', default=[5, 8], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--image_size', type=int, nargs=2, default=[64, 64], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--simulator', '-s', type=str, default='ogre', choices=('square', 'ogre', 'servo'))
    # square simulator
    parser.add_argument('--abs_vel_max', type=float, default=1.0)
    parser.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    # ogre simulator
    parser.add_argument('--dof_min', type=float, nargs='+', default=None)
    parser.add_argument('--dof_max', type=float, nargs='+', default=None)
    parser.add_argument('--vel_min', type=float, nargs='+', default=None)
    parser.add_argument('--vel_max', type=float, nargs='+', default=None)
    parser.add_argument('--dof', type=int, default=5)
    parser.add_argument('--image_scale', '-f', type=float, default=0.15)

    args = parser.parse_args()
    if args.simulator == 'ogre':
        args.dof_min = args.dof_min or [18, 2, -6, np.deg2rad(-20), np.deg2rad(-20)]
        args.dof_max = args.dof_max or [24, 6, -2, np.deg2rad(20), np.deg2rad(20)]
        args.vel_min = args.vel_min or [-0.8]*3 + [np.deg2rad(-7.5)]*2
        args.vel_max = args.vel_max or [0.8]*3 + [np.deg2rad(7.5)]*2
        args.dof_min = args.dof_min[:args.dof]
        args.dof_max = args.dof_max[:args.dof]
        args.vel_min = args.vel_min[:args.dof]
        args.vel_max = args.vel_max[:args.dof]
    else:
        args.dof_min = args.dof_min or (230, 220)
        args.dof_max = args.dof_max or (610, 560)
        args.vel_min = args.vel_min or (-50, -50)
        args.vel_max = args.vel_max or (50, 50)

    if args.simulator == 'square':
        sim = simulator.SquareSimulator(args.image_size, args.square_length, args.abs_vel_max)
    elif args.simulator == 'ogre':
        sim = simulator.OgreSimulator([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                      image_scale=args.image_scale, crop_size=args.image_size)
    elif args.simulator == 'servo':
        sim = simulator.ServoPlatform([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                      image_scale=args.image_scale, crop_size=args.image_size)
    else:
        raise
    ctrl = controller.RandomController(*sim.action_bounds)
    if args.output:
        sim_args = dict(image_size=args.image_size, simulator=args.simulator)
        if args.simulator == 'square':
            sim_args.update(dict(abs_vel_max=args.abs_vel_max, square_length=args.square_length))
        elif args.simulator == 'ogre' or args.simulator == 'servo':
            sim_args.update(dict(dof_min=args.dof_min, dof_max=args.dof_max,
                                 vel_min=args.vel_min, vel_max=args.vel_max,
                                 image_scale=args.image_scale))
        else:
            raise
        collector = DataCollector(args.output, args.num_trajs * args.num_steps, sim_args=sim_args, auto_shuffle=args.shuffle)
    else:
        collector = None

    if args.background_window:
        cv2.namedWindow("Background window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Background window", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    done = False
    for traj_iter in range(args.num_trajs):
        if traj_iter%10 == 0:
            print traj_iter, traj_iter
        try:
            if args.background_window:
                background_shape = (np.random.randint(max(0, args.background_window_size[0]+1-3), args.background_window_size[0]+1),
                                    np.random.randint(max(0, args.background_window_size[0]+1-3), args.background_window_size[1]+1))
                cv2.imshow("Background window", (np.ones(background_shape)[..., None] * np.random.random(3)[None, None, :]))
                key = cv2.waitKey(1)
                key &= 255
                if key == 27 or key == ord('q'):
                    print "Pressed ESC or q, exiting"
                    done = True
                    break
            pos_init = sim.sample_state()
            sim.reset(pos_init)
            for step_iter in range(args.num_steps):
                state = sim.state
                image = sim.observe()
                action = ctrl.step(image)
                action = sim.apply_action(action)
                image_next = sim.observe()

                if collector:
                    collector.add(image_curr=image,
                                  image_diff=image_next - image,
                                  pos=state,
                                  vel=action)

                # visualization
                if args.visualize:
                    vis_image, done = util.visualize_images_callback(image, vis_scale=args.vis_scale)
            if done:
                break
        except KeyboardInterrupt:
            break

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
