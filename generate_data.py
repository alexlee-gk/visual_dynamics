from __future__ import division

import argparse
import numpy as np
import cv2
import h5py
import simulator
import controller
import util

class DataCollector(object):
    def __init__(self, fname, num_datum, auto_shuffle=True):
        self.fname = fname
        self.num_datum = num_datum
        self.auto_shuffle = auto_shuffle
        self.file = h5py.File(self.fname, 'a')
        self.datum_iter = 0

    def add(self, **datum):
        if self.datum_iter < self.num_datum:
            for key, value in datum.items():
                shape = (self.num_datum, ) + value.shape
                if key in self.file:
                    if self.file[key].shape != shape:
                        raise RuntimeError("File already exists and shapes don't match")
                else:
                    self.file.create_dataset(key, shape)
                self.file[key][self.datum_iter] = value
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
            if inds is None:
                inds = np.arange(dataset.shape[0])
                np.random.shuffle(inds)
            else:
                assert len(inds) == dataset.shape[0]
            self.file[key][:] = dataset[()][inds]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--num_trajs', '-n', type=int, default=10, metavar='N', help='total number of data points is N*T')
    parser.add_argument('--num_steps', '-t', type=int, default=10, metavar='T', help='number of time steps per trajectory')
    parser.add_argument('--visualize', '-v', type=int, default=1)
    parser.add_argument('--vis_scale', '-r', type=int, default=10, metavar='R', help='rescale image by R for visualization')
    parser.add_argument('--vel_max', '-m', type=float, default=None)
    parser.add_argument('--image_size', type=int, nargs=2, default=[84, 84], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--simulator', '-s', type=str, default='square', choices=('square', 'ogre'))
    # square simulator
    parser.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    # ogre simulator
    parser.add_argument('--pos_min', type=float, nargs=3, default=[19, 2, -10], metavar=tuple([xyz + '_pos_min' for xyz in 'xyz']))
    parser.add_argument('--pos_max', type=float, nargs=3, default=[21, 6, -6], metavar=tuple([xyz + '_pos_max' for xyz in 'xyz']))
    parser.add_argument('--image_scale', '-f', type=float, default=0.25)

    args = parser.parse_args()
    if args.simulator == 'square':
        if args.vel_max is None:
            args.vel_max = 1.0
    elif args.simulator == 'ogre':
        if args.vel_max is None:
            args.vel_max = 0.2
    else:
        raise
    args.pos_min = np.asarray(args.pos_min)
    args.pos_max = np.asarray(args.pos_max)

    if args.simulator == 'square':
        sim = simulator.SquareSimulator(args.image_size, args.square_length, args.vel_max)
    elif args.simulator== 'ogre':
        sim = simulator.OgreSimulator(args.pos_min, args.pos_max, args.vel_max,
                                      image_scale=args.image_scale, crop_size=args.image_size)
    else:
        raise
    ctrl = controller.RandomController(*sim.action_bounds)
    collector = DataCollector(args.output, args.num_trajs * args.num_steps) if args.output else None

    done = False
    for traj_iter in range(args.num_trajs):
        try:
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
                                  image_next=image_next,
                                  image_diff=image_next - image,
                                  pos=state,
                                  vel=action)

                # visualization
                if args.visualize:
                    vis_image = util.resize_from_scale((image.transpose(1, 2, 0) * 255.0).astype(np.uint8), args.vis_scale)
                    cv2.imshow("Image window", vis_image)
                    key = cv2.waitKey(100)
                    key &= 255
                    if key == 27 or key == ord('q'):
                        print "Pressed ESC or q, exiting"
                        done = True
                        break
            if done:
                break
        except KeyboardInterrupt:
            break

    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
