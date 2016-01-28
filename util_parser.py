import sys
import argparse
import numpy as np
import simulator

# http://stackoverflow.com/questions/6365601/default-sub-command-or-handling-no-sub-command-with-argparse
# https://bitbucket.org/ruamel/std.argparse
def set_default_subparser(self, name, args=None):
    """default subparser selection. Call after setup, just before parse_args()
    name: is the name of the subparser to call by default
    args: if set is the argument list handed to parse_args()

    , tested with 2.7, 3.2, 3.3, 3.4
    it works with 2.6 assuming argparse is installed
    """
    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ['-h', '--help']:  # global help if no subparser
            break
    else:
        for x in self._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            # insert default in first position, this implies no
            # global options without a sub_parsers specified
            if args is None:
                sys.argv.insert(1, name)
            else:
                args.insert(0, name)

def create_square_simulator(args):
    sim = simulator.SquareSimulator(args.image_size, args.square_length, args.abs_vel_max)
    return sim

def create_ogre_simulator(args):
    args.dof_min = args.dof_min[:args.dof]
    args.dof_max = args.dof_max[:args.dof]
    args.vel_min = args.vel_min[:args.dof]
    args.vel_max = args.vel_max[:args.dof]
    sim = simulator.OgreSimulator([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                  background_color=args.background_color,
                                  ogrehead=args.ogrehead,
                                  random_background_color=args.random_background_color,
                                  random_ogrehead=args.random_ogrehead)
    return sim

def create_servo_simulator(args):
    sim = simulator.ServoPlatform([args.dof_min, args.dof_max], [args.vel_min, args.vel_max],
                                  pwm_channels=args.pwm_channels,
                                  camera_id=args.camera_id)
    return sim

def get_sim_args(parser, args):
    for action in parser._actions:
        if type(action) == argparse._SubParsersAction:
            parser_simulator = action.choices[args.simulator]
            break
    sim_args_names = ['simulator'] + [action.dest for action in parser_simulator._actions if type(action) != argparse._HelpAction]
    sim_args = {name: args.__dict__[name] for name in sim_args_names if name in args.__dict__}
    return sim_args

def add_simulator_subparsers(parser):
    subparsers = parser.add_subparsers(dest='simulator', help='simulator')
    parser_square = subparsers.add_parser('square')
    parser_square.add_argument('--image_size', type=int, nargs=2, default=[64, 64], metavar=('HEIGHT', 'WIDTH'))
    parser_square.add_argument('--abs_vel_max', type=float, default=1.0)
    parser_square.add_argument('--square_length', '-l', type=int, default=1, help='required to be odd')
    parser_square.set_defaults(create_simulator=create_square_simulator)

    parser_ogre = subparsers.add_parser('ogre')
    parser_ogre.add_argument('--dof_min', type=float, nargs='+', default=[18, 2, -14, np.deg2rad(-20), np.deg2rad(-20)])
    parser_ogre.add_argument('--dof_max', type=float, nargs='+', default=[24, 6, -2, np.deg2rad(20), np.deg2rad(20)])
    parser_ogre.add_argument('--vel_min', type=float, nargs='+', default=[-0.8]*3 + [np.deg2rad(-7.5)]*2)
    parser_ogre.add_argument('--vel_max', type=float, nargs='+', default=[0.8]*3 + [np.deg2rad(7.5)]*2)
    parser_ogre.add_argument('--dof', type=int, default=5)
    parser_ogre.add_argument('--background_color', type=float, nargs=3, default=[.0]*3, metavar=('R', 'G', 'B'))
    parser_ogre.add_argument('--ogrehead', action='store_true')
    parser_ogre.add_argument('--random_background_color', action='store_true')
    parser_ogre.add_argument('--random_ogrehead', type=int, default=0)
    parser_ogre.set_defaults(create_simulator=create_ogre_simulator)

    parser_servo = subparsers.add_parser('servo')
    parser_servo.add_argument('--dof_min', type=float, nargs='+', default=(230, 220))
    parser_servo.add_argument('--dof_max', type=float, nargs='+', default=(610, 560))
    parser_servo.add_argument('--vel_min', type=float, nargs='+', default=(-50, -50))
    parser_servo.add_argument('--vel_max', type=float, nargs='+', default=(50, 50))
    parser_servo.add_argument('--pwm_channels', '-c', nargs='+', type=int, default=(0, 1))
    parser_servo.add_argument('--camera_id', '-i', type=str, default='C')
    parser_servo.set_defaults(create_simulator=create_servo_simulator)

    parser.set_defaults(get_sim_args=lambda args: get_sim_args(parser, args))

    return subparsers
