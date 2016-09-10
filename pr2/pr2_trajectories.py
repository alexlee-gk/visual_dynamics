import numpy as np
from pr2 import math_utils as mu, retiming, resampling


def follow_body_traj(pr2, bodypart2traj, wait=True, base_frame="/base_footprint", speed_factor=1):
    name2part = {"lgrip": pr2.lgrip,
                 "rgrip": pr2.rgrip,
                 "larm": pr2.larm,
                 "rarm": pr2.rarm,
                 "base": pr2.base}
    for partname in bodypart2traj:
        if partname not in name2part:
            raise Exception("invalid part name %s" % partname)

    #### Go to initial positions #######
    for (name, part) in name2part.items():
        if name in bodypart2traj:
            part_traj = bodypart2traj[name]
            if name == "lgrip" or name == "rgrip":
                part.set_angle(np.squeeze(part_traj)[0])
            elif name == "larm" or name == "rarm":
                part.goto_joint_positions(part_traj[0])
            elif name == "base":
                part.goto_pose(part_traj[0], base_frame)
    pr2.join_all()

    #### Construct total trajectory so we can retime it #######
    n_dof = 0
    trajectories = []
    vel_limits = []
    acc_limits = []
    bodypart2inds = {}
    for (name, part) in name2part.items():
        if name in bodypart2traj:
            traj = bodypart2traj[name]
            if traj.ndim == 1: traj = traj.reshape(-1, 1)
            trajectories.append(traj)
            vel_limits.extend(part.vel_limits)
            acc_limits.extend(part.acc_limits)
            bodypart2inds[name] = range(n_dof, n_dof + part.n_joints)
            n_dof += part.n_joints

    trajectories = np.concatenate(trajectories, 1)

    vel_limits = np.array(vel_limits) * speed_factor

    times = retiming.retime_with_vel_limits(trajectories, vel_limits)
    times_up = np.linspace(0, times[-1], int(np.ceil(times[-1] / .1)))
    traj_up = mu.interp2d(times_up, times, trajectories)

    #### Send all part trajectories ###########
    for (name, part) in name2part.items():
        if name in bodypart2traj:
            part_traj = traj_up[:, bodypart2inds[name]]
            if name == "lgrip" or name == "rgrip":
                part.follow_timed_trajectory(times_up, part_traj.flatten())
            elif name == "larm" or name == "rarm":
                vels = resampling.get_velocities(part_traj, times_up, .001)
                part.follow_timed_joint_trajectory(part_traj, vels, times_up)
            elif name == "base":
                part.follow_timed_trajectory(times_up, part_traj, base_frame)

    if wait:
        pr2.join_all()

    return True
