import numpy as np
import json
import trajoptpy


def plan_up_trajectory(robot, lr, target_pos, max_displacement=0.05, use_cost=False):
    manip_name = {"l": "leftarm", "r": "rightarm"}[lr]
    tool_link_name = "%s_gripper_tool_frame" % lr
    manip = robot.GetManipulator(manip_name)
    tool_link = robot.GetLink(tool_link_name)
    n_steps = int(np.ceil(1.5 * np.linalg.norm(target_pos - tool_link.GetTransform()[:3, 3]) / max_displacement))

    request = {
        "basic_info": {
            "n_steps": n_steps,
            "manip": manip_name,
            "start_fixed": True
        },
        "costs": [
            {
                "type": "joint_vel",
                "params": {"coeffs": [1]}
            },
            {
                "type": "collision",
                "params": {
                    "coeffs": [20],
                    "dist_pen": [0.025]
                }
            }
        ],
        "constraints": [
            {
                "type": "pose",
                "params": {
                    "xyz": target_pos,
                    "wxyz": [0, 1, 0, 0],  # unused
                    "link": tool_link_name,
                    "rot_coeffs": [0, 0, 0],
                    "pos_coeffs": [10, 10, 10]
                }
            },
            {
                "type": "cart_vel",
                "name": "cart_vel",
                "params": {
                    "max_displacement": max_displacement,
                    "first_step": 0,
                    "last_step": n_steps - 1,  # inclusive
                    "link": tool_link_name
                },
            }
        ],
        "init_info": {
            "type": "stationary"
        }
    }
    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, robot.GetEnv())  # create object that stores optimization problem

    # add up costs / constraints
    local_dir = np.array([0., 0., 1.])  # up
    arm_inds = manip.GetArmIndices()
    arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]

    def f(x):
        robot.SetDOFValues(x, arm_inds, False)
        return tool_link.GetTransform()[:2, :3].dot(local_dir)

    def dfdx(x):
        robot.SetDOFValues(x, arm_inds, False)
        world_dir = tool_link.GetTransform()[:3, :3].dot(local_dir)
        return np.array([np.cross(joint.GetAxis(), world_dir)[:2] for joint in arm_joints]).T.copy()

    if use_cost:
        for t in xrange(1, n_steps):
            prob.AddErrorCost(f, dfdx, [(t, j) for j in xrange(7)], "ABS", "up%i" % t)
    else:  # use constraint
        for t in xrange(1, n_steps):
            prob.AddConstraint(f, dfdx, [(t, j) for j in xrange(7)], "EQ", "up%i" % t)

    result = trajoptpy.OptimizeProblem(prob)  # do optimization
    return result.GetTraj()
