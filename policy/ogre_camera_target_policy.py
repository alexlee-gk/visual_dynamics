import numpy as np
import utils.transformations as tf
from policy import Policy


class OgreCameraTargetPolicy(Policy):
    def __init__(self, env, target_env, camera_node_name, agent_node_name, target_node_name, offset, tightness=0.1):
        """
        Args:
            tightness: float between 0 and 1 inclusive
        """
        self.env = env
        self.target_env = target_env
        self.camera_node_name = camera_node_name
        self.agent_node_name = agent_node_name
        self.target_node_name = target_node_name
        # initialize nodes from node names
        def get_descendants(node):
            descendants = node.getChildren()
            for child_node in node.getChildren():
                descendants.extend(get_descendants(child_node))
            return descendants
        nodes = get_descendants(env.app.scene_manager.getRootSceneNode())
        name_to_node = {node.getName(): node for node in nodes}
        camera_node = name_to_node[camera_node_name]
        agent_node = name_to_node[agent_node_name]
        target_node = name_to_node[target_node_name]
        # camera node must be a child of the agent node
        assert camera_node.getParent() == agent_node
        # agent node must be a child of the root node
        assert agent_node.getParent() is not None and \
               agent_node.getParent().getParent() is None
        # target node must be a child of the root node
        assert target_node.getParent() is not None and \
               target_node.getParent().getParent() is None
        self.camera_node = camera_node
        self.agent_node = agent_node
        self.target_node = target_node
        self.offset = np.asarray(offset)
        self.tightness = tightness

    def compute_desired_agent_transform(self, tightness=None):
        tightness = self.tightness if tightness is None else tightness
        # target transform in world coordinates
        target_T = self.target_node.getTransform()
        # target offset relative to target
        target_to_offset_T = tf.translation_matrix(self.offset)
        offset_T = target_T.dot(target_to_offset_T)
        # agent transform in world coordinates
        agent_T = self.agent_node.getTransform()
        # camera transform relative to the agent
        agent_to_camera_T = self.camera_node.getTransform()
        # camera transform in world coordinates
        camera_T = agent_T.dot(agent_to_camera_T)
        # set camera position to the offset target while limiting how fast to move there (controlled by tightness)
        des_camera_pos = (1 - tightness) * camera_T[:3, 3] + tightness * offset_T[:3, 3]
        # point camera in the direction of the target while fixing the yaw axis to be vertical
        des_camera_direction = (target_T[:3, 3] - des_camera_pos)
        des_camera_rot_z = -des_camera_direction
        des_camera_rot_z /= np.linalg.norm(des_camera_rot_z)
        des_camera_rot_x = np.cross(np.array([0., 0., 1.]), des_camera_rot_z)
        des_camera_rot_x /= np.linalg.norm(des_camera_rot_x)
        des_camera_rot_y = np.cross(des_camera_rot_z, des_camera_rot_x)
        des_camera_rot_y /= np.linalg.norm(des_camera_rot_y)
        des_camera_rot = np.array([des_camera_rot_x, des_camera_rot_y, des_camera_rot_z]).T
        # desired camera transform in world coordinates
        des_camera_T = np.eye(4)
        des_camera_T[:3, 3] = des_camera_pos
        des_camera_T[:3, :3] = des_camera_rot
        # agent transform relative to the camera
        camera_to_agent_T = tf.inverse_matrix(agent_to_camera_T)
        # desired agent transform in world coordinates
        des_agent_T = des_camera_T.dot(camera_to_agent_T)
        return des_agent_T

    def act(self, obs):
        # agent transform in world coordinates
        agent_T = self.agent_node.getTransform()
        # desired agent transform in world coordinates
        des_agent_T = self.compute_desired_agent_transform()
        # desired agent transform relative to the agent
        agent_to_des_agent_T = tf.inverse_matrix(agent_T).dot(des_agent_T)
        action, action_rem = np.split(tf.position_axis_angle_from_matrix(agent_to_des_agent_T) / self.env.dt,
                                      self.env.action_space.shape)
        assert all(action_rem == 0)  # action_rem may be empty, in which case this is also True
        return action

    def reset(self):
        # save original state of the target environment
        orig_target_state = self.target_env.get_state()
        # reset the target environment to a random state
        self.target_env.reset()
        target_state = self.target_env.get_state()
        # compute the agent transform for the new state of the target environment
        des_agent_T = self.compute_desired_agent_transform(tightness=1.0)
        # restore original state of the target environment
        self.target_env.reset(orig_target_state)
        return np.concatenate([tf.position_axis_angle_from_matrix(des_agent_T), target_state])

    def _get_config(self):
        config = super(OgreCameraTargetPolicy, self)._get_config()
        config.update({'env': self.env,
                       'target_env': self.target_env,
                       'camera_node_name': self.camera_node_name,
                       'agent_node_name': self.agent_node_name,
                       'target_node_name': self.target_node_name,
                       'offset': self.offset.tolist(),
                       'tightness': self.tightness})
        return config
