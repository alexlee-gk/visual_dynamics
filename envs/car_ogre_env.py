import numpy as np
from collections import defaultdict
import os.path
import ogre
import collada
from envs import OgreEnv, CityOgreEnv
import utils.transformations as tf
import spaces


class CarOgreEnv(OgreEnv):
    def __init__(self, action_space, observation_space, sensor_names, app=None, dt=None, color=None):
        # state
        self._speed = 1.0
        self._lane_offset = 2.0
        self._straight_dist = 0.0
        # road properties
        self._lane_width = 4.0
        self._num_lanes = 2

        self.color = color or 'red'
        self._write_car_material_file(self.color)

        state_space = spaces.BoxSpace([0.0, 0.5 * self._lane_width, -np.inf],
                                      [10.0, (self._num_lanes - 0.5) * self._lane_width, np.inf])
        super(CarOgreEnv, self).__init__(action_space, observation_space, state_space, sensor_names, app=app, dt=dt)
        self.city_env = CityOgreEnv(app=self.app)
        self.city_node = self.city_env.city_node
        self.skybox_node = self.city_env.skybox_node

        car_entity = self.app.scene_manager.createEntity('camaro2_3ds.mesh')
        self.car_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode('car')
        car_local_node = self.car_node.createChildSceneNode()
        car_local_node.attachObject(car_entity)
        car_local_node.setScale(np.array([0.3] * 3))
        car_local_node.setPosition(np.array([0, 0, 1.]))
        car_local_node.setOrientation(
            tf.quaternion_multiply(tf.quaternion_about_axis(np.pi / 2, np.array([1, 0, 0])),
                                   tf.quaternion_about_axis(np.pi, np.array([0, 0, 1]))))

        self.car_camera_node = self.car_node.createChildSceneNode('car_camera')
        car_camera = self.app.scene_manager.createCamera('car_camera')
        car_camera.setNearClipDistance(self.app.camera.getNearClipDistance())
        car_camera.setFarClipDistance(self.app.camera.getFarClipDistance())
        self.car_camera_node.attachObject(car_camera)
        # TODO: put camera in front of car
        self.car_camera_node.setPosition(np.array([0, -4., 3.]) * 4)
        self.car_camera_node.setOrientation(tf.quaternion_about_axis(np.pi / 3, np.array([1, 0, 0])))

        if 'image' in self.sensor_names:
            self.car_camera_sensor = ogre.PyCameraSensor(car_camera, 640, 480)
        if 'depth_image' in self.sensor_names:
            self.car_depth_camera_sensor = ogre.PyDepthCameraSensor(car_camera, 640, 480)

        self._first_render = True

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, speed):
        self._speed = np.clip(speed, self.state_space.low[0], self.state_space.high[0])

    @property
    def lane_offset(self):
        return self._lane_offset

    @lane_offset.setter
    def lane_offset(self, lane_offset):
        self._lane_offset = np.clip(lane_offset, self.state_space.low[1], self.state_space.high[1])

    @property
    def straight_dist(self):
        return self._straight_dist

    @straight_dist.setter
    def straight_dist(self, straight_dist):
        self._straight_dist = np.clip(straight_dist, self.state_space.low[2], self.state_space.high[2])

    def observe(self):
        obs = []
        for sensor_name in self.sensor_names:
            if sensor_name == 'image':
                observation = self.car_camera_sensor.observe()
            elif sensor_name == 'depth_image':
                observation = self.car_depth_camera_sensor.observe()
            else:
                raise ValueError('Unknown sensor name %s' % sensor_name)
            obs.append(observation)
        return tuple(obs)

    def render(self):
        target_node = self.car_node
        if self._first_render:
            self.app.camera.setAutoTracking(True, target_node)
            self.app.camera.setFixedYawAxis(True, np.array([0., 0., 1.]))
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_T = tf.pose_matrix(target_node._getDerivedOrientation(), target_node._getDerivedPosition())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -4., 3.]) * 4)
        self.app.camera.setPosition((1 - tightness) * self.app.camera.getPosition() + tightness * target_camera_pos)
        self.app.root.renderOneFrame()
        self.app.window.update()

    def _write_car_material_file(self, color_name, template_fname=None, new_fname=None):
        """
        TODO: this hard-coded way to generate a material file for the car
        """
        if color_name == 'red':
            ambient_material21 = '0.427451 0.0117647 0.0117647'
            diffuse_material21 = '0.843137 0.027451 0.027451'
            ambient_material23 = '0.498039 0.0509804 0.0509804'
            diffuse_material23 = '1 0.105882 0.105882'
        elif color_name == 'green':
            ambient_material21 = '0.0117647 0.127451 0.0117647'
            diffuse_material21 = '0.027451 0.543137 0.027451'
            ambient_material23 = '0.0509804 0.298039 0.0509804'
            diffuse_material23 = '0.105882 0.4 0.105882'
        elif color_name == 'random':
            ambient_material21 = '%.6f %.6f %.6f' % tuple(np.random.random(3))
            diffuse_material21 = '%.6f %.6f %.6f' % tuple(np.random.random(3))
            ambient_material23 = '%.6f %.6f %.6f' % tuple(np.random.random(3))
            diffuse_material23 = '%.6f %.6f %.6f' % tuple(np.random.random(3))
        else:
            raise ValueError('unknown color %s' % color_name)
        replace = dict(ambient_material21=ambient_material21,
                       diffuse_material21=diffuse_material21,
                       ambient_material23=ambient_material23,
                       diffuse_material23=diffuse_material23)
        template_fname = template_fname or os.path.expanduser('~/rll/python-ogre-lite/media/converted/camaro2_3ds_template.material')
        new_fname = new_fname or os.path.expanduser('~/rll/python-ogre-lite/media/converted/camaro2_3ds.material')
        with open(template_fname, 'r') as template_file:
            with open(new_fname, 'w') as new_file:
                for line in template_file:
                    for old_string, new_string in replace.items():
                        old_string = '%%{%s}' % old_string
                        if old_string in line:
                            line = line.replace(old_string, new_string)
                    new_file.write(line)

    def _get_config(self):
        config = super(CarOgreEnv, self)._get_config()
        config.update({'color': self.color})
        config.pop('state_space')
        return config


class StraightCarOgreEnv(CarOgreEnv):
    def __init__(self, action_space, observation_space, sensor_names, app=None, dt=None, color=None):
        # minimum and maximum position of the car
        # [-51 - 6, -275, 10.7]
        # [-51 + 6, 225, 10.7]
        super(StraightCarOgreEnv, self).__init__(action_space, observation_space, sensor_names, app=app, dt=dt, color=color)
        # modify the straight_dist limits
        self.state_space.low[2] = 0
        self.state_space.high[2] = 275 + 225

    @property
    def position(self):
        return np.array([-51 + self.lane_offset, -275 + self.straight_dist, 10.7])

    def step(self, action):
        self.city_env.step(action)
        forward_acceleration, lateral_velocity = action
        self.speed += forward_acceleration * self.dt
        self.lane_offset += lateral_velocity * self.dt
        self.straight_dist += self.speed * self.dt
        self.car_node.setPosition(self.position)

    def get_state(self):
        return np.array([self.speed, self.lane_offset, self.straight_dist])

    def reset(self, state=None):
        self.city_env.reset()
        if state is None:
            state = self.state_space.sample()
        self.speed, self.lane_offset, self.straight_dist = state
        self.car_node.setPosition(self.position)


class SimpleGeometricCarOgreEnv(CarOgreEnv):
    def __init__(self, action_space, observation_space, sensor_names, app=None, dt=None, graph_collada_fname=None, faces_collada_fname=None, color=None):
        # TODO: find file in path
        self._graph_collada_fname = graph_collada_fname or os.path.expanduser('~/rll/python-ogre-lite/media/converted/_urban-level-02-medium-road-directed-graph.dae')
        self._faces_collada_fname = faces_collada_fname or os.path.expanduser('~/rll/python-ogre-lite/media/converted/_urban-level-02-medium-road-faces.dae')
        self._edge_normals = {}
        self._last_edge = self._edge_normal = None
        self._build_graph()
        # for sampling turns
        self.angle_variance = np.pi / 2.0
        self.angle_thresh = np.pi * 3.0 / 4.0
        # state
        self._start_ind, self._end_ind = self.sample_vertex_inds()
        CarOgreEnv.__init__(self, action_space, observation_space, sensor_names, app=app, dt=dt, color=color)
        self.car_node.setTransform(self.transform)

    @property
    def start_pos(self):
        return self._points[self._start_ind]

    @property
    def end_pos(self):
        return self._points[self._end_ind]

    @property
    def edge_normal(self):
        if (self._start_ind, self._end_ind) != self._last_edge:
            self._last_edge = (self._start_ind, self._end_ind)
            self._edge_normal = self._get_or_compute_edge_normal((self._start_ind, self._end_ind))
        return self._edge_normal

    @property
    def max_straight_dist(self):
        return np.linalg.norm(self.end_pos - self.start_pos)

    def _build_graph(self):
        col = collada.Collada(self._graph_collada_fname)
        geom = col.geometries[0]
        lineset = geom.primitives[0]
        self._graph = defaultdict(set)
        for node1, node2 in lineset.vertex_index:
            self._graph[node1].add(node2)
        self._points = lineset.vertex

    def _get_or_compute_edge_normal(self, edge_inds):
        edge_normal = self._edge_normals.get(edge_inds)
        if edge_normal is None:
            col = collada.Collada(self._faces_collada_fname)
            geom = col.geometries[0]
            triset = geom.primitives[0]
            start_ind, end_ind = edge_inds
            start_point, end_point = self._points[start_ind], self._points[end_ind]
            # alternative implementation that loops over triangles and is ~100x slower
            # # iterate over triangles and check if the start and end points belong to each triangle
            # for tri_inds in triset.vertex_index:
            #     start_tri_ind = -1
            #     end_tri_ind = -1
            #     tri_points = triset.vertex[tri_inds]
            #     for tri_ind, tri_point in enumerate(tri_points):
            #         if np.allclose(tri_point, start_point):
            #             start_tri_ind = tri_ind
            #         if np.allclose(tri_point, end_point):
            #             end_tri_ind = tri_ind
            #     if start_tri_ind == -1 or end_tri_ind == -1:
            #         continue
            #     # check if start ind comes after end ind in the triangle
            #     if (start_tri_ind % 3) == ((end_tri_ind + 1) % 3):
            #         edge_normal = np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
            #         edge_normal /= np.linalg.norm(edge_normal)
            #         break
            start_tri_inds = np.where(np.isclose(triset.vertex[triset.vertex_index.flatten()], start_point).all(axis=1))[0]
            end_tri_inds = np.where(np.isclose(triset.vertex[triset.vertex_index.flatten()], end_point).all(axis=1))[0]
            for start_tri_ind in start_tri_inds:
                for end_tri_ind in end_tri_inds:
                    same_triangle = start_tri_ind // 3 == end_tri_ind // 3
                    if same_triangle and (start_tri_ind % 3) == ((end_tri_ind + 1) % 3):
                        tri_inds = triset.vertex_index[start_tri_ind // 3]
                        tri_points = triset.vertex[tri_inds]
                        edge_normal = np.cross(tri_points[1] - tri_points[0], tri_points[2] - tri_points[0])
                        edge_normal /= np.linalg.norm(edge_normal)
                        break
            self._edge_normals[edge_inds] = edge_normal
        return edge_normal

    def _next_ind(self, ind0, ind1):
        dir = self._points[ind1] - self._points[ind0]
        next_inds = list(self._graph[ind1])
        if len(next_inds) == 1:
            next_ind, = next_inds
        else:
            angle_changes = []
            for next_ind in next_inds:
                next_dir = self._points[next_ind] - self._points[ind1]
                angle_changes.append(tf.angle_between_vectors(dir, next_dir))
            angle_changes = np.asarray(angle_changes)
            probs = np.exp(-(angle_changes ** 2) / (2. * self.angle_variance))
            probs[np.abs(angle_changes) > self.angle_thresh] = 0.0
            probs /= probs.sum()
            next_ind = np.random.choice(next_inds, p=probs)
        return next_ind

    def _compute_rotation(self, rot_y, rot_z=None):
        rot_y /= np.linalg.norm(rot_y)
        if rot_z is None:
            up_v = np.array([0., 0., 1.])
            rot_x = np.cross(rot_y, up_v)
            rot_z = np.cross(rot_x, rot_y)
        else:
            rot_z /= np.linalg.norm(rot_z)
            rot_x = np.cross(rot_y, rot_z)
        rot = np.array([rot_x, rot_y, rot_z]).T
        return rot

    @property
    def transform(self):
        start_T = np.eye(4)
        start_T[:3, :3] = self._compute_rotation(self.end_pos - self.start_pos, self.edge_normal)
        start_T[:3, 3] = self.start_pos
        translate_to_lane_T = tf.translation_matrix(np.array([self._lane_offset, self._straight_dist, 0.]))
        T = start_T.dot(translate_to_lane_T)
        return T

    def step(self, action):
        self.city_env.step(action)
        forward_acceleration, lateral_velocity = action
        self.speed += forward_acceleration * self.dt
        self.lane_offset += lateral_velocity * self.dt

        delta_dist = self.speed * self.dt
        while delta_dist > 0.0:
            remaining_dist = self.max_straight_dist - self._straight_dist
            if delta_dist < remaining_dist:
                self._straight_dist += delta_dist
                delta_dist = 0.0
            else:
                delta_dist -= remaining_dist
                self._straight_dist = 0.0
                # advance to next segment
                self._start_ind, self._end_ind = \
                    self._end_ind, self._next_ind(self._start_ind, self._end_ind)
        self.car_node.setTransform(self.transform)

    def get_state(self):
        return np.array([self.speed, self.lane_offset, self.straight_dist, self._start_ind, self._end_ind])

    def reset(self, state=None):
        self.city_env.reset()
        if state is None:
            # TODO: use state space sample?
            speed = 1.0
            lane_offset = 2.0
            straight_dist = 0.0  # distance along current edge
            state = (speed, lane_offset, straight_dist) + self.sample_vertex_inds()
        self.speed, self.lane_offset, self.straight_dist, self._start_ind, self._end_ind = state
        self.car_node.setTransform(self.transform)

    def sample_vertex_inds(self, start_ind=None):
        start_ind = start_ind or np.random.choice(list(self._graph.keys()))
        end_ind = np.random.choice(list(self._graph[start_ind]))
        return start_ind, end_ind

    def observe(self):
        obs = []
        for sensor_name in self.sensor_names:
            if sensor_name == 'image':
                observation = self.car_camera_sensor.observe()
            elif sensor_name == 'depth_image':
                observation = self.car_depth_camera_sensor.observe()
            else:
                raise ValueError('Unknown sensor name %s' % sensor_name)
            obs.append(observation)
        return tuple(obs)

    def render(self):
        target_node = self.car_node
        if self._first_render:
            self.app.camera.setAutoTracking(True, target_node)
            self.app.camera.setFixedYawAxis(True, np.array([0., 0., 1.]))
            tightness = 1.0
            self._first_render = False
        else:
            tightness = 0.1
        target_T = tf.pose_matrix(target_node._getDerivedOrientation(), target_node._getDerivedPosition())
        target_camera_pos = target_T[:3, 3] + target_T[:3, :3].dot(np.array([0., -4., 3.]) * 4)
        self.app.camera.setPosition((1 - tightness) * self.app.camera.getPosition() + tightness * target_camera_pos)
        self.app.root.renderOneFrame()
        self.app.window.update()

    def _get_config(self):
        config = super(SimpleGeometricCarOgreEnv, self)._get_config()
        config.update({'graph_collada_fname': self._graph_collada_fname})
        config.update({'faces_collada_fname': self._faces_collada_fname})
        return config


class GeometricCarOgreEnv(SimpleGeometricCarOgreEnv):
    def __init__(self, action_space, observation_space, sensor_names, app=None, dt=None, graph_collada_fname=None, faces_collada_fname=None, color=None):
        # TODO: find file in path
        self._graph_collada_fname = graph_collada_fname or os.path.expanduser('~/rll/python-ogre-lite/media/converted/_urban-level-02-medium-road-directed-graph.dae')
        self._faces_collada_fname = faces_collada_fname or os.path.expanduser('~/rll/python-ogre-lite/media/converted/_urban-level-02-medium-road-faces.dae')
        self._edge_normals = {}
        self._last_edge = self._edge_normal = None
        self._build_graph()
        # for sampling turns
        self.angle_variance = np.pi / 2.0
        self.angle_thresh = np.pi * 3.0 / 4.0
        # state
        self._turn_angle = None  # angle along current curve (defined by two adjacent edges)
        self._start_ind, self._middle_ind, self._end_ind = self.sample_vertex_inds()  # SimpleCarOgreEnv's start_ind and end_ind are CarOgreEnv's start_ind and middle_ind
        CarOgreEnv.__init__(self, action_space, observation_space, sensor_names, app=app, dt=dt, color=color)
        self.car_node.setTransform(self.transform)

    @property
    def middle_pos(self):
        return self._points[self._middle_ind]

    @property
    def start_rot(self):
        pt0 = self._points[self._start_ind]
        pt1 = self._points[self._middle_ind]
        return self._compute_rotation(pt1 - pt0, self._get_or_compute_edge_normal((self._start_ind, self._middle_ind)))

    @property
    def middle_rot(self):
        pt0 = self._points[self._start_ind]
        pt1 = self._points[self._middle_ind]
        pt2 = self._points[self._end_ind]
        up_v = np.array([0., 0., 1.])
        rot_z = np.cross(pt1 - pt0, pt2 - pt1)
        if rot_z.dot(up_v) < 0:
            rot_z *= -1.0
        return self._compute_rotation(pt2 - pt1, self._get_or_compute_edge_normal((self._middle_ind, self._end_ind)))

    @property
    def end_rot(self):
        pt1 = self._points[self._middle_ind]
        pt2 = self._points[self._end_ind]
        return self._compute_rotation(pt2 - pt1, self._get_or_compute_edge_normal((self._middle_ind, self._end_ind)))

    @property
    def start_T(self):
        start_T = np.eye(4)
        start_T[:3, :3] = self.start_rot
        start_T[:3, 3] = self.start_pos
        return start_T

    @property
    def middle_T(self):
        middle_T = np.eye(4)
        middle_T[:3, :3] = self.middle_rot
        middle_T[:3, 3] = self.middle_pos
        return middle_T

    @property
    def end_T(self):
        end_T = np.eye(4)
        end_T[:3, :3] = self.end_rot
        end_T[:3, 3] = self.end_pos
        return end_T

    @property
    def max_straight_dist(self):
        return np.linalg.norm(self.middle_pos - self.start_pos)

    @property
    def max_turn_angle(self):
        angle = tf.angle_between_vectors(self.middle_pos - self.start_pos, self.end_pos - self.middle_pos)
        assert 0 <= angle <= np.pi
        return angle

    @property
    def turn_radius(self):
        return self._num_lanes * self._lane_width + self.left_turn * self._lane_offset

    @property
    def left_turn(self):
        collinear = (np.cross(self.middle_pos - self.start_pos, self.end_pos - self.middle_pos) == 0).all()
        if collinear:
            left_turn = 0.0
        else:
            left_turn = np.sign(
                np.cross(self.middle_pos - self.start_pos, self.end_pos - self.middle_pos).dot(self.middle_rot[:, 2]))
            assert left_turn != 0  # TODO: if max_turn_angle is pi, then do left turn
        return left_turn

    @property
    def turn_dist_offset(self):
        """
        Distance from the end of the current edge where the curve starts, which
        is the same the distance from the start of the next edge where the
        curve ends.
        """
        if self.max_turn_angle == np.pi:  # U-turn
            turn_dist_offset = 0.0
        else:
            turn_dist_offset = (self._num_lanes * self._lane_width) / np.tan((np.pi - self.max_turn_angle) / 2)
        return turn_dist_offset

    @property
    def start_turn_angle_offset(self):
        """
        Angle from the start of the curve where the current edge starts.
        """
        return max(0.0, np.arctan2(self.turn_dist_offset - self.max_straight_dist,
                                   self._num_lanes * self._lane_width))

    @property
    def end_turn_angle_offset(self):
        """
        Angle from the end of the curve where the next edge ends.
        """
        return max(0.0, np.arctan2(self.turn_dist_offset - np.linalg.norm(self.end_pos - self.middle_pos),
                                   self._num_lanes * self._lane_width))

    @property
    def transform(self):
        assert (self._straight_dist is None) != (self._turn_angle is None)
        if self._straight_dist is not None:
            translate_to_lane_T = tf.translation_matrix(np.array([self._lane_offset, self._straight_dist, 0.]))
            T = self.start_T.dot(translate_to_lane_T)
        else:  # self._turn_angle is not None
            middle_T = self.middle_T
            left_turn = self.left_turn
            translate_to_center_T = tf.translation_matrix(
                np.array([-left_turn * self._num_lanes * self._lane_width,
                          self.turn_dist_offset,
                          0.]))
            rotate_about_center_T = tf.rotation_matrix(
                left_turn * (self._turn_angle - self.max_turn_angle), middle_T[:3, 2])
            translate_to_lane_T = tf.translation_matrix(
                np.array([left_turn * self._num_lanes * self._lane_width + self._lane_offset, 0., 0.]))
            T = middle_T.dot(translate_to_center_T.dot(rotate_about_center_T.dot(translate_to_lane_T)))
        return T

    def step(self, action):
        self.city_env.step(action)
        forward_acceleration, lateral_velocity = action
        self.speed += forward_acceleration * self.dt
        self.lane_offset += lateral_velocity * self.dt

        delta_dist = self.speed * self.dt
        while delta_dist > 0.0:
            assert (self._straight_dist is None) != (self._turn_angle is None)
            if self._straight_dist is not None:
                remaining_dist = self.max_straight_dist - self._straight_dist - self.turn_dist_offset
                if delta_dist < remaining_dist:
                    self._straight_dist += delta_dist
                    delta_dist = 0.0
                else:
                    delta_dist -= remaining_dist
                    self._straight_dist = None
                    self._turn_angle = self.start_turn_angle_offset
            else:  # self._turn_angle is not None
                remaining_dist = (
                                 self.max_turn_angle - self._turn_angle) * self.turn_radius - self.end_turn_angle_offset
                if delta_dist < remaining_dist:
                    self._turn_angle += delta_dist / self.turn_radius
                    delta_dist = 0.0
                else:
                    delta_dist -= remaining_dist
                    self._turn_angle = None
                    self._straight_dist = self.turn_dist_offset
                    # advance to next segment
                    self._start_ind = self._middle_ind
                    self._middle_ind = self._end_ind
                    self._end_ind = self._next_ind(self._start_ind, self._middle_ind)
        self.car_node.setTransform(self.transform)

    def get_state(self):
        # convert None to -1 so that the state is a numeric array (as opposed to object array)
        straight_dist = self._straight_dist if self._straight_dist is not None else -1
        turn_angle = self._turn_angle if self._turn_angle is not None else -1
        return np.array([self.speed, self.lane_offset, straight_dist, turn_angle,
                         self._start_ind, self._middle_ind, self._end_ind])

    def reset(self, state=None):
        self.city_env.reset()
        if state is None:
            # TODO: use state space sample?
            speed = 1.0
            lane_offset = 2.0
            straight_dist = 0.0  # distance along current edge
            turn_angle = None  # angle along current curve (defined by two adjacent edges)
            vertex_inds = self.sample_vertex_inds()
        else:
            speed, lane_offset, straight_dist, turn_angle = state[:4]
            vertex_inds = state[4:]
        # convert -1 to None
        if straight_dist == -1:
            straight_dist = None
        if turn_angle == -1:
            turn_angle = None
        vertex_inds = [int(ind) for ind in vertex_inds]
        self.speed = speed
        self.lane_offset = lane_offset
        self._straight_dist = straight_dist  # set self._straight_dist directly to prevent clipping of None
        self._turn_angle = turn_angle
        self._start_ind, self._middle_ind, self._end_ind = vertex_inds
        self.car_node.setTransform(self.transform)

    def sample_vertex_inds(self, start_ind=None, middle_ind=None):
        if start_ind is None:
            start_ind = np.random.choice(list(self._graph.keys()))
        if middle_ind is None:
            middle_ind = np.random.choice(list(self._graph[start_ind]))
        elif not middle_ind in self._graph[start_ind]:
            raise ValueError("Invalid start_ind %d and end_ind %d" % (start_ind, middle_ind))
        end_ind = self._next_ind(start_ind, middle_ind)
        return start_ind, middle_ind, end_ind
