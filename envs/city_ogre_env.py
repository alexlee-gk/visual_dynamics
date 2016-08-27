import numpy as np
from envs import OgreEnv
import utils.transformations as tf


class CityOgreEnv(OgreEnv):
    def __init__(self, app=None):
        super(CityOgreEnv, self).__init__(None, None, None, None, app=app)

        light = self.app.scene_manager.createLight('sun')
        light.setPosition(np.array([-2506., -634., 2596.]))

        city_entity = self.app.scene_manager.createEntity('_urban-level-02-medium-3ds_3DS.mesh')
        self.city_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode()
        self.city_node.attachObject(city_entity)
        self.city_node.setOrientation(tf.quaternion_about_axis(np.pi / 2, np.array([1, 0, 0])))

        skybox_entity = self.app.scene_manager.createEntity('skybox-mesh_3DS.mesh')
        self.skybox_node = self.app.scene_manager.getRootSceneNode().createChildSceneNode()
        self.skybox_node.attachObject(skybox_entity)
        self.skybox_node.setOrientation(tf.quaternion_about_axis(np.pi / 2, np.array([1, 0, 0])))

    def step(self, action):
        pass

    def reset(self, state=None):
        pass
