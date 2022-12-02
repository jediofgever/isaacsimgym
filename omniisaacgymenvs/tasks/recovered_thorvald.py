from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
import numpy as np
from math import pow, sqrt, cos, sin, atan2, floor, fabs, fmod
import math

from .thorvald_controller import ThorvaldController

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        jetbot_asset_path =  "/home/atas/Documents/thorvald_sensored.usd"
        world.scene.add(
            WheeledRobot(
                prim_path="/World/Thorvald",
                name="my_thorvald",
                wheel_dof_names=[ "steering0", "steering1", "steering2", "steering3","wheel0", "wheel1", "wheel2", "wheel3"],
                wheel_dof_indices=[0,1,2,3,4,5,6,7],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, -2.0, 1.0])
            )
        )
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("my_thorvald")
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        # Initialize our controller after load and the first reset
        self._my_controller = ThorvaldController()
        return

    def send_robot_actions(self, step_size = 0.001):
        #apply the actions calculated by the controller
        self._jetbot.apply_action(self._my_controller.forward(command=[1.0, 0.0, 1.0], curr_wheel_positions=self._jetbot.get_wheel_positions(), step_size=step_size))
        return