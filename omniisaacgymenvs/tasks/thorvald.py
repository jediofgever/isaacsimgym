# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.thorvald import Thorvald

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.articulations import ArticulationView

import numpy as np
import torch
from math import pow, sqrt, cos, sin, atan2, fabs, fmod
import math

class ThorvaldController(BaseController):
    
    def __init__(self):
        super().__init__(name="thorvald_controller")
        # An open loop controller that uses a unicycle model
        self._wheel_radius = 0.2
        self._robot_length = -0.67792
        self._y_offset = -0.75
        self._wheel_base = 0.1125
        return

    def normalize_angle(self, ang):
        result = fmod(ang + math.pi, 2.0*math.pi)
        if(result <= 0.0): 
            return result + math.pi
        return result - math.pi
  
    def forward(self, command):
        # command will have two elements, first element is the forward velocity
        # second element is the angular velocity (yaw only).
        steering = [0.0, 0.0, 0.0, 0.0]
        speed = [0.0, 0.0, 0.0, 0.0]

        vx = command[0]
        vy = command[1]
        wz = command[2]

        if (wz != 0.0):
            turn_rad_d = sqrt(pow(vx,2) + pow(vy,2)) / wz
            turn_rad_ang = atan2(vy,vx)
            turn_rad_x = - turn_rad_d * sin(turn_rad_ang)
            turn_rad_y = turn_rad_d * cos(turn_rad_ang)

            drive_x = -0.67
            drive_y = -0.75
            steering[0] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[0] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)

            drive_x = 0.67
            drive_y = -0.75
            steering[1] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[1] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)

            drive_x = 0.67
            drive_y = 0.75
            steering[2] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[2] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)

            drive_x = -0.67
            drive_y = 0.75
            steering[3] = self.normalize_angle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + math.pi * (wz < 0))
            speed[3] = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz)
        else:
            for i in range(0,4):            
                steering[i] = atan2(vy,vx)  
                speed[i] = sqrt(pow(vx,2) + pow(vy,2)) / self._wheel_radius


        joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        joint_positions[0:4] = steering
        joint_velocities[4:8] = speed

        # A controller has to return an ArticulationAction
        #return ArticulationAction(joint_positions=joint_positions, joint_velocities=joint_velocities)   
        return joint_positions, joint_velocities   

class ThorvaldTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]
        self._num_observations = 60
        self._num_actions = 8
        
        self._wheel_dof_names = [ "steering0", "steering1", "steering2", "steering3","wheel0", "wheel1", "wheel2", "wheel3"]
        self._wheel_dof_indices = wheel_dof_indices = [0,1,2,3,4,5,6,7]
        self._num_wheel_dof = len(self._wheel_dof_indices)
        
        self._my_controller = ThorvaldController()

        self._thorvald_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name=name, env=env)

        self._commands = []

        for i in range(0, self.num_envs):
            self._commands.append( [random.uniform(1.0, 2.5), 0.0, random.uniform(-1.5, 1.5)])

        return

    def set_up_scene(self, scene) -> None:
        self.get_thorvald()
        RLTask.set_up_scene(self, scene)
        self._thorvalds = ArticulationView(prim_paths_expr="/World/envs/.*/Thorvald", name="thorvald_view", reset_xform_properties=False)


        scene.add(self._thorvalds)
        return

    def get_thorvald(self):

        thorvald = Thorvald(prim_path=self.default_zero_env_path + "/Thorvald", name="my_thorvald",
                wheel_dof_names=[ "steering0", "steering1", "steering2", "steering3","wheel0", "wheel1", "wheel2", "wheel3"],
                wheel_dof_indices=[0,1,2,3,4,5,6,7],
                create_robot=True,
                position=self._thorvald_position)

        self._sim_config.apply_articulation_settings("Thorvald", get_prim_at_path(thorvald.prim_path), self._sim_config.parse_actor_config("Thorvald"))

    def get_observations(self) -> dict:
        poses = self._thorvalds.get_joint_positions()
        observations = {
            self._thorvalds.name: {
                "obs_buf": poses
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:

        all_positions = self._thorvalds.get_joint_positions()
        joint_positions = torch.zeros([all_positions.shape[0], all_positions.shape[1]], dtype=torch.float32, device=self._device)
        joint_velocities = torch.zeros([all_positions.shape[0], all_positions.shape[1]], dtype=torch.float32, device=self._device)

        for i in range(0, all_positions.shape[0]):
            pos, velo = self._my_controller.forward(command=self._commands[i])
            joint_positions[i] = torch.tensor(pos,dtype=torch.float32, device=self._device)
            joint_velocities[i] = torch.tensor(velo, dtype=torch.float32, device=self._device)
        

        indices = torch.arange(self._thorvalds.count, dtype=torch.int32, device=self._device)
        self._thorvalds.apply_action(control_actions=ArticulationAction(joint_positions=joint_positions, 
                                                                        joint_velocities=joint_velocities,
                                                                        joint_indices=[0,1,2,3,4,5,6,7]), indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF positions
        pass

    def post_reset(self):
        self._thorvalds.post_reset()
        self._thorvalds.switch_control_mode("position", joint_indices=[0,1,2,3])
        self._thorvalds.switch_control_mode("velocity", joint_indices=[4,5,6,7])


    def calculate_metrics(self) -> None:
        pass

    def is_done(self) -> None:
        pass

