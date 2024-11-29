# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import torch

import omni.usd

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectMARLEnv
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.math import apply_delta_pose
from omni.isaac.lab.utils.math import *

from .mi_arm_hand_over_camera_env_cfg import MiArmHandOverRGBCameraEnvCfg
from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
from .pose_predictor import PosePredictor, PosePredictorCfg

# import omni.replicator.core as rep
# rep.settings.set_render_rtx_realtime(antialiasing="FXAA")

class MiArmHandOverRGBCameraEnv(DirectMARLEnv):
    cfg: MiArmHandOverRGBCameraEnvCfg

    def __init__(self, cfg: MiArmHandOverRGBCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_right_hand_dofs = self.right_hand.num_joints
        self.num_left_hand_dofs = self.left_hand.num_joints

        # buffers for position targets
        self.right_robot_prev_targets = torch.zeros(
            (self.num_envs, self.num_right_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_robot_curr_targets = torch.zeros(
            (self.num_envs, self.num_right_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_robot_prev_targets = torch.zeros(
            (self.num_envs, self.num_left_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_robot_curr_targets = torch.zeros(
            (self.num_envs, self.num_left_hand_dofs), dtype=torch.float, device=self.device
        )
        self.next_right_ee_pose_b = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.next_left_ee_pose_b = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)

        self.last_embeddings = torch.zeros((self.num_envs, 64), dtype=torch.float, device=self.device)
        self.last_right_actions = torch.zeros((self.num_envs, 9), dtype=torch.float, device=self.device)

        # list of hand actuated joints
        self.right_hand_actuated_dof_indices = list()
        for joint_name in cfg.right_hand_actuated_joint_names:
            self.right_hand_actuated_dof_indices.append(self.right_hand.joint_names.index(joint_name))
        self.right_hand_actuated_dof_indices.sort()

        self.left_hand_actuated_dof_indices = list()
        for joint_name in cfg.left_hand_actuated_joint_names:
            self.left_hand_actuated_dof_indices.append(self.left_hand.joint_names.index(joint_name))
        self.left_hand_actuated_dof_indices.sort()

        # list of arm actuated joints
        self.right_arm_actuated_dof_indices = list()
        for joint_name in cfg.right_arm_actuated_joint_names:
            self.right_arm_actuated_dof_indices.append(self.right_hand.joint_names.index(joint_name))
        self.right_arm_actuated_dof_indices.sort()

        self.left_arm_actuated_dof_indices = list()
        for joint_name in cfg.left_arm_actuated_joint_names:
            self.left_arm_actuated_dof_indices.append(self.left_hand.joint_names.index(joint_name))
        self.left_arm_actuated_dof_indices.sort()

        # finger bodies
        self.right_finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.right_finger_bodies.append(self.right_hand.body_names.index(body_name))
        self.right_finger_bodies.sort()
        self.num_fingertips = len(self.right_finger_bodies)

        self.left_finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.left_finger_bodies.append(self.left_hand.body_names.index(body_name))
        self.left_finger_bodies.sort()
        self.num_fingertips = len(self.left_finger_bodies)

        # joint limits
        right_joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.right_robot_dof_lower_limits = right_joint_pos_limits[..., 0]
        self.right_robot_dof_upper_limits = right_joint_pos_limits[..., 1]

        left_joint_pos_limits = self.left_hand.root_physx_view.get_dof_limits().to(self.device)
        self.left_robot_dof_lower_limits = left_joint_pos_limits[..., 0]
        self.left_robot_dof_upper_limits = left_joint_pos_limits[..., 1]

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # ik controller setup
        self.right_diff_ik_controller = DifferentialIKController(self.cfg.diff_ik_controller, num_envs=self.num_envs, device=self.device)
        self.right_ee_id = self.right_hand.body_names.index(self.cfg.right_ee_body_name)

        self.left_diff_ik_controller = DifferentialIKController(self.cfg.diff_ik_controller, num_envs=self.num_envs, device=self.device)
        self.left_ee_id = self.left_hand.body_names.index(self.cfg.left_ee_body_name)

        if self.right_hand.is_fixed_base:
            self.right_ee_jacobi_idx = self.right_ee_id - 1
        else:
            self.right_ee_jacobi_idx = self.right_ee_id

        if self.left_hand.is_fixed_base:
            self.left_ee_jacobi_idx = self.left_ee_id - 1
        else:
            self.left_ee_jacobi_idx = self.left_ee_id

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.left_robot_dof_speed_scales = torch.ones_like(self.left_robot_dof_lower_limits)
        self.right_robot_dof_speed_scales = torch.ones_like(self.right_robot_dof_lower_limits)

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.46, 0.22, 0.13], device=self.device)
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        # self.right_ee_frame_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
        # self.left_ee_frame_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
        # self.goal_frame_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        self.feature_extractor = FeatureExtractor(self.cfg.feature_extractor, self.device)
        self.pose_predictor = PosePredictor(self.cfg.pose_predictor, self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.right_hand = Articulation(self.cfg.right_robot_cfg)
        self.left_hand = Articulation(self.cfg.left_robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # get stage
        stage = omni.usd.get_context().get_stage()
        # add semantics for in-hand cube
        prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("ball")
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.2))
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["right_robot"] = self.right_hand
        self.scene.articulations["left_robot"] = self.left_hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, (0.0, 0.0, 1.0))
        # add tiled camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add sensors to scene
        self.scene.sensors["tiled_camera"] = self._tiled_camera

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        # # right arm target
        # right_delta_pose = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        # right_delta_pose[:, 3:6] = self.actions["right_hand"][:, 0:3] * 0.02
        # _, self.goal_right_ee_quat_b = apply_delta_pose(self.right_ee_pos_b, self.right_ee_quat_b, right_delta_pose)

        # # left arm target
        # left_delta_pose = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        # left_delta_pose[:, 3:6] = self.actions["left_hand"][:, 0:3] * 0.02
        # _, self.goal_left_ee_quat_b = apply_delta_pose(self.left_ee_pos_b, self.left_ee_quat_b, left_delta_pose)

        # # compute right ik
        # self.next_right_ee_pose_b = torch.cat((self.next_right_ee_pose_b[:, :3], self.goal_right_ee_quat_b), dim=-1)
        # self.right_diff_ik_controller.set_command(self.next_right_ee_pose_b, self.right_ee_pos_b, self.right_ee_quat_b)
        # self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices] = self.right_diff_ik_controller.compute(
        #     self.right_ee_pos_b, self.right_ee_quat_b, self.right_jacobian, self.right_arm_joint_pos)
        
        # self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices] = saturate(
        #     self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices],
        #     self.right_robot_dof_lower_limits[:, self.right_arm_actuated_dof_indices],
        #     self.right_robot_dof_upper_limits[:, self.right_arm_actuated_dof_indices],
        # )

        # # compute left ik
        # self.next_left_ee_pose_b = torch.cat((self.next_left_ee_pose_b[:, :3], self.goal_left_ee_quat_b), dim=-1)
        # self.left_diff_ik_controller.set_command(self.next_left_ee_pose_b, self.left_ee_pos_b, self.left_ee_quat_b)
        # self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices] = self.left_diff_ik_controller.compute(
        #     self.left_ee_pos_b, self.left_ee_quat_b, self.left_jacobian, self.left_arm_joint_pos)
        
        # self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices] = saturate(
        #     self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices],
        #     self.left_robot_dof_lower_limits[:, self.left_arm_actuated_dof_indices],
        #     self.left_robot_dof_upper_limits[:, self.left_arm_actuated_dof_indices],
        # )

        # # right arm target
        # self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]] = scale(
        #     self.actions["right_hand"][:, 0:3],
        #     self.right_robot_dof_lower_limits[:, self.right_arm_actuated_dof_indices[-3:]],
        #     self.right_robot_dof_upper_limits[:, self.right_arm_actuated_dof_indices[-3:]],
        # )
        # self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]] = (
        #     self.cfg.act_moving_average_arm * self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]]
        #     + (1.0 - self.cfg.act_moving_average_arm) * self.right_robot_prev_targets[:, self.right_arm_actuated_dof_indices[-3:]]
        # )
        # self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]] = saturate(
        #     self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]],
        #     self.right_robot_dof_lower_limits[:, self.right_arm_actuated_dof_indices[-3:]],
        #     self.right_robot_dof_upper_limits[:, self.right_arm_actuated_dof_indices[-3:]],
        # )

        # # left arm target
        # self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]] = scale(
        #     self.actions["left_hand"][:, 0:3],
        #     self.left_robot_dof_lower_limits[:, self.left_arm_actuated_dof_indices[-3:]],
        #     self.left_robot_dof_upper_limits[:, self.left_arm_actuated_dof_indices[-3:]],
        # )
        # self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]] = (
        #     self.cfg.act_moving_average_arm * self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]]
        #     + (1.0 - self.cfg.act_moving_average_arm) * self.left_robot_prev_targets[:, self.left_arm_actuated_dof_indices[-3:]]
        # )
        # self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]] = saturate(
        #     self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]],
        #     self.left_robot_dof_lower_limits[:, self.left_arm_actuated_dof_indices[-3:]],
        #     self.left_robot_dof_upper_limits[:, self.left_arm_actuated_dof_indices[-3:]],
        # )

        # right arm target
        self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]] = \
            self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]] + \
                self.dt * self.actions["right_hand"][:, :3].clamp(-1.0, 1.0) * self.cfg.action_scale

        self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]] = saturate(
            self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices[-3:]],
            self.right_robot_dof_lower_limits[:, self.right_arm_actuated_dof_indices[-3:]],
            self.right_robot_dof_upper_limits[:, self.right_arm_actuated_dof_indices[-3:]],
        )

        # left arm target
        self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]] = \
            self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]] + \
                self.dt * self.actions["left_hand"][:, :3].clamp(-1.0, 1.0) * self.cfg.action_scale

        self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]] = saturate(
            self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices[-3:]],
            self.left_robot_dof_lower_limits[:, self.left_arm_actuated_dof_indices[-3:]],
            self.left_robot_dof_upper_limits[:, self.left_arm_actuated_dof_indices[-3:]],
        )

        # right hand target
        self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices] = scale(
            self.actions["right_hand"][:, 3:9],
            self.right_robot_dof_lower_limits[:, self.right_hand_actuated_dof_indices],
            self.right_robot_dof_upper_limits[:, self.right_hand_actuated_dof_indices],
        )
        self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices] = (
            self.cfg.act_moving_average_hand * self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average_hand) * self.right_robot_prev_targets[:, self.right_hand_actuated_dof_indices]
        )
        self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices] = saturate(
            self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices],
            self.right_robot_dof_lower_limits[:, self.right_hand_actuated_dof_indices],
            self.right_robot_dof_upper_limits[:, self.right_hand_actuated_dof_indices],
        )

        # left hand target
        self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices] = scale(
            self.actions["left_hand"][:, 3:9],
            self.left_robot_dof_lower_limits[:, self.left_hand_actuated_dof_indices],
            self.left_robot_dof_upper_limits[:, self.left_hand_actuated_dof_indices],
        )
        self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices] = (
            self.cfg.act_moving_average_hand * self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average_hand) * self.left_robot_prev_targets[:, self.left_hand_actuated_dof_indices]
        )
        self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices] = saturate(
            self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices],
            self.left_robot_dof_lower_limits[:, self.left_hand_actuated_dof_indices],
            self.left_robot_dof_upper_limits[:, self.left_hand_actuated_dof_indices],
        )

        # set targets
        self.right_hand.set_joint_position_target(
            self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices], joint_ids=self.right_hand_actuated_dof_indices
        )
        self.left_hand.set_joint_position_target(
            self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices], joint_ids=self.left_hand_actuated_dof_indices
        )
        self.right_hand.set_joint_position_target(
            self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices], joint_ids=self.right_arm_actuated_dof_indices
        )
        self.left_hand.set_joint_position_target(
            self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices], joint_ids=self.left_arm_actuated_dof_indices
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "right_hand": torch.cat(
                (
                    # ---- right hand ----
                    # DOF positions (19)
                    unscale(self.right_hand_dof_pos, self.right_robot_dof_lower_limits, self.right_robot_dof_upper_limits),
                    # applied actions (9)
                    self.actions["right_hand"],
                    # ---- tiled camera data ----
                    self.embeddings,
                    self.goal_pos,
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (
                    # ---- left hand ----
                    # DOF positions (24)
                    unscale(self.left_hand_dof_pos, self.left_robot_dof_lower_limits, self.left_robot_dof_upper_limits),
                    # applied actions (9)
                    self.actions["left_hand"],
                    # ---- tiled camera data ----
                    self.embeddings,
                    self.goal_pos,
                ),
                dim=-1,
            ),
        }

        return observations

    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # ---- right hand ----
                # DOF positions (12)
                unscale(self.right_hand_dof_pos, self.right_robot_dof_lower_limits, self.right_robot_dof_upper_limits),
                # DOF velocities (12)
                self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                # fingertip positions (5 * 3)
                self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # fingertip rotations (5 * 4)
                self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # fingertip linear and angular velocities (5 * 6)
                self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # applied actions (20)
                self.actions["right_hand"],
                # ---- left hand ----
                # DOF positions (12)
                unscale(self.left_hand_dof_pos, self.left_robot_dof_lower_limits, self.left_robot_dof_upper_limits),
                # DOF velocities (12)
                self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                # fingertip positions (5 * 3)
                self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # fingertip rotations (5 * 4)
                self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # fingertip linear and angular velocities (5 * 6)
                self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # applied actions (20)
                self.actions["left_hand"],
                # ---- object ----
                # positions (3)
                self.object_pos,
                # rotations (4)
                self.object_rot,
                # linear velocities (3)
                self.object_linvel,
                # angular velocities (3)
                self.cfg.vel_obs_scale * self.object_angvel,
                # ---- goal ----
                # positions (3)
                self.goal_pos,
                self.goal_rot,
                self.left_ee_pos,
                # rotations (4)
                self.left_ee_quat,
                # goal-object rotation diff (4)
                quat_mul(self.object_rot, quat_conjugate(self.left_ee_quat)),

                self.embeddings,
            ),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # compute rewards
        goal_dist = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        next_goal_dist = torch.norm(self.predicted_next_object_pose - self.goal_pos, p=2, dim=-1)
        hand_object_dist = torch.norm(self.right_ee_pos - self.object_pos, p=2, dim=-1)

        rew_dist = 2 * torch.exp(-self.cfg.dist_reward_scale * goal_dist)
        rew_dist = torch.where(hand_object_dist < 0.05, torch.zeros_like(rew_dist), rew_dist)
        rew_next_dist = 0.2 * torch.exp(-self.cfg.dist_reward_scale * next_goal_dist)

        # compute penalties
        # hand_dist = torch.norm(self.right_ee_pos - self.left_ee_pos, p=2, dim=-1)
        # hand_dist_penalty = -0.00 * torch.exp(-2.0 * hand_dist)

        # hand_dist_penalty = torch.where(hand_object_dist < 0.1, torch.zeros_like(hand_object_dist), -0.1 * torch.exp(-2.0 * hand_object_dist))

        action_penalty_hand = self.action_scope_right_hand + self.action_scope_left_hand
        action_penalty_arm = self.action_scope_right_arm + self.action_scope_left_arm
        action_penalty = - 0.0001 * action_penalty_arm - 0.00001 * action_penalty_hand

        if self.cfg.is_training:
            # log reward components
            if "log" not in self.extras:
                self.extras["log"] = dict()
            self.extras["log"]["dist_goal"] = goal_dist.mean()
            self.extras["log"]["dist_reward"] = rew_dist.mean()

            self.extras["log"]["next_dist_reward"] = rew_next_dist.mean()
            self.extras["log"]["next_dist_goal"] = next_goal_dist.mean()
            
            # self.extras["log"]["hand_dist_penalty"] = hand_dist_penalty.mean()
            self.extras["log"]["action_penalty"] = action_penalty.mean()

            self.extras["log"]["feature_extractor_loss"] = self.feature_extractor_loss.mean()
            self.extras["log"]["pose_predictor_loss"] = self.pose_predictor_loss.mean()

        return {
            "right_hand": rew_dist + rew_next_dist + action_penalty, 
            "left_hand": rew_dist + rew_next_dist + action_penalty
        }

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()
        # reset when object has fallen
        out_of_reach = self.object_pos[:, 2] <= self.cfg.fall_dist

        # reset when episode ends
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_reach for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.right_hand._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # reset right hand
        delta_max = self.right_robot_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.right_robot_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_right_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.right_hand.data.default_joint_pos[env_ids]  # + self.cfg.reset_dof_pos_noise * rand_delta
        dof_pos[:, -3:] = dof_pos[:, -3:] + self.cfg.reset_dof_pos_noise * rand_delta[:, -3:]

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_right_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel[env_ids] # + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel[:, -3:] = dof_vel[:, -3:] + self.cfg.reset_dof_vel_noise * dof_vel_noise[:, -3:]

        self.right_robot_prev_targets[env_ids] = dof_pos
        self.right_robot_curr_targets[env_ids] = dof_pos

        self.right_diff_ik_controller.reset(env_ids)
        self.next_right_ee_pose_b[env_ids] = torch.tensor(self.cfg.init_right_ee_pose, device=self.device).clone()

        self.right_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.last_embeddings[env_ids] = torch.zeros((64), dtype=torch.float, device=self.device)
        self.last_right_actions[env_ids] = torch.zeros((9), dtype=torch.float, device=self.device)

        # reset left hand
        delta_max = self.left_robot_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.left_robot_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_left_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.left_hand.data.default_joint_pos[env_ids]  # + self.cfg.reset_dof_pos_noise * rand_delta
        dof_pos[:, -3:] = dof_pos[:, -3:] + self.cfg.reset_dof_pos_noise * rand_delta[:, -3:]

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_left_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel[env_ids]  # + self.cfg.reset_dof_vel_noise * dof_vel_noise
        dof_vel[:, -3:] = dof_vel[:, -3:] + self.cfg.reset_dof_vel_noise * dof_vel_noise[:, -3:]

        self.left_robot_prev_targets[env_ids] = dof_pos
        self.left_robot_curr_targets[env_ids] = dof_pos

        self.left_diff_ik_controller.reset(env_ids)
        self.next_left_ee_pose_b[env_ids] = torch.tensor(self.cfg.init_left_ee_pose, device=self.device).clone()

        self.left_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _compute_intermediate_values(self):
        # data for right hand
        self.right_fingertip_pos = self.right_hand.data.body_pos_w[:, self.right_finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w[:, self.right_finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w[:, self.right_finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos
        self.right_hand_dof_vel = self.right_hand.data.joint_vel

        # obtain quantities from simulation
        self.right_jacobian = self.right_hand.root_physx_view.get_jacobians()[:, self.right_ee_jacobi_idx, :, self.right_arm_actuated_dof_indices]
        self.right_ee_pose_w = self.right_hand.data.body_state_w[:, self.right_ee_id, 0:7]
        self.right_ee_pos_w = self.right_ee_pose_w[:, 0:3]
        self.right_root_pose_w = self.right_hand.data.root_state_w[:, 0:7]
        self.right_ee_pos = self.right_ee_pos_w - self.scene.env_origins
        self.right_arm_joint_pos = self.right_hand.data.joint_pos[:, self.right_arm_actuated_dof_indices]
        # compute frame in root frame
        self.right_ee_pos_b, self.right_ee_quat_b = subtract_frame_transforms(
            self.right_root_pose_w[:, 0:3], self.right_root_pose_w[:, 3:7], self.right_ee_pose_w[:, 0:3], self.right_ee_pose_w[:, 3:7]
        )

        # data for left hand
        self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.left_finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.left_finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.left_finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos
        self.left_hand_dof_vel = self.left_hand.data.joint_vel

        # obtain quantities from simulation
        self.left_jacobian = self.left_hand.root_physx_view.get_jacobians()[:, self.left_ee_jacobi_idx, :, self.left_arm_actuated_dof_indices]
        self.left_ee_pose_w = self.left_hand.data.body_state_w[:, self.left_ee_id, 0:7]
        self.left_ee_pos_w = self.left_ee_pose_w[:, 0:3]
        self.left_ee_quat = self.left_ee_pose_w[:, 3:7]
        self.left_root_pose_w = self.left_hand.data.root_state_w[:, 0:7]
        self.left_ee_pos = self.left_ee_pos_w - self.scene.env_origins
        self.left_arm_joint_pos = self.left_hand.data.joint_pos[:, self.left_arm_actuated_dof_indices]
        # compute frame in root frame
        self.left_ee_pos_b, self.left_ee_quat_b = subtract_frame_transforms(
            self.left_root_pose_w[:, 0:3], self.left_root_pose_w[:, 3:7], self.left_ee_pose_w[:, 0:3], self.left_ee_pose_w[:, 3:7]
        )

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # train CNN to regress on keypoint positions
        feature_extractor_loss, _, embeddings = self.feature_extractor.step(
            self._tiled_camera.data.output["rgb"],
            self._tiled_camera.data.output["semantic_segmentation"][..., :3],
            self.object_pos,
        )
        self.embeddings = embeddings.clone().detach()
        self.feature_extractor_loss = feature_extractor_loss.clone().detach()

        # predict next pose
        if self.cfg.is_training:
            pose_predictor_loss, _ = self.pose_predictor.step(
                self.last_embeddings,
                self.last_right_actions,
                self.object_pos,
                infer=False,
            )
            self.pose_predictor_loss = pose_predictor_loss.clone().detach()

        _input_embeddings = self.embeddings.clone()
        _, predicted_next_object_pose = self.pose_predictor.step(
            _input_embeddings,
            self.actions["right_hand"],
            None,
            infer=True,
        )
        self.predicted_next_object_pose = predicted_next_object_pose.clone().detach()

        self.last_embeddings = self.embeddings.clone()
        self.last_right_actions = self.actions["right_hand"].clone()

        # update marker positions
        # self.right_ee_frame_marker.visualize(self.right_ee_pose_w[:, 0:3], self.right_ee_pose_w[:, 3:7])
        # self.left_ee_frame_marker.visualize(self.left_ee_pose_w[:, 0:3], self.left_ee_pose_w[:, 3:7])
        # self.goal_frame_marker.visualize(self.next_left_ee_pose_b[:, 0:3] + self.scene.env_origins, self.next_left_ee_pose_b[:, 3:7])

        self.action_scope_right_hand = torch.norm(
            self.right_robot_curr_targets[:, self.right_hand_actuated_dof_indices]
              - self.right_robot_prev_targets[:, self.right_hand_actuated_dof_indices], p=2, dim=-1)
        self.action_scope_left_hand = torch.norm(
            self.left_robot_curr_targets[:, self.left_hand_actuated_dof_indices]
              - self.left_robot_prev_targets[:, self.left_hand_actuated_dof_indices], p=2, dim=-1)
        self.action_scope_right_arm = torch.norm(
            self.right_robot_curr_targets[:, self.right_arm_actuated_dof_indices]
              - self.right_robot_prev_targets[:, self.right_arm_actuated_dof_indices], p=2, dim=-1)
        self.action_scope_left_arm = torch.norm(
            self.left_robot_curr_targets[:, self.left_arm_actuated_dof_indices]
              - self.left_robot_prev_targets[:, self.left_arm_actuated_dof_indices], p=2, dim=-1)
        
        # save current targets
        self.right_robot_prev_targets[:, self.right_arm_actuated_dof_indices] = self.right_robot_curr_targets[
            :, self.right_arm_actuated_dof_indices
        ]
        self.left_robot_prev_targets[:, self.left_arm_actuated_dof_indices] = self.left_robot_curr_targets[
            :, self.left_arm_actuated_dof_indices
        ]
        self.right_robot_prev_targets[:, self.right_arm_actuated_dof_indices] = self.right_robot_curr_targets[
            :, self.right_arm_actuated_dof_indices
        ]
        self.left_robot_prev_targets[:, self.left_arm_actuated_dof_indices] = self.left_robot_curr_targets[
            :, self.left_arm_actuated_dof_indices
        ]


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
