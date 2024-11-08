# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from omni.isaac.lab_assets.xiaomi_hand import XIAOMI_HAND_CFG

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, RigidObjectCfg
from omni.isaac.lab.envs import DirectMARLEnvCfg, ViewerCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import PhysxCfg, SimulationCfg
from omni.isaac.lab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from .feature_extractor import FeatureExtractor, FeatureExtractorCfg


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("right_hand"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("right_hand", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )


@configclass
class MiHandOverRGBCameraEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.5
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {"right_hand": 9, "left_hand": 9}

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.2, -0.35, 0.6), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
        ),
        width=224,
        height=224,
    )
    feature_extractor = FeatureExtractorCfg()

    num_channels = 3
    observation_spaces = {
        "right_hand": 46 + 128, 
        "left_hand": 46 + 128
    }

    state_space = 232 + 128
    write_image_to_file = True

    # change viewer settings
    # viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )
    # robot
    right_robot_cfg: ArticulationCfg = XIAOMI_HAND_CFG.replace(prim_path="/World/envs/env_.*/RightRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos={"R_(?!thumb).*_proximal_joint": 0.172,
                   "R_(?!thumb).*_distal_joint": 0.172,
                   "R_thumb_opp_joint": 1.526,
                   "R_thumb_proximal_joint": 0.318,
                   "R_thumb_middle_joint": -0.793,
                   "R_thumb_distal_joint": -0.53,
                   "D6Joint:.*": 0.0,
                   },
        )
    )
    left_robot_cfg: ArticulationCfg = XIAOMI_HAND_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.7, 0.5),
            rot=(-0.5, 0.5, 0.5, 0.5),
            joint_pos={"R_(?!thumb).*_proximal_joint": 0.172,
                   "R_(?!thumb).*_distal_joint": 0.172,
                   "R_thumb_opp_joint": 1.526,
                   "R_thumb_proximal_joint": 0.318,
                   "R_thumb_middle_joint": -0.793,
                   "R_thumb_distal_joint": -0.53,
                   "D6Joint:.*": 0.0,
                   },
        )
    )
    actuated_joint_names = [
        "R_pinky_finger_proximal_joint",
        "R_ring_finger_proximal_joint",
        "R_middle_finger_proximal_joint",
        "R_index_finger_proximal_joint",
        "R_thumb_proximal_joint",
        "R_thumb_opp_joint",
        "D6Joint:0",
        "D6Joint:1",
        "D6Joint:2",
    ]
    fingertip_body_names = [
        "R_pinky_fingertip",
        "R_ring_fingertip",
        "R_middle_fingertip",
        "R_index_fingertip",
        "R_thumbtip",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.SphereCfg(
            radius=0.0335,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.1, 0.54), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.0335,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
            ),
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4, replicate_physics=True)

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # scales and constants
    fall_dist = 0.24
    vel_obs_scale = 0.2
    act_moving_average = 1.0
    # reward-related scales
    dist_reward_scale = 20.0


@configclass
class MiHandOverRGBCameraEnvPlayCfg(MiHandOverRGBCameraEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)