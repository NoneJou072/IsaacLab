# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from omni.isaac.lab_assets.xiaomi_r_arm import XIAOMI_R_ARM_CFG
from omni.isaac.lab_assets.xiaomi_l_arm import XIAOMI_L_ARM_CFG

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
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from omni.isaac.lab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from .feature_extractor import FeatureExtractor, FeatureExtractorCfg
from .pose_predictor import PosePredictor, PosePredictorCfg



@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    # robot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     min_step_count_between_reset=720,
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_robot"),
    #         "static_friction_range": (1.3, 1.31),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
    #         "stiffness_distribution_params": (0.75, 1.5),
    #         "damping_distribution_params": (0.3, 3.0),
    #         "operation": "scale",
    #         "distribution": "log_uniform",
    #     },
    # )
    # robot_joint_limits = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("right_hand", joint_names=".*"),
    #         "lower_limit_distribution_params": (0.00, 0.01),
    #         "upper_limit_distribution_params": (0.00, 0.01),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )

    # -- object
    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "static_friction_range": (0.4, 1.3),
    #         "dynamic_friction_range": (1.0, 1.0),
    #         "restitution_range": (1.0, 1.0),
    #         "num_buckets": 250,
    #     },
    # )
    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.4, 0.8),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # -- scene
    # reset_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     is_global_time=True,
    #     interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )


@configclass
class MiArmHandOverRGBCameraEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 4
    possible_agents = ["right_hand", "left_hand"]
    action_spaces = {"right_hand": 9, "left_hand": 13}

    # diff_ik_controller
    diff_ik_controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.01}
    )

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1112975,  -0.03028673,  0.59222811), rot=(0.90595612, -0.00510674,  0.42333978,  0.00092648), convention="world"),
        data_types=["rgb", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=60.0, clipping_range=(0.2, 4.0)
        ),
        width=320,
        height=200,
    )
    feature_extractor = FeatureExtractorCfg(write_image_to_file=False, load_checkpoint=False)
    pose_predictor = PosePredictorCfg(load_checkpoint=False)

    observation_spaces = {
        "right_hand": 16 + 16*1, 
        "left_hand": 13 + 32*3
    }

    state_space = 103 + 16
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
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # robot
    right_robot_cfg: ArticulationCfg = XIAOMI_R_ARM_CFG.replace(prim_path="/World/envs/env_.*/RightRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0, 0.0),
            joint_pos={"R_(?!thumb).*_proximal_joint": 0.572, # /0.172
                   "R_thumb_opp_joint": 0.3,  # /1.526
                   "R_thumb_proximal_joint": 0.573, # /0.873
                   "R_ARM1_SHOULDER_P": -0.263,  # -15.06
                   "R_ARM2_SHOULDER_R": -0.453,  # -25.95
                   "R_ARM3_SHOULDER_Y": 0.597,  # 34.24
                   "R_ARM4_ELBOW_P": -1.26,  # -72.07
                   "R_ARM5_ELBOW_Y": -2.07,  # -118.6
                   "R_ARM6_WRIST_R": -0.26,  # -15
                   "R_ARM7_WRIST_P": 0.0675,  # 3.87
                   },
        )
    )
    right_hand_actuated_joint_names = [
        "R_pinky_finger_proximal_joint",
        "R_ring_finger_proximal_joint",
        "R_middle_finger_proximal_joint",
        "R_index_finger_proximal_joint",
        "R_thumb_proximal_joint",
        "R_thumb_opp_joint",
    ]

    right_arm_actuated_joint_names = [
        "R_ARM1_SHOULDER_P", 
        "R_ARM2_SHOULDER_R",
        "R_ARM3_SHOULDER_Y",
        "R_ARM4_ELBOW_P",
        "R_ARM5_ELBOW_Y",
        "R_ARM6_WRIST_R",
        "R_ARM7_WRIST_P"
    ]

    right_ee_body_name = "R_hand_contact"

    init_right_ee_pose = [0.501, -0.204, 0.1296, -0.1443, 0.6739, 0.1104, 0.7161]
    init_left_ee_pose = [0.49, 0.21, 0.1, -0.0727, -0.9020,  0.1253, -0.4068]

    left_robot_cfg: ArticulationCfg = XIAOMI_L_ARM_CFG.replace(prim_path="/World/envs/env_.*/LeftRobot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0, 0.0),
            joint_pos={"L_(?!thumb).*_proximal_joint": -0.567,
                   "L_thumb_opp_joint": -0.193,
                   "L_thumb_proximal_joint": -0.94,
                   "AL_SHOULDER_P": -0.1248,  # -7.15
                   "AL_SHOULDER_R": 0.1331,  # 7.63
                   "AL_SHOULDER_Y": -0.251,  # -5.3
                   "AL_ELBOW_P": -1.07,  # -90.3
                   "AL_ELBOW_Y": 1.67,  # 79.3
                   "AL_WRIST_R": -0.209,  # 39.2
                   "AL_WRIST_P": 0.3639,  # 20.8
                   },
        )
    )

    left_hand_actuated_joint_names = [
        "L_pinky_finger_proximal_joint",
        "L_ring_finger_proximal_joint",
        "L_middle_finger_proximal_joint",
        "L_index_finger_proximal_joint",
        "L_thumb_proximal_joint",
        "L_thumb_opp_joint",
    ]

    left_arm_actuated_joint_names = [
        "AL_SHOULDER_P", 
        "AL_SHOULDER_R",
        "AL_SHOULDER_Y",
        "AL_ELBOW_P",
        "AL_ELBOW_Y",
        "AL_WRIST_R",
        "AL_WRIST_P"
    ]

    right_fingertip_body_names = [
        "R_pinky_fingertip",
        "R_ring_fingertip",
        "R_middle_fingertip",
        "R_index_fingertip",
        "R_thumbtip",
    ]

    left_fingertip_body_names = [
        "L_pinky_finger_distal",
        "L_ring_finger_distal",
        "L_middle_finger_distal",
        "L_index_finger_distal",
        "L_thumb_distal",
    ]

    left_ee_body_name = "L_HAND_CONTACT"

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CuboidCfg(
            size=(0.045, 0.045, 0.045),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 1.0, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8, restitution_combine_mode="multiply"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
                # linear_damping=0.0,
                # angular_damping=1.0,
                # max_angular_velocity = 0.1
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, -0.20, 0.13), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.CuboidCfg(
                size=(0.045, 0.045, 0.045),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
            ),
        },
    )

    # next object
    next_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "next_goal": sim_utils.CuboidCfg(
                size=(0.02, 0.02, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.9, 0.2)),
            ),
        },
    )
    next_object_cfg2: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "next_goal": sim_utils.CuboidCfg(
                size=(0.02, 0.02, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.2)),
            ),
        },
    )
    # domain randomization
    events: EventCfg = EventCfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4, replicate_physics=True)

    # reset
    reset_position_noise = 0.005  # range of position at reset
    reset_dof_pos_noise = 0.1  # range of dof pos at reset
    reset_dof_vel_noise = 0.01  # range of dof vel at reset
    # scales and constants
    fall_dist = -0.15
    vel_obs_scale = 0.2
    act_moving_average_hand = 1
    act_moving_average_arm = 0.05
    action_scale = 4.0
    # reward-related scales
    dist_reward_scale = 20.0

    is_training = True

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # action_noise_model: dict[str, NoiseModelWithAdditiveBiasCfg] = {
    #     "right_hand": NoiseModelWithAdditiveBiasCfg(
    #         noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #         bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
    #     ),
    #     "left_hand": NoiseModelWithAdditiveBiasCfg(
    #         noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #         bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
    #     ),
    # }
    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: dict[str, NoiseModelWithAdditiveBiasCfg] = {
        "right_hand": NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        ),
        "left_hand": NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
        ),
    }


@configclass
class MiArmHandOverRGBCameraEnvPlayCfg(MiArmHandOverRGBCameraEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(
        train=False, load_checkpoint=True, write_image_to_file=False, show_sensitive_map=False, record_sensitivity_map=False
    )
    pose_predictor = PosePredictorCfg(train=False, load_checkpoint=True)

    action_noise_model: object = None
    observation_noise_model: object = None
    events: object = None

    is_training = False
