# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Xiaomi Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Xiaomi Hand with implicit actuator model.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

XIAOMI_R_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/mi/zhr/IsaacLab/xiaomi_V2/usd/M92C3_doublearm1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(stiffness=.0000),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0, 0.0),
        joint_pos={"R_(?!thumb).*_proximal_joint": 0.172,
                   "R_(?!thumb).*_distal_joint": 0.172,
                   "R_thumb_opp_joint": 1.526,
                   "R_thumb_proximal_joint": 0.318,
                   "R_thumb_middle_joint": -0.793,
                   "R_thumb_distal_joint": -0.53,
                   "R_ARM1_SHOULDER_P":-0.822,
                   "R_ARM2_SHOULDER_R":-1.1,
                   "R_ARM3_SHOULDER_Y":1.73,
                   "R_ARM4_ELBOW_P":-1.84,
                   "R_ARM5_ELBOW_Y":-2.42,
                   "R_ARM6_WRIST_R":0.686,
                   "R_ARM7_WRIST_P":0.578,
                    # "R_ARM1_SHOULDER_P":0,
                    # "R_ARM2_SHOULDER_R":0,
                    # "R_ARM3_SHOULDER_Y":0,
                    # "R_ARM4_ELBOW_P":0,
                    # "R_ARM5_ELBOW_Y":0,
                    # "R_ARM6_WRIST_R":0,
                    # "R_ARM7_WRIST_P":0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["R_.*_proximal_joint", "R_thumb_opp_joint"],
            effort_limit=0.9,
            velocity_limit=10.0,
            stiffness={
                "R_(?!thumb).*_(proximal|distal)_joint": 1.0,
                "R_thumb_proximal_joint": 0.9,
                "R_thumb_opp_joint": 1.0,
            },
            damping=0.15,
            friction=0.01,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["R_ARM1_SHOULDER_P", "R_ARM2_SHOULDER_R","R_ARM3_SHOULDER_Y",
                              "R_ARM4_ELBOW_P","R_ARM5_ELBOW_Y","R_ARM6_WRIST_R","R_ARM7_WRIST_P"],
            effort_limit=100,
            velocity_limit=1.0,
            stiffness={
                "R_ARM1_SHOULDER_P": 400,
                "R_ARM2_SHOULDER_R": 500,
                "R_ARM3_SHOULDER_Y": 400,
                "R_ARM4_ELBOW_P": 400,
                "R_ARM5_ELBOW_Y": 400,
                "R_ARM6_WRIST_R": 500,
                "R_ARM7_WRIST_P": 500,
            },
            damping={
                "R_ARM1_SHOULDER_P": 80,
                "R_ARM2_SHOULDER_R": 40,
                "R_ARM3_SHOULDER_Y": 80,
                "R_ARM4_ELBOW_P": 80,
                "R_ARM5_ELBOW_Y": 80,
                "R_ARM6_WRIST_R": 40,
                "R_ARM7_WRIST_P": 40,
            },
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Xiaomi Hand robot."""
