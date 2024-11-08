# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
ShadowHand Over environment.
"""

import gymnasium as gym

from . import agents
from .mi_hand_over_env import MiHandOverEnv
from .mi_hand_over_camera_env import MiHandOverRGBCameraEnv
from .mi_hand_over_camera_env2 import MiHandOverRGBCameraEnv as MiHandOverRGBCameraEnv2
from .mi_hand_over_env_cfg import MiHandOverEnvCfg
from .mi_hand_over_camera_env_cfg import MiHandOverRGBCameraEnvCfg
from .mi_hand_over_camera_env2_cfg import MiHandOverRGBCameraEnvCfg as MiHandOverRGBCameraEnv2Cfg
from .mi_hand_over_camera_env2_cfg import MiHandOverRGBCameraEnvPlayCfg as MiHandOverRGBCameraEnv2PlayCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Mi-Hand-Over-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.mi_hand_over:MiHandOverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MiHandOverEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Mi-Hand-Over-RGB-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.mi_hand_over:MiHandOverRGBCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MiHandOverRGBCameraEnvCfg,
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_camera_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Mi-Hand-Over-RGB-Camera-Direct-v1",
    entry_point="omni.isaac.lab_tasks.direct.mi_hand_over:MiHandOverRGBCameraEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MiHandOverRGBCameraEnv2Cfg,
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_camera_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Mi-Hand-Over-RGB-Camera-Direct-Play-v1",
    entry_point="omni.isaac.lab_tasks.direct.mi_hand_over:MiHandOverRGBCameraEnv2",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MiHandOverRGBCameraEnv2PlayCfg,
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_camera_mappo_cfg.yaml",
    },
)
