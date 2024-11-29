# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision.transforms.v2

from omni.isaac.lab.utils import configclass


class PosePredictorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(9 + 64, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.linear(x)


@configclass
class PosePredictorCfg:
    """Configuration for the pose predictor model."""

    train: bool = True
    """If True, the pose predictor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the pose predictor model is loaded from a checkpoint. Default is False."""


class PosePredictor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: PosePredictorCfg, device: str):
        """Initialize the pose predictor model.

        Args:
            cfg (PosePredictorCfg): Configuration for the pose predictor model.
            device (str): Device to run the model on.
        """

        self.cfg = cfg
        self.device = device

        # pose predictor model
        self.pose_predictor = PosePredictorNetwork()
        self.pose_predictor.to(self.device)

        self.step_count = 0
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs/pose_predictor")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading pose predictor checkpoint from {checkpoint}")
            self.pose_predictor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.pose_predictor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.pose_predictor.train()
        else:
            self.pose_predictor.eval()

        self.save_step = 0

    def step(
        self, embeddings: torch.Tensor, actions: torch.Tensor, gt_pose: torch.Tensor, infer=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            embeddings (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            gt_pose (torch.Tensor): Ground truth pose tensor (position and corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """

        if not infer:
            self.pose_predictor.train()
            with torch.enable_grad():
                with torch.inference_mode(False):
                    inputs = torch.concat((embeddings, actions), dim=-1)
                    self.optimizer.zero_grad()

                    predicted_pose = self.pose_predictor(inputs)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    self.step_count += 1

                    if self.step_count % 10000 == 0:
                        torch.save(
                            self.pose_predictor.state_dict(),
                            os.path.join(self.log_dir, f"mlp_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )

                    return pose_loss, predicted_pose
        else:
            self.pose_predictor.eval()
            inputs = torch.concat((embeddings, actions), dim=-1)
            predicted_pose = self.pose_predictor(inputs)
            return torch.zeros(1), predicted_pose
