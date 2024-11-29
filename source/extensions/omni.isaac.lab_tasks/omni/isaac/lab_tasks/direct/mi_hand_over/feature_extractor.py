# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18, ResNet18_Weights

from omni.isaac.lab.sensors import save_images_to_file
from omni.isaac.lab.utils import configclass
import torchvision.transforms.v2


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=32, bias=True)

        self.linear = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.data_transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.Resize((224, 224)),
            torchvision.transforms.v2.ToTensor(),
            torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x1, x2):
        f1 = self.get_features(x1)
        f2 = self.get_features(x2)
        x = torch.cat((f1, f2), dim=-1)
        out = self.linear(x)
        return out, x
    
    def get_features(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.data_transforms(x)

        features = self.resnet(x)
        return features.view(-1, 32)


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str):
        """Initialize the feature extractor model.

        Args:
            cfg (FeatureExtractorCfg): Configuration for the feature extractor model.
            device (str): Device to run the model on.
        """

        self.cfg = cfg
        self.device = device

        # Feature extractor model
        self.feature_extractor = FeatureExtractorNetwork()
        self.feature_extractor.to(self.device)

        self.step_count = 0
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs/feature_extractor")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.cfg.load_checkpoint:
            list_of_files = glob.glob(self.log_dir + "/*.pth")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint = os.path.join(self.log_dir, latest_file)
            print(f"[INFO]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
            self.l2_loss = nn.MSELoss()
            self.feature_extractor.train()
        else:
            self.feature_extractor.eval()

        self.save_step = 0

    def _preprocess_images(
        self, rgb_img: torch.Tensor, segmentation_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed RGB, depth, and segmentation
        """
        rgb_img = rgb_img / 255.0
        # process segmentation image
        segmentation_img = segmentation_img / 255.0
        mean_tensor = torch.mean(segmentation_img, dim=(1, 2), keepdim=True)
        segmentation_img -= mean_tensor

        return rgb_img, segmentation_img

    def _save_images(self, rgb_img: torch.Tensor, segmentation_img: torch.Tensor):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
        """
        save_images_to_file(rgb_img, "mi_robot_rgb.png")
        save_images_to_file(segmentation_img, "mi_robot_seg.png")

    def step(
        self, rgb_img: torch.Tensor, segmentation_img: torch.Tensor, gt_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts the features using the images and trains the model if the train flag is set to True.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            gt_pose (torch.Tensor): Ground truth pose tensor (position and corners). Shape: (N, 27).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Pose loss and predicted pose.
        """

        rgb_img, segmentation_img = self._preprocess_images(rgb_img, segmentation_img)
        if self.cfg.write_image_to_file:
            self.save_step += 1
            if self.save_step % 100 == 0:
                self._save_images(rgb_img, segmentation_img)
                # transform_resize = torchvision.transforms.v2.Resize((224, 224))
                # resized_img = transform_resize(rgb_img.clone().permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                # self._save_images(resized_img)

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    rgb_img_input = rgb_img.clone()
                    seg_img_input = segmentation_img.clone()
                    self.optimizer.zero_grad()

                    predicted_pose, features = self.feature_extractor(rgb_img_input, seg_img_input)
                    pose_loss = self.l2_loss(predicted_pose, gt_pose.clone()) * 100

                    pose_loss.backward()
                    self.optimizer.step()

                    self.step_count += 1

                    if self.step_count % 10000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )
                    return pose_loss, predicted_pose, features
        else:
            rgb_img_input = rgb_img.clone()
            seg_img_input = segmentation_img.clone()
            predicted_pose, features = self.feature_extractor(rgb_img_input, seg_img_input)
            return torch.zeros(1), predicted_pose, features
