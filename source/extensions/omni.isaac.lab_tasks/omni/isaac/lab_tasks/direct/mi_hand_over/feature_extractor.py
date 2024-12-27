# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
from typing import Dict, List, Optional

from torch import Tensor
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

from omni.isaac.lab.sensors import save_images_to_file
from omni.isaac.lab.utils import configclass
import torchvision.transforms.v2

import cv2
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from vision.resnet import resnet18, resnet14
from vision.sensitivity_utils import SensitivityAnalyzer


def generate_gaussian_noise(height, width, mean=0, std=1):
    """生成高斯噪声背景图像"""
    noise = np.random.normal(mean, std, (height, width, 3))  # 3 通道：RGB
    noise = np.clip(noise, 0, 1)  # 将噪声限制在 [0, 1] 范围内
    return torch.tensor(noise, dtype=torch.float32, device="cuda")
    

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class FeatureExtractorNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = resnet14(
            weights=ResNet18_Weights.DEFAULT,
            norm_layer=FrozenBatchNorm2d,
            use_attention=True,
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=4,
            num_classes=16,
            finetuned=True,
        )

        # self.lstm = nn.LSTM(16, 128, 1, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(16, 3),
        )
        self.data_transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.Resize((224, 224)),
            torchvision.transforms.v2.ToTensor(),
            torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.data_transforms(x)

        features = self.encoder(x)
        pos = self.linear(features)

        return pos, features


@configclass
class FeatureExtractorCfg:
    """Configuration for the feature extractor model."""

    train: bool = True
    """If True, the feature extractor model is trained during the rollout process. Default is False."""

    load_checkpoint: bool = False
    """If True, the feature extractor model is loaded from a checkpoint. Default is False."""

    write_image_to_file: bool = False
    """If True, the images from the camera sensor are written to file. Default is False."""

    show_sensitive_map: bool = False
    record_sensitivity_map: bool = False


class FeatureExtractor:
    """Class for extracting features from image data.

    It uses a CNN to regress keypoint positions from normalized RGB, depth, and segmentation images.
    If the train flag is set to True, the CNN is trained during the rollout process.
    """

    def __init__(self, cfg: FeatureExtractorCfg, device: str, num_envs: int = 1):
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
            print(f"[F.E.]: Loading feature extractor checkpoint from {checkpoint}")
            self.feature_extractor.load_state_dict(torch.load(checkpoint, weights_only=True))

        if self.cfg.train:
            self.optimizer = self.make_optimizer()
            # self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
            self.criterion = nn.MSELoss()
            self.feature_extractor.train()
            print(f"[F.E.]: Feature extractor model is in train mode.")
        else:
            self.criterion = nn.MSELoss()
            self.feature_extractor.eval()
            print(f"[F.E.]: Feature extractor model is in evaluation mode.")

        if self.cfg.show_sensitive_map:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            self.sensitivity_vis = SensitivityAnalyzer(
                device="cuda"
            )
            self.sensitivity_vis.set_model(
                self.feature_extractor.encoder, 
                target_layers=[self.feature_extractor.encoder.layer4])
            if self.cfg.record_sensitivity_map:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
                fps = 30  # 每秒帧数
                output_video = os.path.join(self.log_dir, "sensitivity_map.mp4")
                self.sensitivity_recorder = cv2.VideoWriter(output_video, fourcc, fps, (140, 90))

        # process mujoco image
        mj_img_paths = glob.glob("/home/mi/zhr/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/mi_hand_over/assets/*.png")
        self.mj_img_list = []
        for mj_img_path in mj_img_paths:
            print(f"[F.E.]: Loading image from {mj_img_path}")
            mj_img = cv2.imread(mj_img_path, 1)
            mj_img = cv2.cvtColor(mj_img, cv2.COLOR_BGR2RGB)
            mj_img = torch.from_numpy(mj_img).to(device=self.device) / 255.0
            mj_img = mj_img.unsqueeze(0).repeat(num_envs, 1, 1, 1)
            self.mj_img_list.append(mj_img)

        self.save_step = 0

    def make_optimizer(self, weight_decay=None):
        params = [
            {'params': self.feature_extractor.encoder.fc.parameters(), 'lr': 1e-4},
            {'params': self.feature_extractor.linear.parameters(), 'lr': 1e-4},
        ]
        if self.feature_extractor.encoder.pooling_method == "spatial_learned_embeddings":
            params.append({'params': self.feature_extractor.encoder.embeddings.parameters(), 'lr': 1e-4})
            params.append({'params': self.feature_extractor.encoder.fce.parameters(), 'lr': 1e-4})

        for layer in [self.feature_extractor.encoder.layer3, self.feature_extractor.encoder.layer4]:
            for name, module in layer.named_modules():
                depth = name.count('.')  # 根据 `.` 计数来判断深度
                if depth != 1:
                    continue
                print(f"[F.E.]: Processing module: {name}")

                if name.endswith('attention'):
                    print(f"[F.E.]: Found attention layer: {name}")
                    params.append({'params': module.parameters(), 'lr': 1e-4})
                else:
                    params.append({'params': module.parameters(), 'lr': 1e-5})

        if weight_decay is not None:
            optimizer = torch.optim.AdamW(self.feature_extractor.parameters(), lr=1e-5, weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)

        return optimizer
        
    def _preprocess_images(
        self, rgb_img: torch.Tensor, segmentation_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocesses the input images.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
            segmentation_img (torch.Tensor): Segmentation image tensor. Shape: (N, H, W, 3)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Preprocessed RGB, depth, and segmentation
        """
        transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.CenterCrop((100, 160)),
            torchvision.transforms.v2.Resize((200, 320)),
        ])
        # process rgb image
        rgb_img = rgb_img / 255.0
        rgb_img = rgb_img[:, 70:160, 80:220, :]
        # rgb_img = transforms(rgb_img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # process segmentation image
        segmentation_img = segmentation_img / 255.0
        segmentation_img = segmentation_img[:, 70:160, 80:220, :]
        # segmentation_img = transforms(segmentation_img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        obj_in_image = torch.where(segmentation_img.sum(dim=(1, 2, 3)) > 0, torch.tensor(1.0), torch.tensor(0.0))
        
        # replace the background color
        mask = (segmentation_img == 0).all(dim=-1)
        rnid = np.random.randint(0, 8)
        if rnid < len(self.mj_img_list):
            segmentation_img[mask] = self.mj_img_list[rnid][mask]
        elif rnid == 7:
            background_noise = generate_gaussian_noise(segmentation_img.shape[1], segmentation_img.shape[2]).unsqueeze(0).expand(mask.shape[0], -1, -1, -1)
            segmentation_img[mask] = background_noise[mask]
        else:
            segmentation_img[mask] = rgb_img[mask]
        # random_background_color = np.random.uniform(0, 1, 3).astype(np.float32)
        # segmentation_img[mask] = torch.tensor(random_background_color, device=self.device)
        
        # replace the object color
        random_object_color = np.random.uniform(0, 1, 3).astype(np.float32)
        object_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        segmentation_img[~mask] = torch.tensor(object_color, device=self.device)

        return rgb_img, segmentation_img, obj_in_image

    def _save_images(self, rgb_img: torch.Tensor, segmentation_img: torch.Tensor):
        """Writes image buffers to file.

        Args:
            rgb_img (torch.Tensor): RGB image tensor. Shape: (N, H, W, 3).
        """
        save_images_to_file(rgb_img, "mi_robot_rgb.png")
        save_images_to_file(segmentation_img, "mi_robot_seg.png")
        if self.cfg.show_sensitive_map:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    rgb_img_load = cv2.imread("mi_robot_rgb.png", 1)[:segmentation_img.shape[1], :segmentation_img.shape[2], ::-1]
                    sensitivity_img = self.sensitivity_vis.load_image(rgb_img_load)
                    if self.cfg.record_sensitivity_map:
                        self.sensitivity_recorder.write(sensitivity_img)

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

        rgb_img, input_img, obj_in_img = self._preprocess_images(rgb_img, segmentation_img)

        if self.cfg.write_image_to_file:
            self.save_step += 1
            if self.save_step % 10 == 0:
                self._save_images(rgb_img, input_img)
                if self.save_step == 100 and self.cfg.show_sensitive_map:
                    self.sensitivity_recorder.release()
                    print(f"[F.E.]: Saved sensitivity map to {os.path.join(self.log_dir, 'sensitivity_map.mp4')}")

        if self.cfg.train:
            with torch.enable_grad():
                with torch.inference_mode(False):
                    inputs = input_img.clone()
                    self.optimizer.zero_grad()

                    predicted_pose, features = self.feature_extractor(inputs)
                    # Only update the pose for the object in the image
                    predicted_pose = torch.where(obj_in_img.unsqueeze(-1) == 1, predicted_pose, gt_pose.clone())

                    mask = obj_in_img == 1
                    if mask.shape[0] == 0:
                        return torch.tensor(0.0), predicted_pose, features
                    
                    filtered_predicted_pose = predicted_pose[mask]
                    filtered_gt_pose = gt_pose[mask]
                    pose_loss = self.criterion(filtered_predicted_pose, filtered_gt_pose) * 1000

                    pose_loss.backward()
                    self.optimizer.step()
                    # self.scheduler.step()

                    self.step_count += 1
                    if self.step_count % 1000 == 0:
                        torch.save(
                            self.feature_extractor.state_dict(),
                            os.path.join(self.log_dir, f"cnn_{self.step_count}_{pose_loss.detach().cpu().numpy()}.pth"),
                        )
                    return pose_loss, predicted_pose, features
        else:
            inputs = input_img.clone()
            predicted_pose, features = self.feature_extractor(inputs)
            pose_loss = self.criterion(predicted_pose, gt_pose.clone()) * 100
            return pose_loss, predicted_pose, features
