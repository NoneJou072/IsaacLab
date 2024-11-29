import argparse
import os
import cv2
import yaml
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, preprocess_image
)
from torch import nn

import os
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Normalize
import cv2

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

import torchvision.transforms.v2


class ResNet18(ResNet):
    def __init__(self, num_classes=1000):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        resnet18_pretrained_state_dict = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
        self.load_state_dict(resnet18_pretrained_state_dict)

        for name, param in self.named_parameters():
            print(name)
            if 'conv1' in name or 'bn1' in name or \
            'layer1' in name or 'layer2' in name or 'layer3' in name:
                param.requires_grad = False


class FeatureExtractorNetwork(nn.Module):
    """CNN architecture used to regress keypoint positions of the in-hand cube from image data."""

    def __init__(self):
        super().__init__()

        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.resnet.parameters():
            param.requires_grad = True
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_ftrs, out_features=32, bias=True)

        # self.features_extractor = nn.Sequential(*list(ResNet18().children())[:-1])  # 输出通道数为512
        # self.mlp = nn.Sequential(
        #     nn.Linear(512, 32),
        # )

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

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def get_features(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.data_transforms(x)

        features = self.resnet(x)
        # features = self.features_extractor(x)
        # features = self.mlp(features.squeeze(-1).squeeze(-1))
        return features.view(-1, 32)
    
class SensitivityAnalyzer(object):
    def __init__(self, device='cpu') -> None:
        self.device = device
        self.aug_smooth = False
        self.eigen_smooth = False
        self.output_dir = "/home/mi/zhr/"
        self.use_goal_image = False
        self.methods = {
                        "gradcam": GradCAM,
                        "hirescam": HiResCAM,
                        "scorecam": ScoreCAM,
                        "gradcam++": GradCAMPlusPlus,
                        "ablationcam": AblationCAM,
                        "xgradcam": XGradCAM,
                        "eigencam": EigenCAM,
                        "eigengradcam": EigenGradCAM,
                        "layercam": LayerCAM,
                        "fullgrad": FullGrad,
                        "gradcamelementwise": GradCAMElementWise,
                        'kpcacam': KPCA_CAM
                        }
        self.method = "gradcam"

    def set_model(self, model, target_layers):
        # model = models.resnet18(pretrained=True).to(torch.device(self.device)).eval()
        # if self.use_goal_image:
        #     model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #     model.fc = nn.Linear(model.fc.in_features, 1000)  # 修改全连接层，保持输出类别数不变

        # model.load_state_dict(body.state_dict(), strict=False)

        # target_layers = [model.layer4]
        cam_algorithm = self.methods[self.method]
        self.cam = cam_algorithm(model=model, target_layers=target_layers)
        self.cam.batch_size = 32

    def load_image(self, image_array: torch.Tensor, targets=None):
        rgb_img = image_array.copy()  # [200, 320, 3]
        if rgb_img.shape[2] == 3:
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]).to(self.device)  # [1, 3, 200, 320]
        elif rgb_img.shape[2] == 6:
            input_tensor = preprocess_image(rgb_img,
                                            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).to(self.device)  # [1, 3, 200, 320]
        print("input_tensor.shape)", input_tensor.shape)
        grayscale_cam = self.cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=self.aug_smooth,
                            eigen_smooth=self.eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img[:,:,0:3], grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cam_output_path = os.path.join(self.output_dir, f'{self.method}.jpg')
        print(cam_output_path)
        cv2.imwrite(cam_output_path, cam_image)
        cv2.imshow("cam", cam_image)
        cv2.waitKey(1000)


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    sensitivity_vis = SensitivityAnalyzer(
                device="cuda"
            )
    model = FeatureExtractorNetwork()
    # model.load_state_dict(torch.load("/home/mi/zhr/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/mi_hand_over/logs/feature_extractor/cnn_99800_0.04608960449695587.pth", weights_only=True))
    model.eval()

    print("model", model)
    sensitivity_vis.set_model(
        model.resnet, 
        target_layers=[model.resnet.layer4])
    
    rgb_img = cv2.imread("/home/mi/zhr/IsaacLab/mi_robot_rgb.png", 1)[:200, :320, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    print("rgb_img.shape", rgb_img.shape)
    sensitivity_vis.load_image(image_array=rgb_img)
