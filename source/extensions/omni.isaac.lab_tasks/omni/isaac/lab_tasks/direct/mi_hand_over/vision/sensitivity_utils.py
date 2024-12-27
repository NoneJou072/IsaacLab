import os
import argparse

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet
from torchvision.models.resnet import BasicBlock
import torchvision.transforms.v2
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, preprocess_image
)

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from resnet import resnet18, resnet14


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

    def __init__(self, use_attention=False):
        super().__init__()

        self.encoder = resnet14(
            weights=ResNet18_Weights.DEFAULT,
            norm_layer=FrozenBatchNorm2d,
            use_attention=use_attention,
            pooling_method="spatial_learned_embeddings",
            num_spatial_blocks=4,
            num_classes=32,
            finetuned=False,
        )

        self.linear = nn.Sequential(
            nn.Linear(32, 3),
        )
        self.data_transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.Resize((224, 224)),
            torchvision.transforms.v2.ToTensor(),
            torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x2):
        f2 = self.get_features(x2)
        out = self.linear(f2)
        return out, f2
    
    def get_features(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.data_transforms(x)
        features = self.encoder(x)
        return features


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
        self.method = "gradcam++"

    def set_model(self, model, target_layers):
        cam_algorithm = self.methods[self.method]
        self.cam = cam_algorithm(model=model, target_layers=target_layers)
        self.cam.batch_size = 32

    def load_image(self, image_array: np.ndarray):
        rgb_img = cv2.resize(image_array, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]).to(self.device)  # [1, 3, 200, 320]

        grayscale_cam = self.cam(input_tensor=input_tensor,
                            targets=None,
                            aug_smooth=self.aug_smooth,
                            eigen_smooth=self.eigen_smooth)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img[:,:,0:3], grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cam_output_path = os.path.join(self.output_dir, f'{self.method}.jpg')
        cam_image = cv2.resize(cam_image, (140, 90))

        cv2.imwrite(cam_output_path, cam_image)
        return cam_image

    def show_image(self, path):
        cam_image = cv2.imread(path)
        cv2.imshow("cam", cam_image)
        cv2.waitKey(1000)


def compare_model_weights(model_before, model_after):
    """
    比较加载前后模型的参数，输出不同的参数
    """
    # 获取模型参数字典
    state_dict_before = model_before.state_dict()
    state_dict_after = model_after.state_dict()

    # 存储不同的参数
    differences = {}

    # 遍历参数字典
    for param_name in state_dict_before:
        if not torch.equal(state_dict_before[param_name], state_dict_after[param_name]):
            differences[param_name] = {
                "before": state_dict_before[param_name],
                "after": state_dict_after[param_name]
            }

    return differences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Grad-CAM')
    parser.add_argument('-c', '--checkpoint', required=False, default=None, type=str, help='Path to model checkpoint') 
    parser.add_argument('--use-attention', action='store_true', help='Use attention in the model')
    args = parser.parse_args()

    sensitivity_vis = SensitivityAnalyzer(device="cuda")
    model = FeatureExtractorNetwork(use_attention=args.use_attention).to("cuda")
    model2 = FeatureExtractorNetwork(use_attention=args.use_attention).to("cuda")

    pth_name = args.checkpoint
    print("Loading model from:", pth_name)
    if pth_name is not None:
        state_dict = torch.load(
            f"/home/mi/zhr/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/mi_hand_over/logs/feature_extractor/{pth_name}",
            weights_only=False,
            map_location="cuda"
        )
        # model.load_state_dict(state_dict)
        for name, param in model.named_parameters():
            if name in state_dict:
                param.data = state_dict[name]

        for name, buf in model.named_buffers():
            if name in state_dict:
                buf.data = state_dict[name].to(buf.device)

    model.eval()
    model2.eval()
    differences = compare_model_weights(model2, model)

    # 打印不同的参数
    if differences:
        print("加载前后不同的参数：")
        for param_name, values in differences.items():
            print(f"参数名称: {param_name}")
    else:
        print("加载前后的参数完全一致！")
    sensitive_layers = [
        # model.resnet.conv1,
        # model.resnet.layer1,
        # model.resnet.layer2,
        # model.resnet.layer3,
        # model.resnet.layer3[0].conv1,
        # model.resnet.layer3[0].conv2,
        # model.resnet.layer3[0].attention,
        # model.resnet.layer3[1].conv1,
        # model.resnet.layer3[1].conv2,
        # model.resnet.layer3[1].attention,
        # model.resnet.layer4[0].conv1,
        # model.resnet.layer4[0].conv2,
        # model.resnet.layer4[0].attention.SpatialAtt.sigmoid,
        model.encoder.layer4[-1],
        # model.resnet.layer4,
        # model.resnet.layer4[1].conv1,
        # model.resnet.layer4[1].conv2,
        # model.resnet.layer4[1].attention,
    ]
    # model.eval()
    # print(model.resnet)
    for layer in sensitive_layers:
        sensitivity_vis.set_model(
            model.encoder, 
            target_layers=[layer])
        
        for i in range(1):
            for j in range(1):
                # rgb_img = cv2.imread("/home/mi/zhr/IsaacLab/mi_robot_rgb.png", 1)[90*i:90+90*i, 140*j:140+140*j, ::-1]
                rgb_img = cv2.imread("/home/mi/zhr/mi_genesis/rgb_img.png", 1)[:, :, ::-1]
                
                cam_image = sensitivity_vis.load_image(image_array=rgb_img)
                cv2.imshow("cam", cv2.resize(cam_image, (560, 360)))
                cv2.waitKey(1000)
