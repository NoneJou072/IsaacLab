from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from cbam import *
from se import *
from spatial import SpatialLearnedEmbeddings


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if use_attention:
            self.attention = CBAM(planes, 16)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if hasattr(self, 'attention'):
            out = self.attention(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, att_type=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

        if att_type == 'cbam':
            self.attention = CBAM(planes * 4, 16)
        elif att_type == 'se':
            self.attention = SELayer(planes * 4, 16)
        else:
            self.attention = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_attention = False,
        pooling_method: str = "spatial_learned_embeddings",
        num_spatial_blocks: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.num_spatial_blocks = num_spatial_blocks
        self.pooling_method = pooling_method

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.num_classes = num_classes
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], use_attention=use_attention)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], use_attention=use_attention)
        if self.pooling_method == 'avg':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pooling_method == 'spatial_learned_embeddings':
            self.embeddings = SpatialLearnedEmbeddings(
                height=7,
                width=7,
                channel=512,
                num_features=self.num_spatial_blocks,
            )
            self.fce = nn.Linear(512 * self.num_spatial_blocks, num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        print("\n[INFO]: block.expansion", block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        use_attention: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, use_attention
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_attention=use_attention,
                )
            )
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.pooling_method == 'avg':
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        elif self.pooling_method == 'spatial_learned_embeddings':
            x = self.embeddings(x)
            x = nn.Dropout(0.1)(x)
            x = self.fce(x)
            x = nn.LayerNorm(self.num_classes, device=x.device)(x)
            x = torch.tanh(x)
        else:
            raise NotImplementedError(f"Pooling method {self.pooling_method} not implemented")  # pragma: no cover

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    # if weights is not None:
    #     _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        # model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)
        for name, param in model.named_parameters():
            if name in state_dict:
                param.data = state_dict[name]
        for name, buf in model.named_buffers():
            if name in state_dict:
                buf.data = state_dict[name].to(buf.device)  # 处理 BatchNorm 的运行时统计量

    if kwargs["finetuned"]:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        if model.pooling_method == "spatial_learned_embeddings":
            for param in model.embeddings.parameters():
                param.requires_grad = True
            for param in model.fce.parameters():
                param.requires_grad = True

    return model



def resnet14(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 1, 1], weights, progress, **kwargs)


def resnet18(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


def ResNet34(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)


def ResNet50(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


def ResNet101(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)


def ResNet152(*, weights = None, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet(Bottleneck, [3, 8, 36, 3], weights, progress, **kwargs)
