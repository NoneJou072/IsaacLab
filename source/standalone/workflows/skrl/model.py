import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import Normalize

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import time

class ResNet10(ResNet):
    def __init__(self, num_classes=1000):
        super(ResNet10, self).__init__(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
        # self.fc = nn.Linear(512, num_classes)

        # print the model structure
        resnet18_pretrained_state_dict = resnet18(weights=ResNet18_Weights.DEFAULT).state_dict()
        resnet10_state_dict = self.state_dict()

        # 根据 ResNet10 的结构调整加载 ResNet18 的预训练参数
        pretrained_dict = {k: v for k, v in resnet18_pretrained_state_dict.items() if k in resnet10_state_dict}
        resnet10_state_dict.update(pretrained_dict)

        # 将预训练参数加载到 ResNet10 模型中
        self.load_state_dict(resnet10_state_dict)

        # 冻结需要保持不变的层，通常是前几个卷积层
        for name, param in self.named_parameters():
            if 'conv1' in name or 'bn1' in name or 'layer1' in name or 'layer2' in name:
                param.requires_grad = False

        
# define models (stochastic and deterministic models) using mixins
class Policy2(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.normalize = Normalize(self.mean, self.std)
        
        resnet = ResNet10()
        self.features_extractor = nn.Sequential(*list(resnet.children())[:-1])  # 输出通道数为512
        
        self.net = nn.Sequential(nn.Linear(512 + 46, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        state = inputs["states"]
        low_dim_state = state[:, :46]
        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        rgb_state = state[:, 46:].view(-1, 224, 224, 3).permute(0, 3, 1, 2) / 255.0
        rgb_state = self.normalize(rgb_state)
        # features = self.features_extractor(rgb_state).squeeze(-1).squeeze(-1)
        features = torch.zeros(low_dim_state.shape[0], 512).to(self.device)
        mean_actions = self.net(torch.cat([features, low_dim_state], dim=-1))

        return mean_actions, self.log_std_parameter, {}
    
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU(),
                                 nn.Linear(256, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        mean_actions = self.net(inputs["states"])

        return mean_actions, self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
