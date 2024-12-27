import torch
import torch.nn as nn


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height: int, width: int, channel: int, num_features: int = 5, param_dtype=torch.float32):
        super(SpatialLearnedEmbeddings, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features
        self.param_dtype = param_dtype

        # Initialize the kernel with the same shape as in JAX: (height, width, channel, num_features)
        self.kernel = nn.Parameter(torch.empty(self.height, self.width, self.channel, self.num_features, dtype=self.param_dtype, device='cuda'))
        nn.init.kaiming_normal_(self.kernel, mode='fan_out', nonlinearity='relu')  # Equivalent to 'lecun_normal' init in JAX

    def forward(self, features):
        """
        features is a tensor of shape (B, C, H, W)
        """
        squeeze = False
        if len(features.shape) == 3:
            # If input tensor has shape (C, H, W), we add a batch dimension
            features = features.unsqueeze(0)
            squeeze = True
        features = features.permute(0, 2, 3, 1)  # Shape: (B, H, W, C)
        batch_size = features.shape[0]

        # We need to broadcast the kernel and features for elementwise multiplication
        features_expanded = features.unsqueeze(-1)  # Shape: (B, H, W, C, 1)
        kernel_expanded = self.kernel.unsqueeze(0)  # Shape: (1, H, W, C, num_features)
        # Perform the elementwise multiplication and sum over H and W (axis 1, 2)
        features = torch.sum(features_expanded * kernel_expanded, dim=(1, 2))  # Shape: (B, num_features)
        features = features.reshape(batch_size, -1)
        if squeeze:
            features = features.squeeze(0)
        return features
    