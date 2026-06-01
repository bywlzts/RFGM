import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from skimage.feature import canny
from skimage.color import rgb2gray
import numpy as np

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class ResidualBlock(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.bn = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn(self.conv1(x)), inplace=True)
        out = self.conv2(out)
        return identity + out

class CannyEdgeExtractor:
    def __init__(self, sigma=2):
        """
        初始化 Canny 边缘提取器
        :param sigma: 高斯模糊的标准差
        """
        self.sigma = sigma

    def tensor_to_numpy(self, tensor):
        """
        将 Tensor 转换为 NumPy 格式
        :param tensor: 输入 Tensor (B, C, H, W)
        :return: 转换后的 NumPy 数组 (B, H, W)
        """
        # 确保 Tensor 在 CPU 上
        tensor = tensor.detach().cpu().numpy()

        # 如果是 RGB 图像 (B, 3, H, W)，转换为灰度图
        if tensor.shape[1] == 3:
            tensor = 0.299 * tensor[:, 0, :, :] + 0.587 * tensor[:, 1, :, :] + 0.114 * tensor[:, 2, :, :]

        return tensor

    def numpy_to_tensor(self, array):
        """
        将 NumPy 数组转换为 Tensor 格式
        :param array: 输入 NumPy 数组 (B, H, W)
        :return: 转换后的 Tensor (B, 1, H, W)
        """
        # 添加通道维度并转换为 Tensor
        return torch.from_numpy(array).unsqueeze(1).float()

    def extract_edges(self, tensor):
        """
        提取 Canny 边缘
        :param tensor: 输入 Tensor (B, C, H, W)
        :return: 边缘图 Tensor (B, 1, H, W)
        """
        # 将 Tensor 转换为 NumPy 格式
        numpy_images = self.tensor_to_numpy(tensor)

        # 对每张图像提取 Canny 边缘
        edges = []
        for img in numpy_images:
            edge = canny(img, sigma=self.sigma).astype(np.float32)  # 提取边缘
            edges.append(edge)

        # 将结果堆叠为 NumPy 数组 (B, H, W)
        edges = np.stack(edges, axis=0)

        # 将结果转换回 Tensor 格式
        return self.numpy_to_tensor(edges)