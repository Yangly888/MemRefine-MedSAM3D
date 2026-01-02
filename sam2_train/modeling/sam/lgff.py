import torch
import torch.nn as nn
import torch.nn.functional as F

class LGFFBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dim=2):
        """
        dim=2 for 2D image segmentation (LGAG), dim=3 for 3D segmentation (DFF)
        """
        super(LGFFBlock, self).__init__()
        assert dim in [2, 3], "Only support 2D or 3D"

        self.dim = dim
        inter_channels = inter_channels or in_channels * 2
        Conv = nn.Conv2d if dim == 2 else nn.Conv3d
        BN = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        Pool = nn.AdaptiveAvgPool2d if dim == 2 else nn.AdaptiveAvgPool3d

        # LGAG path: gate mechanism
        self.W_g = nn.Sequential(
            Conv(in_channels, inter_channels, kernel_size=3, padding=1, groups=1),
            BN(inter_channels)
        )
        self.W_x = nn.Sequential(
            Conv(in_channels, inter_channels, kernel_size=3, padding=1, groups=1),
            BN(inter_channels)
        )
        self.psi = nn.Sequential(
            Conv(inter_channels, 1, kernel_size=1),
            BN(1),
            nn.Sigmoid()
        )

        # DFF-style attention path
        self.avg_pool = Pool(1)
        self.conv_atten = nn.Sequential(
            Conv(in_channels * 2, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_reduce = Conv(in_channels * 2, in_channels, kernel_size=1)

        self.conv1 = Conv(in_channels, 1, kernel_size=1)
        self.conv2 = Conv(in_channels, 1, kernel_size=1)

        self.nonlin = nn.Sigmoid()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # LGAG-style gate attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        gate_out = x * psi

        # DFF-style fusion
        concat = torch.cat([g, x], dim=1)
        att = self.conv_atten(self.avg_pool(concat))
        fused = concat * att
        fused = self.conv_reduce(fused)

        att_weight = self.conv1(g) + self.conv2(x)
        att_weight = self.nonlin(att_weight)

        fused = fused * att_weight

        # Final fusion: combine两路输出（可选：加权平均）
        out = 0.5 * gate_out + 0.5 * fused
        return out
