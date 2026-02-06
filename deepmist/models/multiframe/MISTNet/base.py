import torch
import torch.nn as nn


def make_layer(block, in_channels, out_channels, num_blocks=1):
    layers = [block(in_channels, out_channels)]
    for _ in range(num_blocks - 1):
        layers.append(block(out_channels, out_channels))

    return nn.Sequential(*layers)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7.'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, block, num_blocks, nb_filter):
        super(ResNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder_1 = make_layer(block, in_channels, nb_filter[0])
        self.encoder_2 = make_layer(block, nb_filter[0], nb_filter[1], num_blocks=num_blocks[0])
        self.encoder_3 = make_layer(block, nb_filter[1], nb_filter[2], num_blocks=num_blocks[1])
        self.encoder_4 = make_layer(block, nb_filter[2], nb_filter[3], num_blocks=num_blocks[2])

    def forward(self, x):
        feat_1 = self.encoder_1(x)  # [1, 8, 384, 384]
        feat_2 = self.encoder_2(self.pool(feat_1))  # [1, 16, 192, 192]
        feat_3 = self.encoder_3(self.pool(feat_2))  # [1, 32, 96, 96]
        feat_4 = self.encoder_4(self.pool(feat_3))  # [1, 64, 48, 48]

        return feat_1, feat_2, feat_3, feat_4
