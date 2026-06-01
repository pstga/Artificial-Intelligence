import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_planes = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, ratio=reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResidualBlock(nn.Module):
    # Enforced stride=1 so we control spatial downsampling externally via pooling
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.cbam = CBAMBlock(out_channels, reduction=16)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet_Medium(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomResNet_Medium, self).__init__()

        # Stem: 1 channel, no stride here to preserve full resolution early on
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        # Your brilliant asymmetric pooling (Compress Y, Keep X)
        self.pool_h = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        # Standard symmetric pooling
        self.pool_hw = nn.MaxPool2d(kernel_size=2, stride=2)

        # High capacity, standard ResNet block depths
        self.layer1 = nn.Sequential(ResidualBlock(64, 64),)
        self.layer2 = nn.Sequential(ResidualBlock(64, 128),)
        self.layer3 = nn.Sequential(ResidualBlock(128, 256),
                                    ResidualBlock(256, 256))

        # --- DIMENSIONALITY REDUCTION BOTTLENECK ---
        # 1x1 Conv reduces channels from 256 down to 64.
        # This acts as a cross-channel aggregator, throwing away redundant noise filters.
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.5)

        # 64 (avg) + 64 (max) = 128 features entering FC
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_h(x)  # Shrink Y

        x = self.layer1(x)
        x = self.pool_h(x)  # Shrink Y

        x = self.layer2(x)

        x = self.layer3(x)
        x = self.pool_hw(x)

        # --- Apply the 1x1 Dim Reduction ---
        x = self.dim_reduction(x)

        # Dual pool the compressed channels
        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)
        x = torch.cat([avg, mx], dim=1)

        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        logits = self.fc2(x)

        return logits