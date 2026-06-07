import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch)

        self.shortcut = nn.Identity()

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)

        out += identity

        return F.relu(out)


class SignalNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=(11,5),
                padding=(5,2),
                bias=False
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool_y = nn.MaxPool2d(
            kernel_size=(2,1),
            stride=(2,1)
        )

        self.layer1 = ResidualBlock(64,64)

        self.layer2 = ResidualBlock(
            64,
            128,
            dilation=1
        )

        self.layer3 = ResidualBlock(
            128,
            256,
            dilation=2
        )

        self.layer4 = ResidualBlock(
            256,
            256,
            dilation=4
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(

            nn.Linear(512,256),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(256,num_classes)
        )

    def forward(self,x):

        x = self.stem(x)

        x = self.pool_y(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.pool_y(x)

        x = self.layer3(x)

        x = self.layer4(x)

        avg = self.avg_pool(x).flatten(1)
        mx = self.max_pool(x).flatten(1)

        x = torch.cat([avg,mx],dim=1)

        return self.classifier(x)