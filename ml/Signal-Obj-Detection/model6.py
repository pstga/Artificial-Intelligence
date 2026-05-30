import torch
import torch.nn as nn
import torch.nn.functional as F


class AddFreqCoords(nn.Module):
    def __init__(self):
        super(AddFreqCoords, self).__init__()

    def forward(self, x):
        batch_size, _, y_dim, x_dim = x.size()
        y_coords = torch.linspace(-1, 1, steps=y_dim, device=x.device)
        y_coords = y_coords.unsqueeze(0).unsqueeze(-1).expand(batch_size, 1, y_dim, x_dim)

        return torch.cat([x, y_coords], dim=1)


class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, x):
        batch_size, _, y_dim, x_dim = x.size()
        y_coords = torch.linspace(-1, 1, steps=y_dim, device=x.device)
        y_coords = y_coords.unsqueeze(0).unsqueeze(-1).expand(batch_size, 1, y_dim, x_dim)
        x_coords = torch.linspace(-1, 1, steps=x_dim, device=x.device)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, y_dim, x_dim)
        return torch.cat([x, y_coords, x_coords], dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        reduced_channels = max(1, channels // reduction)

        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction=16)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out



class CustomResNet_Large(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet_Large, self).__init__()

        self.addcoords = AddFreqCoords()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool_h = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.pool_hw = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = ResidualBlock(in_channels=32, out_channels=64, stride=1)
        self.layer2 = ResidualBlock(in_channels=64, out_channels=128, stride=1)
        self.layer3 = ResidualBlock(in_channels=128, out_channels=256, stride=1)
        self.layer4 = ResidualBlock(in_channels=256, out_channels=256, stride=1)

        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.addcoords(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool_h(x)

        x = self.layer1(x)
        x = self.pool_h(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool_hw(x)


        x = self.layer4(x)
        x = self.pool_h(x)

        x = self.spatial_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(F.relu(self.bn_fc(self.fc1(x))))
        logits = self.fc2(x)

        return logits.squeeze(1)
