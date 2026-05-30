import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, x):
        batch_size, _, y_dim, x_dim = x.size()

        # Create Y coordinates (from -1 to 1)
        y_coords = torch.linspace(-1, 1, steps=y_dim, device=x.device)
        y_coords = y_coords.unsqueeze(0).unsqueeze(-1).expand(batch_size, 1, y_dim, x_dim)

        # Create X coordinates (from -1 to 1)
        x_coords = torch.linspace(-1, 1, steps=x_dim, device=x.device)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, y_dim, x_dim)

        # Concatenate coordinates to the original image channels
        return torch.cat([x, y_coords, x_coords], dim=1)


class StridedVisionCNN3(nn.Module):
    def __init__(self, num_classes):
        super(StridedVisionCNN3, self).__init__()

        self.addcoords = AddCoords()

        # BLOCK 1: Reduced from 32 to 16 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(16)

        # BLOCK 2: Reduced from 64 to 32 channels
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn_conv2 = nn.BatchNorm2d(32)

        # BLOCK 3: Reduced from 128 to 64 channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_conv3 = nn.BatchNorm2d(64)

        # BLOCK 4: Reduced from 128 to 64 channels
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn_conv4 = nn.BatchNorm2d(64)

        # BLOCK 5: NEW LAYER added for extra depth
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn_conv5 = nn.BatchNorm2d(128)

        # Keep the 8x8 spatial grid
        self.spatial_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.dropout = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(in_features=128 * 64, out_features=64)
        self.bn_fc1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.addcoords(x)

        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = F.relu(self.bn_conv2(self.conv2(x)))
        x = F.relu(self.bn_conv3(self.conv3(x)))
        x = F.relu(self.bn_conv4(self.conv4(x)))

        # Pass through the new 5th layer
        x = F.relu(self.bn_conv5(self.conv5(x)))

        x = self.spatial_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        logits = self.fc2(x)

        return logits