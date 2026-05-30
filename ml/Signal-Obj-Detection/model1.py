import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionCNN(nn.Module):
    def __init__(self, num_classes):
        super(VisionCNN, self).__init__()

        # Block 1: Drastically reduced channels (16 instead of 64)
        # Change in_channels to 1 if you convert images to grayscale
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2 (32 instead of 128)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 3 (64 instead of 256)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Increased dropout to 50% to force the network to rely on multiple features
        self.dropout = nn.Dropout(p=0.5)

        # Adjusted linear layer to match the new channel count (64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.bn5 = nn.BatchNorm1d(32)

        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Global Pooling & Flatten
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)

        # Fully Connected Block
        x = self.dropout(F.relu(self.bn5(self.fc1(x))))

        # Output Logits
        logits = self.fc2(x)

        return logits