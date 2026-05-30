import torch
import torch.nn as nn
import torch.nn.functional as F

class StridedVisionCNN(nn.Module):
    def __init__(self, num_classes):
        super(StridedVisionCNN, self).__init__()

        # Block 1 (stride=2 cuts the image size in half immediately)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2 (stride=2 cuts it in half again)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 3 (stride=2 cuts it in half a final time)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=64 * 16, out_features=64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # The downsampling happens directly inside the conv layers now
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.spatial_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.dropout(F.relu(self.bn5(self.fc1(x))))
        logits = self.fc2(x)

        return logits