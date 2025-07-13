import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRGCN(nn.Module):
    def __init__(self, num_classes=28, input_dim=21*3, hidden_dim=256):
        super(CTRGCN, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Graph convolution layers
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected classifier
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Flatten the input (21 keypoints × 3 coordinates)
        x = x.view(x.shape[0], -1)

        # First layer
        x = F.relu(self.bn1(self.fc1(x)))

        # GCN layers
        x = F.relu(self.gcn1(x))
        x = F.relu(self.gcn2(x))

        # Output layer
        x = self.fc2(x)

        return x