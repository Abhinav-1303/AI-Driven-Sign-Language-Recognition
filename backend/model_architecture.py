import torch
import torch.nn as nn
import torch.nn.functional as F

# 🧠 CTRGCN = "Channel-wise Topology Refinement Graph Convolutional Network"
# This is a simple custom version using fully connected layers for sign classification.

class CTRGCN(nn.Module):
    def __init__(self, num_classes=28, input_dim=21*3, hidden_dim=256):
        """
        num_classes: Number of output classes (A-Z + space + delete = 28)
        input_dim: Flattened keypoints size = 21 landmarks * 3 coordinates (x, y, z)
        hidden_dim: Size of the intermediate (hidden) layers
        """
        super(CTRGCN, self).__init__()

        # 🔹 Input layer: Converts raw keypoints into hidden representation
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # (63 → 256)
        self.bn1 = nn.BatchNorm1d(hidden_dim)        # BatchNorm helps stabilize training

        # 🔸 "Graph" convolution layers (simulated with linear layers for now)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)  # 256 → 256
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)  # 256 → 256

        # 🔹 Final output layer: Maps hidden state to class logits (scores)
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # 256 → 28

    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, 63)
        Returns logits for each class (before softmax)
        """
        # 🟦 Ensure input is flattened (e.g., in case keypoints come in as 21×3)
        x = x.view(x.shape[0], -1)

        # 🔹 First layer with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.fc1(x)))

        # 🔸 Simulated GCN layers (using linear + ReLU)
        x = F.relu(self.gcn1(x))
        x = F.relu(self.gcn2(x))

        # 🔹 Output layer: logits for 28 classes
        x = self.fc2(x)

        return x
