import torch
from torch import nn
import torchvision
from torchvision import models

class BuildingClassificationModel(nn.Module):

    def __init__(self, dropout=0.15, negative_slope=0.1):
        super().__init__()

        mobile = models.mobilenet_v2(weights='DEFAULT')
        self.cnn = nn.Sequential(mobile.features[:12])

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(96,96),
            nn.LayerNorm(96),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),

            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),

            nn.Linear(32,4)
        )

    def forward(self, x):
        encoded_output = self.cnn(x)
        logits = self.fc(encoded_output)
        return logits