import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, img_d, channels_d, features_d):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels_d, features_d, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(features_d, features_d*3, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            # (1/2 * 1/2 * img_d) * (1/2 * 1/2 * img_d)
            self._block(features_d * 3 *(img_d ** 2 // 16), 128),
            self._block(128, 64),
            self._block(64, 3),
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)

# t = torch.randn(64, 3, 256, 256)
# net = CNN(256, 3, 6)
# net(t)
# print(net(t).shape)