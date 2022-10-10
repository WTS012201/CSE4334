import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, img_d=640, features_d=6):
        super().__init__()
        self.img_d = img_d
        self.features_d = features_d
        self.seq = nn.Sequential(
            nn.Conv2d(3, self.features_d, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.features_d, self.features_d*3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            # (1/2 * 1/2 * img_d) * (1/2 * 1/2 * img_d)
            self._block(self.features_d * 3 *(self.img_d ** 2 // 16), 128),
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