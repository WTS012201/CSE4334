import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, pad):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k_size, stride, pad)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample, stride):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1, 1, 0),
            ConvBlock(out_channels, out_channels, 3, stride, 1),
            ConvBlock(out_channels, out_channels * 4, 1, 1, 0),
        )
        self.relu = nn.ReLU()
        self.down_sample = down_sample if in_channels != out_channels else None

    def forward(self, x):
        identity = x.clone()
        x = self.layers(x)
        identity = self.down_sample(identity) if self.down_sample else identity
        x += identity
        x = self.relu(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, n_classes, depth):
        super(ResNet152, self).__init__()
        self.in_channels = 64
        self.out_channels = 64

        self.conv1 = ConvBlock(depth, self.out_channels, 7, 2, 3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2, 1)

        self.conv2_x = self._gen_layer(3, self.out_channels, 1)
        self.out_channels *= 2
        self.conv3_x = self._gen_layer(8, self.out_channels, 2)
        self.out_channels *= 2
        self.conv4_x = self._gen_layer(36, self.out_channels, 2)
        self.out_channels *= 2
        self.conv5_x = self._gen_layer(3, self.out_channels, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.layers = nn.Sequential(
            self.conv1(),
            self.bn1(),
            self.relu(),
            self.max_pool(),
            self.layer1(),
            self.layer2(),
            self.layer3(),
            self.layer4(),
            self.avg_pool()
        )
        self.fcl = nn.Linear(2048, n_classes)

    def _gen_layer(self, n_res_blocks, out_channels, stride):
        layers = []
        down_sample = ConvBlock(self.in_channels, out_channels * 4, 1, stride)

        layers.append(
            ResBlock(self.in_channels, out_channels, down_sample, stride)
        )
        self.in_channels = out_channels * 4

        for _ in range(n_res_blocks - 1):
            layers.append(ResBlock(self.in_channels, out_channels, None, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(1)
        x = self.fcl(x)

        return x
