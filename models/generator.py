import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, target_size):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, bias=False)
        self.leaky_relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.up = nn.UpsamplingBilinear2d(size=target_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(self.bn(x))
        x = self.up(x)

        return x


class Generator(nn.Module):
    def __init__(self, num_fakes=16, rand_dim=128, out_size=(32, 32), use_gpu=False):
        super(Generator, self).__init__()

        assert num_fakes > 0
        assert rand_dim > 0
        self.num_fakes = num_fakes
        self.rand_dim = rand_dim
        self.use_gpu = use_gpu

        self.block1 = BasicBlock(4, 32, (out_size[0] // 4 * 1 + 8, out_size[1] // 4 * 1 + 8))
        self.block2 = BasicBlock(32, 64, (out_size[0] // 4 * 2 + 8, out_size[1] // 4 * 2 + 8))
        self.block3 = BasicBlock(64, 128, (out_size[0] // 4 * 3 + 8, out_size[1] // 4 * 3 + 8))
        self.block4 = BasicBlock(128, 128, out_size)
        self.conv = nn.Conv2d(128, 1, kernel_size=3, padding=1)

        self.linear = nn.Linear(rand_dim, 256, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self):
        z = torch.randn(self.num_fakes, self.rand_dim)
        if self.use_gpu:
            z = z.cuda()
        x = self.linear(z)
        x = x.view(self.num_fakes, 4, 8, 8)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        out = torch.sigmoid(x)

        return out
