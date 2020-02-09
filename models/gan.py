import torch
from torch import nn

from models.generator import Generator
from models.discriminator import Discriminator


class GAN(nn.Module):
    def __init__(self, num_fakes=16, rand_dim=128, size=(32, 32), use_gpu=False):
        super(GAN, self).__init__()

        self.generator = Generator(num_fakes, rand_dim, size, use_gpu)
        self.discriminator = Discriminator((1,) + size)

    def forward(self, x=None, mode='D'):
        if mode == 'D':
            with torch.no_grad():
                fake_imgs = self.generator()
            self.discriminator.freeze(False)
            fake_imgs = fake_imgs.detach()
            imgs = torch.cat([x, fake_imgs], dim=0)
            out = self.discriminator(imgs)
            return out
        else:
            self.discriminator.freeze(True)
            fake_imgs = self.generator()
            out = self.discriminator(fake_imgs)
            return fake_imgs, out
