import functools
import models.archs.arch_util as arch_util
from models.archs.RFGM import *
from models.archs.PatchMamba import *
from models.archs.PatchMamba import *
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, nf=64):
        super(model, self).__init__()
        self.AmpNet = RFGM(8)
        self.nf = nf
        self.mambablock = PatchMamba(input_channels=self.nf, num_blocks=(3, 3), num_mamba_layers=6)
        self.encoder_decoder = GradMamba(nf=nf)

    def forward(self, x):
        _, _, H, W = x.shape
        img_amp_enhanced = self.AmpNet(x)
        x_center = img_amp_enhanced

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), "reflect")
            x = F.pad(x, (0, pad_w, 0, pad_h), "reflect")
        out_noise = self.encoder_decoder(x_center, x)
        out_mamba = self.mambablock(torch.cat([x_center, x],dim=1))
        out_mamba = out_mamba[:, :, :H, :W]
        out_noise = out_mamba + out_noise

        return out_noise, img_amp_enhanced, img_amp_enhanced, img_amp_enhanced