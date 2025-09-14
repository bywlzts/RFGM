import torch
import torch.nn as nn
import torch.nn.functional as F

class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc,nc,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)

class ProcessBlock(nn.Module):
    def __init__(self, in_nc, spatial = True):
        super(ProcessBlock,self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
        self.frequency_process = LightTopKFreBlock(nc = in_nc, top_k=in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

    def forward(self, x):
        xori = x
        x_out_four = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = torch.cat([x_spatial,x_out_four],1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_out_four)

        return x_out+xori

class RFGM(nn.Module):
    def __init__(self, nc,n=1):
        super(RFGM,self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            ProcessBlock(nc),
        )
        self.conv1 = ProcessBlock(nc)
        self.conv2 = ProcessBlock(nc)
        self.conv3 = ProcessBlock(nc)
        self.conv4 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2d(nc * 2, 3, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(torch.cat((x2, x3), dim=1))
        x5 = self.conv5(torch.cat((x1, x4), dim=1))
        xout = self.convout(torch.cat((x, x5), dim=1))

        return xout

class LightTopKFreBlock(nn.Module):
    def __init__(self, nc, top_k):
        super(LightTopKFreBlock, self).__init__()

        self.nc = nc
        self.top_k = top_k

        self.conv0 = nn.Conv2d(nc, nc, 1, 1, 0)

        self.process1_mag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.process1_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.process2_pha = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )

        self.magGuideFusion = MagGuidedFusion(channels=nc)
        self.conv_out = nn.Conv2d(nc * 2, nc, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_conv0 = self.conv0(x)
        x_freq = torch.fft.rfft2(x_conv0, norm='backward')
        mag0 = torch.abs(x_freq)
        pha0 = torch.angle(x_freq)
        mag1 = self.process1_mag(mag0)
        pha1 = self.process1_pha(pha0)
        pha_cat = torch.cat((pha0, pha1), dim=1)
        pha_out = self.process2_pha(pha_cat)
        mag_out, mag0_weight = self.magGuideFusion(mag0, mag1)
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        x_out_freq = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out_freq, s=(H, W), norm='backward')
        return x_out


class MagGuidedFusion(nn.Module):
    def __init__(self, channels):

        super(MagGuidedFusion, self).__init__()
        self.channels = channels

        self.expand_conv = nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, mag0, mag1):
        B, C, H, W = mag0.shape

        mag0_flat = mag0.view(B, C, -1)  # (B, C, H*W)
        mag1_flat = mag1.view(B, C, -1)  # (B, C, H*W)

        mag0_norm = F.normalize(mag0_flat, dim=-1)  # (B, C, H*W)
        mag1_norm = F.normalize(mag1_flat, dim=-1)  # (B, C, H*W)

        similarity_matrix = torch.bmm(mag0_norm, mag1_norm.transpose(1, 2))  # (B, C, C)
        similarity_scores = similarity_matrix.mean(dim=-1)  # (B, C)
        top1_indices = torch.argmax(similarity_scores, dim=-1)  # (B,)
        mag0_top1 = torch.stack([mag0[b, top1_indices[b]] for b in range(B)], dim=0).unsqueeze(1)  # (B, 1, H, W)
        mag0_expanded = self.expand_conv(mag0_top1)  # (B, C, H, W)
        mag0_weight = torch.sigmoid(mag0_expanded)  # (B, C, H, W)
        fused_features = mag1 * mag0_weight + mag1  # (B, C, H, W)

        return fused_features, mag0_weight