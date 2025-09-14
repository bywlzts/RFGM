import torch.nn as nn
import torch
import models.archs.arch_util as arch_util
from models.archs.GradMamba import *
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import torch.nn.functional as F
import functools
from models.archs.SS2D_arch import *
import numbers

class GradMamba(nn.Module):
    def __init__(self, nf=64, num_mamba_blocks=2):
        super(GradMamba, self).__init__()

        self.nf = nf
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2d(3 * 2, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, 5)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.Highconv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.grad_mamba_blocks1 = nn.ModuleList([
            GradStateSpaceBlock(nf, d_state=16) for _ in range(num_mamba_blocks)
        ])
        self.grad_mamba_blocks2 = nn.ModuleList([
            GradStateSpaceBlock(nf, d_state=16) for _ in range(num_mamba_blocks)
        ])
        self.grad_mamba_blocks3 = nn.ModuleList([
            GradStateSpaceBlock(nf, d_state=16) for _ in range(num_mamba_blocks)
        ])

        self.gt_gradient_extractor = GradientExtractor()


    def forward(self, x_center, x):
        B, C, H, W = x.shape

        L1_fea_1 = self.lrelu(self.conv_first_1(torch.cat((x_center, x), dim=1)))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))

        mamba_input = self._spatial_to_sequence(L1_fea_2)  # (B, H3*W3, C3)
        B2, C2, H2, W2 = L1_fea_2.shape
        for i, mamba_block in enumerate(self.grad_mamba_blocks1):
            mamba_input = mamba_block(mamba_input, (H2, W2))
        L1_fea_2 = self._sequence_to_spatial(mamba_input, H2, W2)
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        mamba_input = self._spatial_to_sequence(L1_fea_3)  # (B, H3*W3, C3)
        B3, C3, H3, W3 = L1_fea_3.shape
        for i, mamba_block in enumerate(self.grad_mamba_blocks2):
            mamba_input = mamba_block(mamba_input, (H3, W3))
        L1_fea_3_enhanced = self._sequence_to_spatial(mamba_input, H3, W3)


        fea = self.feature_extraction(L1_fea_3)
        fea = self.Highconv(torch.cat([L1_fea_3_enhanced, fea], dim=1))
        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))

        mamba_input = self._spatial_to_sequence(out_noise)  # (B, H3*W3, C3)
        B4, C4, H4, W4 = out_noise.shape
        for i, mamba_block in enumerate(self.grad_mamba_blocks3):
            mamba_input = mamba_block(mamba_input, (H4, W4))
        out_noise = self._sequence_to_spatial(mamba_input, H4, W4)

        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)

        out_noise = out_noise + x

        out_noise = out_noise[:, :, :H, :W]

        return out_noise

    def _spatial_to_sequence(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

    def _sequence_to_spatial(self, x, H, W):
        B, N, C = x.shape
        return x.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

class ImageBlockProcessor:
    def __init__(self, num_blocks=None):
        self.num_blocks = num_blocks or 4

    def _get_block_layout(self, H, W):
        if isinstance(self.num_blocks, tuple):
            num_rows, num_cols = self.num_blocks
        else:
            total_blocks = self.num_blocks
            aspect_ratio = W / H
            num_rows = int(math.sqrt(total_blocks / aspect_ratio))
            num_rows = max(1, num_rows)
            num_cols = total_blocks // num_rows

            while num_rows * num_cols < total_blocks:
                if num_cols * aspect_ratio < num_rows:
                    num_cols += 1
                else:
                    num_rows += 1

        return num_rows, num_cols

    def split_and_stack_blocks(self, image):
        B, C, H, W = image.shape
        device = image.device

        num_rows, num_cols = self._get_block_layout(H, W)

        block_H = H // num_rows
        block_W = W // num_cols

        if block_H % 8 != 0:
            pad_H = 8 - (block_H % 8)
        else:
            pad_H = 0

        if block_W % 8 != 0:
            pad_W = 8 - (block_W % 8)
        else:
            pad_W = 0

        if pad_H > 0 or pad_W > 0:
            image = F.pad(image, (0, pad_W, 0, pad_H), mode='reflect')

        _, _, H_padded, W_padded = image.shape
        block_H = H_padded // num_rows
        block_W = W_padded // num_cols

        blocks = []
        for i in range(num_rows):
            for j in range(num_cols):
                start_H = i * block_H
                end_H = (i + 1) * block_H
                start_W = j * block_W
                end_W = (j + 1) * block_W
                block = image[:, :, start_H:end_H, start_W:end_W]
                blocks.append(block)
        stacked_blocks = torch.cat(blocks, dim=1).to(device)  # (B, C*num_blocks, block_H, block_W)


        block_info = {
            'original_shape': (B, C, H, W),
            'padded_shape': (B, C, H_padded, W_padded),
            'num_blocks': (num_rows, num_cols),
            'block_size': (block_H, block_W),
            'pad_size': (pad_H, pad_W)
        }

        return stacked_blocks, block_info

    def unstack_and_merge_blocks(self, stacked_blocks, block_info):
        B, C, H, W = block_info['original_shape']
        _, _, H_padded, W_padded = block_info['padded_shape']
        num_rows, num_cols = block_info['num_blocks']
        block_H, block_W = block_info['block_size']
        split_blocks = torch.split(stacked_blocks, C, dim=1)
        merged_image = torch.zeros((B, C, H_padded, W_padded),
                                   device=stacked_blocks.device,
                                   dtype=stacked_blocks.dtype)

        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                start_H = i * block_H
                end_H = (i + 1) * block_H
                start_W = j * block_W
                end_W = (j + 1) * block_W
                merged_image[:, :, start_H:end_H, start_W:end_W] = split_blocks[idx]
                idx += 1
        merged_image = merged_image[:, :, :H, :W]

        return merged_image


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class PatchMamba(torch.nn.Module):
    def __init__(self,  input_channels, num_blocks=4, num_mamba_layers=2,  LayerNorm_type='WithBias'):
        super(PatchMamba, self).__init__()
        self.block_processor = ImageBlockProcessor(num_blocks)
        self.num_mamba_layers = num_mamba_layers
        if isinstance(num_blocks, tuple):
            total_blocks = num_blocks[0] * num_blocks[1]
        else:
            total_blocks = num_blocks

        total_channels = input_channels * total_blocks
        self.init_ccnv = nn.Conv2d(6, input_channels, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(total_channels, total_channels, (1, 1))
        self.norm1 = LayerNorm(total_channels, LayerNorm_type)
        self.conv_out = nn.Conv2d(input_channels, 3, 3, 1, 1, bias=True)

        self.hhmamba = nn.ModuleList([
            nn.ModuleList([
                SS2D6(d_model=total_channels),
                PreNorm(total_channels, FeedForward(dim=total_channels))
            ])
            for _ in range(self.num_mamba_layers)
        ])


    def forward(self, x):
        x = self.init_ccnv(x)
        stacked_blocks, block_info = self.block_processor.split_and_stack_blocks(x)
        processed_blocks = self.conv1(stacked_blocks)
        processed_blocks = self.norm1(processed_blocks)

        for (ss2d, ff) in self.hhmamba:
            y = processed_blocks.permute(0, 2, 3, 1)
            processed_blocks = ss2d(y) + processed_blocks.permute(0, 2, 3, 1)
            processed_blocks = ff(processed_blocks) + processed_blocks
            processed_blocks = processed_blocks.permute(0, 3, 1, 2)

        merged_image = self.block_processor.unstack_and_merge_blocks(processed_blocks, block_info)
        merged_image = self.conv_out(merged_image)

        return merged_image
