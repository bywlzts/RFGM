import torch.nn as nn
import torch
import models.archs.arch_util as arch_util
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import torch.nn.functional as F
import functools
import math


def semantic_neighbor(x, index):  #  from MambaIRv2
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


class GradientExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        # Sobel算子
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]
        ]).float().unsqueeze(0).unsqueeze(0))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1], [0, 0, 0], [1, 2, 1]
        ]).float().unsqueeze(0).unsqueeze(0))

        # 拉普拉斯算子
        self.register_buffer('laplacian', torch.tensor([
            [0, -1, 0], [-1, 4, -1], [0, -1, 0]
        ]).float().unsqueeze(0).unsqueeze(0))

        self.channel_fusion = nn.Conv2d(3, 1, 1, bias=False)  # 融合x,y,laplacian

    def forward(self, x):
        if x.size(1) > 1:
            gray = x.mean(dim=1, keepdim=True)
        else:
            gray = x

        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        grad_lap = F.conv2d(gray, self.laplacian, padding=1)

        grad_stack = torch.cat([grad_x, grad_y, grad_lap], dim=1)  # (B, 3, H, W)
        gradient_magnitude = self.channel_fusion(grad_stack)  # (B, 1, H, W)
        gradient_magnitude = torch.abs(gradient_magnitude)

        return gradient_magnitude

class SimplifiedGradientToPriority(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, gradient_map):
        gradient = gradient_map.squeeze(1)  # (B, H, W)
        B, H, W = gradient.shape
        gradient_flat = gradient.view(B, -1)  # (B, H*W)
        grad_min = gradient_flat.min(dim=1, keepdim=True)[0]  # (B, 1)
        grad_max = gradient_flat.max(dim=1, keepdim=True)[0]  # (B, 1)
        gradient_norm = (gradient_flat - grad_min) / (grad_max - grad_min + 1e-8)
        gradient_norm = gradient_norm.view(B, H, W)  # (B, H, W)
        priority_score = self.scale * gradient_norm + self.offset
        priority_score = torch.sigmoid(priority_score)
        return priority_score


class GradStateSpaceBlock(nn.Module):
    def __init__(self, dim, d_state, mlp_ratio=2.0, scale=1.0):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.scale = scale

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gradient_extractor = GradientExtractor()
        self.gradient_to_priority = SimplifiedGradientToPriority()
        self.gradient_mamba = GradientGuidedMamba(dim, d_state, mlp_ratio)

        self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                nn.SiLU(),
                nn.Linear(int(dim * mlp_ratio), dim),
            )

    def forward(self, x, x_size):
        B, N, C = x.shape
        H, W = x_size

        x_spatial = x.permute(0, 2, 1).view(B, C, H, W)
        extracted_gradient = self.gradient_extractor(x_spatial)  # (B, 1, H, W)
        gradient_priority = self.gradient_to_priority(extracted_gradient)  # (B, H, W)
        residual = x
        x = self.norm1(x)
        x = self.gradient_mamba(x, x_size, gradient_priority)
        x = residual + self.scale * x
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.scale * x
        return x

class GradientGuidedMamba(nn.Module):
    def __init__(self, dim, d_state, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state

        self.selectiveScan = GradientGuidedSelectiveScan(
            d_model=hidden, d_state=self.d_state, expand=1
        )

        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

    def forward(self, x, x_size, gradient_score):
        B, n, C = x.shape
        H, W = x_size

        gradient_score_flat = gradient_score.view(B, -1)  # (B, HW)
        x_sort_values, x_sort_indices = torch.sort(gradient_score_flat, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x_spatial = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_proj = self.in_proj(x_spatial)
        x_proj = x_proj * torch.sigmoid(self.CPE(x_proj))

        cc = x_proj.shape[1]
        x_proj = x_proj.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x_proj, x_sort_indices)

        y = self.selectiveScan(semantic_x, gradient_score)
        y = self.out_proj(self.out_norm(y))

        x_output = semantic_neighbor(y, x_sort_indices_reverse)

        return x_output


class GradientGuidedSelectiveScan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 投影层
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.gradient_C_enhancer = nn.Linear(1, self.d_state)
        self.gradient_C_weight = nn.Parameter(torch.tensor(0.1))

        self.feedback_A_weight = nn.Parameter(torch.tensor(0.05))

        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, gradient_score):
        B, L, C = x.shape
        K = 1
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        gradient_flat = gradient_score.view(B, -1, 1)  # (B, L, 1)
        gradient_enhancement = self.gradient_C_enhancer(gradient_flat)  # (B, L, d_state)
        gradient_enhancement = gradient_enhancement.view(B, K, self.d_state, L)

        Cs_enhanced = Cs + self.gradient_C_weight * gradient_enhancement

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs_enhanced = Cs_enhanced.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        # 选择性扫描
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs_enhanced, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float
        return out_y[:, 0]

    def forward(self, x: torch.Tensor, gradient_score):
        y = self.forward_core(x, gradient_score)
        y = y.permute(0, 2, 1).contiguous()
        return y

