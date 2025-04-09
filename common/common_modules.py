import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops.misc import Permute
from torchvision.ops import StochasticDepth

from torch.optim.lr_scheduler import OneCycleLR
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def create_onecycle_scheduler(optimizer,
                              steps_per_epoch, 
                              epochs, 
                              max_lr=0.01, 
                              pct_start=0.3, 
                              anneal_strategy='cos', 
                              div_factor=25.0, 
                              final_div_factor=10000.0):
    """
    OneCycleLR 스케줄러를 생성하는 함수

    :param optimizer: 최적화기 (torch.optim.Optimizer)
    :param steps_per_epoch: 에포크 당 스텝 수 (배치 수)
    :param epochs: 총 학습 에포크 수
    :param max_lr: 최대 학습률
    :param pct_start: 학습률이 증가하는 초기 비율
    :param anneal_strategy: 학습률 감소 전략 ('cos' or 'linear')
    :param div_factor: 초기 학습률은 max_lr / div_factor
    :param final_div_factor: 최종 학습률은 initial_lr / final_div_factor
    :return: OneCycleLR 스케줄러
    """
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs, 
        pct_start=pct_start, 
        anneal_strategy=anneal_strategy, 
        div_factor=div_factor, 
        final_div_factor=final_div_factor
    )
    return scheduler

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 in_proj_bias=True, 
                 out_proj_bias=True):
        super().__init__()
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"
        '''
        Just multi head self attention
        in_ch = n_heads * d_embed 
        
        Args:
         n_heads (int): the number of heads
         d_embed (int): dimension of attention 
        '''

        # this combines the Q, K, V matrices into one matrix
        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)

        # this one represents the O matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads
        self.scale   = self.d_heads ** 0.5
        
    def forward(self, 
                x: torch.Tensor):
        '''
         Args:
          x: (batch, seq_len, d_embed)
        
         return: 
          output: (batch, seq_len, d_embed)
        '''

        batch, seq_len, d_embed = x.size()
        inter_size = (batch, seq_len, self.n_heads, self.d_heads)

        # Q, K, V: (batch, seq_len, d_embed)
        Q, K, V = self.in_proj(x).chunk(3, dim=-1)

        # Q, K, V: (batch, seq_len, d_embed) -> (batch, heads, seq_len, d_heads)
        Q = Q.view(inter_size).transpose(1, 2).contiguous()
        K = K.view(inter_size).transpose(1, 2).contiguous()
        V = V.view(inter_size).transpose(1, 2).contiguous()

        # weight: (batch, heads, seq_len, seq_len)
        weight = torch.matmul(Q, K.transpose(-1, -2))
        weight /= self.scale
        weight = torch.softmax(weight, dim=-1)

        # output: (batch, seq_len, d_embed)
        output = torch.matmul(weight, V).transpose(1, 2).contiguous().view(batch, seq_len, d_embed)
        output = self.out_proj(output)
        return output
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 d_condition: int, 
                 in_proj_bias=True, 
                 out_proj_bias=True):
        super().__init__()
        assert d_embed % n_heads == 0, "d_embed must be divisible by n_heads"

        '''
        Just cross attention with conditional vector

         Args:
          n_heads (int): the number of heads
          d_embed (int): dimension of attention 
          d_condition (int): dimension of condition 
        '''

        self.Q = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.K = nn.Linear(d_condition, d_embed, bias=in_proj_bias) # d_condition -> d * n
        self.V = nn.Linear(d_condition, d_embed, bias=in_proj_bias) # d_condition -> d * n
        self.O = nn.Linear(d_embed, d_embed, bias=in_proj_bias)

        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads
        self.scale   = self.d_heads ** 0.5

    def forward(self, 
                x: torch.Tensor, 
                condition: torch.Tensor):
        '''
         Args:
          x (latent): (batch, seq_len_q, d_q)
          y (condition): (batch, seq_len_kv, d_kv)

         return:
          output: (batch, seq_len_q, d_q)
        '''
        latents_shape = x.size()

        # divide each embedding of Q into multiple heads such that d_heads * n_heads = d_q
        inter_shape = (latents_shape[0], -1, self.n_heads, self.d_heads)

        # Q: (batch, seq_len_q, d_q) -> (batch, seq_len_q, d_q) 
        # K, V: (batch, seq_len_kv, d_kv) -> (batch, seq_len_kv, d_kv)
        Q = self.Q(x)
        K = self.K(condition)
        V = self.V(condition)

        # Q: (batch, seq_len_q, d_q) -> (batch, heads, seq_len_q, d_heads)
        # K, V: (batch, seq_len_kv, d_kv) -> (batch, heads, seq_len_kv, d_heads)
        Q = Q.view(inter_shape).transpose(1, 2).contiguous()
        K = K.view(inter_shape).transpose(1, 2).contiguous()
        V = V.view(inter_shape).transpose(1, 2).contiguous()

        # weight: (batch, heads, seq_len_q, seq_len_kv)
        weight = torch.matmul(Q, K.transpose(-1, -2))
        weight /= self.scale
        weight = torch.softmax(weight, dim=-1)

        # output: (batch, heads, seq_len_q, seq_len_kv) -> (batch, seq_len_q, d_q)
        output = torch.matmul(weight, V).contiguous().view(latents_shape)
        output = self.O(output)

        return output

class SelfAttentionBlock(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 num_groups: int):
        super().__init__()
        '''
        attention block (stable diffusion style)
        in_ch = n_heads * d_embed 

         Args:
          n_heads (int): the number of heads
          d_embed (int): dimension of attention 
          d_condition (int): dimension of condition 
          num_groups (int): the number of groups
        '''

        self.in_ch = n_heads * d_embed
        
        self.group_norm = nn.GroupNorm(num_groups, self.in_ch, eps=1e-6)
        self.conv_input = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=1, padding=0)

        self.layer_norm_1 = nn.LayerNorm(self.in_ch)
        self.attention_1 = MultiHeadSelfAttention(self.in_ch, n_heads, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(self.in_ch)
        self.attention_2 = MultiHeadSelfAttention(self.in_ch, n_heads, in_proj_bias=False)
        
        self.layer_norm_3 = nn.LayerNorm(self.in_ch)

        self.linear_geglu_1 = nn.Linear(self.in_ch, 4 * self.in_ch * 2)
        self.linear_geglu_2 = nn.Linear(self.in_ch * 4, self.in_ch)

        self.conv_output = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=1, padding=0)

    def forward(self, 
                x: torch.Tensor):
        '''
        Args:
         x: (batch, in_ch, height, width)

        return:
         x + residue: (batch, in_ch, height, width)
        '''

        residue_long = x

        # x: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        x = self.group_norm(x)
        x = self.conv_input(x)

        batch, in_ch, height, width = x.size()

        # x: (batch, in_ch, height, width) -> (batch, height*width, in_ch)
        x = x.view(batch, in_ch, height*width)
        x = x.transpose(-1, -2).contiguous()

        # Normalization + self attention and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch)
        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Normalization + cross attention and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch)
        x = self.layer_norm_2(x)
        x = self.attention_2(x)
        x += residue_short

        # Normalization + FFN with GeGLU and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch * 4), (batch, seq_len, in_ch * 4)
        x = self.layer_norm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        # Element-wise product
        x = x * F.gelu(gate)

        # x: (batch, seq_len, in_ch * 4) -> (batch, seq_len, in_ch)
        x = self.linear_geglu_2(x)
        x += residue_short

        # x: (batch, seq_len, in_ch) -> (batch, in_ch, height, width)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(batch, in_ch, height, width)

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long
    
class UpSampleTranspose(nn.Module):
    def __init__(self, 
                 in_ch: int):
        super().__init__()
        '''
        Just 2x upsample
         Args:
          in_ch (int): channels of input images
        '''
        self.upsample_conv = nn.ConvTranspose2d(in_ch, in_ch, 4, 2, 1)

    def forward(self, 
                x: torch.Tensor):
        '''
         Args:
          x: (batch, in_ch, height, width)

         return:
          x: (batch, in_ch, height*2, width*2) 
        '''
        return self.upsample_conv(x)
    
class UpSample(nn.Module):
    def __init__(self, 
                 in_ch: int):
        super().__init__()
        '''
        Just 2x upsample
         Args:
          in_ch (int): channels of input images
        '''
        self.upsample = nn.Upsample(scale_factor=2)
        self.upsample_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1)
    
    def forward(self, 
                x: torch.Tensor):
        '''
         Args:
          x: (batch, in_ch, height, width)

         return:
          x: (batch, in_ch, height*2, width*2) 
        '''
        x = self.upsample(x)
        return self.upsample_conv(x)

class DownSample(nn.Module):
    def __init__(self, 
                 in_ch: int):
        super().__init__()
        '''
        Just 2x downsample
         Args:
          in_ch (int): channels of input images
        '''
        self.downsample_conv = nn.Conv2d(in_ch, in_ch, 3, 2, 1)

    def forward(self, 
                x: torch.Tensor):
        '''
         Args:
          x: (batch, in_ch, height, width)

         return:
          x: (batch, in_ch, height/2, width/2)
        '''
        return self.downsample_conv(x)
    
class SpaceToDepth(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def forward(self, x):
        """
        Downscale method that uses the depth dimension to
        downscale the spatial dimensions.
        Args:
            x (torch.Tensor): a tensor to downscale.
            size (int): the scaling factor.

        Returns:
            (torch.Tensor): new spatial downscaled tensor.
        """
        b, c, h, w = x.shape
        out_h = h // self.size
        out_w = w // self.size
        out_c = c * (self.size * self.size)
        x = x.reshape(b, c, out_h, self.size, out_w, self.size)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.reshape(b, out_c, out_h, out_w)
        return x
    
class DownSampleDepth(nn.Module):
    def __init__(self,
                 in_ch: int):
        super().__init__()

        self.spacetodepth = SpaceToDepth()
        self.downsample_conv = nn.Conv2d(4 * in_ch, in_ch, 3, 1, 1)

    def forward(self, x):
        x = self.spacetodepth(x)
        return self.downsample_conv(x)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.silu(x)

class Swish(nn.Module):
    """
    Swish activation function first introduced in the paper https://arxiv.org/abs/1710.05941v2
    It has shown to be working better in many datasets as compares to ReLU.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

#### swin ####
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to(self.logit_scale.device))).exp()

        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.to(x.device)
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_mask = self.create_mask(H, W)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self, H, W):
        flops = 0
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x, H // 2, W // 2

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def flops(self, H, W):
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W

    def extra_repr(self) -> str:
        return f"dim={self.dim}, depth={self.depth}"

    def flops(self, H, W):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(H, W)
        if self.downsample is not None:
            flops += self.downsample.flops(H, W)
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H // self.patch_size[0], W // self.patch_size[1]

    def flops(self, H, W):
        flops = (H // self.patch_size[0]) * (W // self.patch_size[1]) * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += (H // self.patch_size[0]) * (W // self.patch_size[1]) * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, 
                 in_channels: int,
                 embed_dim: int,
                 patch_size: list,
                 depths: list,
                 num_heads: list,
                 window_size: int,
                 mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 1000, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        #seq_out=[]
        x, H, W = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
            #B, L, C = x.shape
            #H_=W_=int(np.sqrt(L))
            #x_= x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            #seq_out.append(x_)
            

        ### modified head ###
        ### do not flatten ###
        B, L, C = x.shape
        H = W = int(np.sqrt(L))  # Assuming H == W for simplicity; if not, you need to know H and W separately
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.patch_embed.img_size[0], self.patch_embed.img_size[1]
        flops += self.patch_embed.flops(H, W)
        for i, layer in enumerate(self.layers):
            flops += layer.flops(H // (2 ** (i + 1)), W // (2 ** (i + 1)))
        flops += self.num_features * H * W // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.image_height = image_height
        self.image_width = image_width
        self.channels = channels
        self.dim=dim

        self.dow = DownSample(128)
        self.dow2=DownSample(128)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
       
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_image = nn.Sequential(
            nn.Linear(dim,dim),
            nn.LayerNorm(dim),
            
        )

    def forward(self, img):
        #print(f"Input shape: {img.shape}")
        #x = self.dow(img)
        
      

        x = self.to_patch_embedding(img)
        #print(f"After patch embedding: {x.shape}")
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        #print(f"After transformer: {x.shape}")

        x = self.to_image(x)
        #print(f"Output shape: {x.shape}")
        x = x.reshape(-1, 1024, 16, 16)
        #print(f"Output shape: {x.shape}")
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
