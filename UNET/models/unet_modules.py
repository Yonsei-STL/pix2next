import os
import sys
import math
# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.common_modules import *


import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        # Compute mean and standard deviation along the output channel axis
        mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
        std = self.weight.std(dim=[1, 2, 3], keepdim=True)
        
        # Standardize the weights
        weight = (self.weight - mean) / (std + 1e-5)
        
        # Perform convolution with standardized weights
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
class Residual(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 num_groups: int):
        super().__init__()

        self.proj = WeightStandardizedConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(num_features=out_ch)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 num_groups: int):
        super().__init__()

        self.block1 = Residual(in_ch, out_ch, num_groups)
        self.block2 = Residual(out_ch, out_ch, num_groups)

        if in_ch == out_ch:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self,
                latent: torch.Tensor):
        
        h = self.block1(latent)
        h = self.block2(h)
        
        return h + self.residual_layer(latent)

class SDResidualBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 num_groups: int):
        super().__init__()
        """
        UNet residual block

        Args:
        in_ch (int): channels of input images
        out_ch (int): channels of output images
        d_embedded_time (int): d_time * 4 (Change to 2 * out_ch to get scale and shift)
        num_groups (int): the number of groups
        """
        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.conv_feature = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.norm_merged = nn.GroupNorm(num_groups, out_ch)
        self.conv_merged = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        if in_ch == out_ch:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,
                latent: torch.Tensor):
        """
        Args:
        latent: (batch, in_ch, height, width)
        embedded_time: (batch, d_embedded_time)

        Returns:
        merged (skip connection): (batch, out_ch, height, width)
        """
        feature = self.norm(latent)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        feature = self.norm_merged(feature)
        feature = F.silu(feature)
        feature = self.conv_merged(feature)

        return feature + self.residual_layer(latent)

class CrossAttentionBlock(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 d_condition: int, 
                 num_groups: int,
                 num_condition_groups: int):
        super().__init__()
        '''
        Unet attention block
        in_ch = n_heads * d_embed 

         Args:
          n_heads (int): the number of heads
          d_embed (int): dimension of attention 
          d_condition (int): dimension of condition 
          num_groups (int): the number of groups
        '''
        if d_condition % n_heads != 0:
            self.cond_change = True
            self.condition_conv = nn.Conv2d(d_condition, d_condition * n_heads, 1, 1, 0)
            self.num_condition_groups = d_condition
        else:
            self.cond_change = False
            self.num_condition_groups = num_condition_groups

        self.in_ch = n_heads * d_embed
        self.in_ch_c = d_condition * n_heads

        self.group_norm = nn.GroupNorm(num_groups, self.in_ch, eps=1e-6)
        self.conv_input = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

        self.group_norm_c = nn.GroupNorm(self.num_condition_groups, self.in_ch_c, eps=1e-6)
        self.conv_input_c = nn.Conv2d(self.in_ch_c, self.in_ch_c, kernel_size=3, padding=1)

        #self.layer_norm_1 = nn.LayerNorm(self.in_ch)
        self.attention_1 = MultiHeadCrossAttention(self.in_ch, n_heads, d_condition, in_proj_bias=False)
        
        self.linear_geglu_1 = nn.Linear(self.in_ch, self.in_ch * 4)
        self.linear_geglu_2 = nn.Linear(self.in_ch * 4, self.in_ch)

        self.conv_output = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

    def forward(self,
                x: torch.Tensor, 
                condition: torch.Tensor):
        '''
        Args:
         x: (batch, in_ch, height, width)
         condition: (batch, in_ch, height, width)

        return:
         x + residue: (batch, in_ch, height, width)
        '''

        if self.cond_change == True:
            condition = self.condition_conv(condition)

        residue_long = x

        # x: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        x = self.conv_input(x)
        x = self.group_norm(x)

        # condition: (batch, in_ch_c, height, width) -> (batch, in_ch_c, height, width)
        condition = self.conv_input_c(condition)
        condition = self.group_norm_c(condition)
        
        batch, in_ch, height, width = x.size()
        batch_c, in_ch_c, height_c, width_c = condition.size()

        # x: (batch, in_ch, height, width) -> (batch, height*width, in_ch)
        x = x.view(batch, in_ch, height*width).contiguous()
        x = x.transpose(-1, -2).contiguous()

        # condition: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        condition = condition.view(batch_c, in_ch_c, height_c*width_c).contiguous()
        condition = condition.transpose(-1, -2).contiguous()

        # Normalization + self attention and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch)
        #x = self.layer_norm_1(x)
        x = self.attention_1(x, condition)
        x += residue_short

        # Normalization + FFN with GeGLU and skip connection
        x = self.linear_geglu_1(x)
        x = self.linear_geglu_2(x)

        # x: (batch, seq_len, in_ch) -> (batch, in_ch, height, width)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(batch, in_ch, height, width).contiguous()

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class SDAttentionBlock(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 d_condition: int, 
                 num_groups: int,
                 num_condition_groups: int):
        super().__init__()
        '''
        Unet attention block
        in_ch = n_heads * d_embed 

         Args:
          n_heads (int): the number of heads
          d_embed (int): dimension of attention 
          d_condition (int): dimension of condition 
          num_groups (int): the number of groups
        '''
        if d_condition % n_heads != 0:
            self.cond_change = True
            self.condition_conv = nn.Conv2d(d_condition, d_condition * n_heads, 1, 1, 0)
            self.num_condition_groups = d_condition
            self.in_ch_c = d_condition * n_heads

        else:
            self.cond_change = False
            self.num_condition_groups = num_condition_groups
            self.in_ch_c = d_condition

        self.in_ch = n_heads * d_embed

        self.group_norm = nn.GroupNorm(num_groups, self.in_ch, eps=1e-6)
        self.conv_input = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

        self.group_norm_c = nn.GroupNorm(self.num_condition_groups, self.in_ch_c, eps=1e-6)
        self.conv_input_c = nn.Conv2d(self.in_ch_c, self.in_ch_c, kernel_size=3, padding=1)

        self.layer_norm_1 = nn.LayerNorm(self.in_ch)
        self.attention_1 = MultiHeadSelfAttention(self.in_ch, n_heads, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(self.in_ch)
        self.attention_2 = MultiHeadCrossAttention(self.in_ch, n_heads, d_condition, in_proj_bias=False)
        
        self.layer_norm_3 = nn.LayerNorm(self.in_ch)

        self.linear_geglu_1 = nn.Linear(self.in_ch, 4 * self.in_ch * 2)
        self.linear_geglu_2 = nn.Linear(self.in_ch * 4, self.in_ch)

        self.conv_output = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

    def forward(self,
                x: torch.Tensor, 
                condition: torch.Tensor):
        '''
        Args:
         x: (batch, in_ch, height, width)
         condition: (batch, in_ch, height, width)

        return:
         x + residue: (batch, in_ch, height, width)
        '''

        if self.cond_change == True:
            condition = self.condition_conv(condition)

        residue_long = x

        # x: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        x = self.conv_input(x)
        x = self.group_norm(x)

        # condition: (batch, in_ch_c, height, width) -> (batch, in_ch_c, height, width)
        condition = self.conv_input_c(condition)
        condition = self.group_norm_c(condition)
        
        batch, in_ch, height, width = x.size()
        batch_c, in_ch_c, height_c, width_c = condition.size()

        # x: (batch, in_ch, height, width) -> (batch, height*width, in_ch)
        x = x.view(batch, in_ch, height*width).contiguous()
        x = x.transpose(-1, -2).contiguous()

        # condition: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        condition = condition.view(batch_c, in_ch_c, height_c*width_c).contiguous()
        condition = condition.transpose(-1, -2).contiguous()

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
        x = self.attention_2(x, condition)
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
        x = x.view(batch, in_ch, height, width).contiguous()

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long
    
class UnetOutputLayer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 num_groups: int):
        super().__init__()

        #self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_ch)
        #self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        '''
        Args:
         x: (batch, in_channels, height, width)

        return:
         x: (batch, out_channels, height, width)
        '''

        #x = self.group_norm(x)
        #x = self.act(x)
        x = self.conv(x)
        return x

class Switch(nn.Sequential):
    def forward(self, x, condition = None):
        for layer in self:
            if isinstance(layer, (SDResidualBlock, ResidualBlock)):
                x = layer(x)
            elif isinstance(layer, (SDAttentionBlock, CrossAttentionBlock)):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x

