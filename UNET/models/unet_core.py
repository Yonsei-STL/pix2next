import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UNET.models.unet_modules import *
from common.intern_image import *

class UnetCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.skip_condition_type = config['model']['unet']['skip_connection_type']
        self.d_condition = config['model']['unet']['condition']['d_condition']

        self.num_groups = config['model']['unet']['num_groups']
        self.num_condition_groups = config['model']['unet']['num_condition_groups']

        self.target_ch = config['model']['unet']['target_ch']

        # Initialize blocks
        self.encoder = self._initialize_block(
            config['model']['unet']['encoder_block_types'],
            config['model']['unet']['encoder_block_args']
        )

        self.bottle_neck = self._initialize_block(
            config['model']['unet']['bottle_neck_block_types'],
            config['model']['unet']['bottle_neck_block_args']
        )

        self.decoder = self._initialize_block(
            config['model']['unet']['decoder_block_types'],
            config['model']['unet']['decoder_block_args']
        )

        self.internimage = InternImage(seq=True)
        #ViT(image_size = (32, 32),patch_size = (1, 1), dim = 256, depth = 6, heads = 16, mlp_dim = 512,channels = 1024)

        self.output_res_block = self._initialize_output_res_block(config)
        self.output_block = self._initialize_output_block(config)

        if self.skip_condition_type == 'concat':
            print('skip type: concat')
            self.forward = self.forward_without_contour_concat
        elif self.skip_condition_type == 'add':
            print('skip type: add')
            self.forward = self.forward_without_contour_add
            
    def _initialize_block(self, block_types, block_args):
        block = nn.ModuleList()
        self._build_blocks(block, block_types, block_args)
        return block

    def _build_blocks(self, module_list, block_types, block_args):
        for block_type, args in zip(block_types, block_args):
            if isinstance(block_type, list):
                sub_blocks = [self._build_sub_block(sub_blocks_type, *sub_args) for sub_blocks_type, sub_args in zip(block_type, args)]
                module_list.append(Switch(*sub_blocks))
            else:
                module_list.append(Switch(self._build_sub_block(block_type, *args)))

    def _build_sub_block(self, sub_blocks_type, *args):
        block_mapping = {
            'conv2d':             lambda: Conv2d(*args),
            'residual':           lambda: ResidualBlock(*args, self.num_groups),
            'sd_residual':        lambda: SDResidualBlock(*args, self.num_groups),
            'sd_attention':       lambda: SDAttentionBlock(*args, self.num_groups, self.num_condition_groups),
            'cross_attention':    lambda: CrossAttentionBlock(*args, self.d_condition, self.num_groups, self.num_condition_groups),
            'self_attention':     lambda: SelfAttentionBlock(*args, self.num_groups),
            'downsample':         lambda: DownSample(*args),
            'downsample_depth':   lambda: DownSampleDepth(*args),
            'upsample_transpose': lambda: UpSampleTranspose(*args),
            'upsample':           lambda: UpSample(*args),
            'silu':               lambda: SiLU(),
            'swish':              lambda: Swish(),
            'internimage':        lambda: InternImage(*args),
            'pre_group':          lambda: PreGroup(*args),
        }
        return block_mapping[sub_blocks_type]()

    def _initialize_output_res_block(self, config):
        # Get the output channel of the last decoder layer
        last_layer_type = config['model']['unet']['decoder_block_types'][-1]
        last_layer_args = config['model']['unet']['decoder_block_args'][-1]

        first_layer_args = config['model']['unet']['encoder_block_args'][0][0]
        d_input = first_layer_args[0]

        if isinstance(last_layer_type, list):
            last_layer_type = last_layer_type[-1]
            last_layer_args = last_layer_args[-1]

        if last_layer_type in ['residual', 'sd_residual', 'conv2d']:
            d_output = last_layer_args[1]
        elif last_layer_type in ['upsample_transpose', 'upsample', 'downsample', 'downsample_depth']:
            d_output = last_layer_args[0]
        else:
            raise ValueError(f"Unsupported layer type: {last_layer_type}")

        return ResidualBlock(d_output + d_input, d_output, self.num_groups)

    def _initialize_output_block(self, config):
        # Get the output channel of the last decoder layer
        last_layer_type = config['model']['unet']['decoder_block_types'][-1]
        last_layer_args = config['model']['unet']['decoder_block_args'][-1]

        if isinstance(last_layer_type, list):
            last_layer_type = last_layer_type[-1]
            last_layer_args = last_layer_args[-1]

        if last_layer_type in ['residual', 'sd_residual', 'conv2d']:
            d_output = last_layer_args[1]
        elif last_layer_type in ['upsample_transpose', 'upsample', 'downsample', 'downsample_depth']:
            d_output = last_layer_args[0]
        else:
            raise ValueError(f"Unsupported layer type: {last_layer_type}")

        return UnetOutputLayer(d_output, self.target_ch, self.num_groups)

    def _get_input_channels(self, args):
        if isinstance(args[0], list):
            return args[0][0]
        else:
            return args[0]

    def _get_output_channels(self, args):
        if isinstance(args[-1], list):
            if len(args[-1]) == 1: # if down or up sample
                return args[-1][-1] 
            else:
                return args[-1][1]
        else:
            return args[0]

    def forward_without_contour_add(self, x, condition=None):    
        skip_connections = []
        residue = x

        for layer in self.encoder:
            x = layer(x, condition)
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                skip_connections.append(x)

        for layer in self.bottle_neck:
            x = layer(x, condition)

        for layer in self.decoder:
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x += skip_connections.pop()
            x = layer(x, condition)

        x = self.output_res_block(torch.cat([x, residue], dim=1), embedded_time)
        x = self.output_block(x)
        return x
    
    def forward_without_contour_concat(self, x, condition=None):
        skip_connections = []
        residue = x

        condition = self.internimage(x)
    
        for layer in self.encoder:
            x = layer(x, condition[0])
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                skip_connections.append(x)

        for layer in self.bottle_neck:
            x = layer(x, condition[1])

        for layer in self.decoder:
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, condition[3])

        x = self.output_res_block(torch.cat([x, residue], dim=1))
        x = self.output_block(x)
        return x
