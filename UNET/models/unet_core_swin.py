import sys
import os

import kornia

import torch
import torch.nn as nn
import torch.nn.functional as F

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UNET.models.unet_modules import *

class UnetCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.skip_condition_type = config['model']['unet']['skip_connection_type']
        self.contour_cond = config['model']['unet']['condition']['contour_condition']
                
        self.d_condition = config['model']['unet']['condition']['d_condition']
        self.d_embedded_time = config['model']['unet']['time_embedding']['d_time'] * 4

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

        self.swin = SwinTransformerV2(*config['model']['unet']['swin_block_args'][0])

        self.output_res_block = self._initialize_output_res_block(config)
        self.output_block = self._initialize_output_block(config)

        # Create Zero Convs for each encoder and decoder block if contour condition is used
        if self.contour_cond:
            self.contour_extractor = kornia.filters.Canny()

            self.encoder_zero_convs_in = nn.ModuleList()
            self.encoder_zero_convs_out = nn.ModuleList()

            self.decoder_zero_convs_in = nn.ModuleList()
            self.decoder_zero_convs_out = nn.ModuleList()

            # Canny must be 1 channel
            encoder_out_channels = []
            for block_type, args in zip(config['model']['unet']['encoder_block_types'], config['model']['unet']['encoder_block_args']):
                if block_type not in ['downsample_depth', 'downsample']:
                    encoder_in_channels = self._get_input_channels(args)
                    self.encoder_zero_convs_in.append(ZeroConv(1, encoder_in_channels))

                    encoder_out_channels.append(self._get_output_channels(args))
                    self.encoder_zero_convs_out.append(ZeroConv(encoder_out_channels[-1], encoder_out_channels[-1]))

            decoder_zero_convs_index = 0
            for block_type, args in zip(config['model']['unet']['decoder_block_types'], config['model']['unet']['decoder_block_args']):
                if block_type != 'upsample':
                    decoder_in_channels = self._get_input_channels(args)
                    if decoder_zero_convs_index < len(encoder_out_channels):
                        decoder_in_channels -= encoder_out_channels[-decoder_zero_convs_index-1]
                    self.decoder_zero_convs_in.append(ZeroConv(1, decoder_in_channels))
                    decoder_out_channels = self._get_output_channels(args)
                    self.decoder_zero_convs_out.append(ZeroConv(decoder_out_channels, decoder_out_channels))
                    decoder_zero_convs_index += 1

        # Define the forward method based on the configuration
        if self.contour_cond and self.skip_condition_type == 'concat':
            print('with contour / skip type: concat')
            self.forward = self.forward_with_contour_concat
        elif self.contour_cond and self.skip_condition_type == 'add':
            print('with contour / skip type: add')
            self.forward = self.forward_with_contour_add
        elif self.contour_cond == False and self.skip_condition_type == 'concat':
            print('without contour / skip type: concat')
            self.forward = self.forward_without_contour_concat
        elif self.contour_cond == False and self.skip_condition_type == 'add':
            print('without contour / skip type: add')
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
            'sd_attention':       lambda: SDAttentionBlock(*args, self.d_condition, self.num_groups, self.num_condition_groups),
            'cross_attention':    lambda: CrossAttentionBlock(*args, self.d_condition, self.num_groups, self.num_condition_groups),
            'self_attention':     lambda: SelfAttentionBlock(*args, self.num_groups),
            'downsample':         lambda: DownSample(*args),
            'downsample_depth':   lambda: DownSampleDepth(*args),
            'upsample_transpose': lambda: UpSampleTranspose(*args),
            'upsample':           lambda: UpSample(*args),
            'non_local':          lambda: NonLocalBlock(*args, self.num_groups),
            'silu':               lambda: SiLU(),
            'swish':              lambda: Swish(),
            'swin':               lambda: SwinTransformerV2(*args),
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

        condition = self.swin(x)

        for layer in self.encoder:
            x = layer(x, condition) # seq_out[0]
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                skip_connections.append(x)

        for layer in self.bottle_neck:
            x = layer(x, condition)

        for layer in self.decoder:
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, condition)

        x = self.output_res_block(torch.cat([x, residue], dim=1))
        x = self.output_block(x)
        return x

    # need to rewrite
    def forward_with_contour_add(self, x, condition=None):
        skip_connections = []
        residue = x
        
        _, contour = self.contour_extractor(condition)
        
        encoder_conv_index = 0
        for i, layer in enumerate(self.encoder):
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                embedded_contour = self.encoder_zero_convs_in[encoder_conv_index](contour)

                if x.shape[2:] != embedded_contour.shape[2:]:
                    embedded_contour = F.interpolate(embedded_contour, size=x.shape[2:], mode='bilinear', align_corners=False)

                x += embedded_contour

            x = layer(x, condition)
            
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x = self.encoder_zero_convs_out[encoder_conv_index](x)
                skip_connections.append(x)
                encoder_conv_index += 1

        for layer in self.bottle_neck:
            x = layer(x, condition)

        decoder_conv_index = 0
        for i, layer in enumerate(self.decoder):
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                embedded_contour = self.decoder_zero_convs_in[decoder_conv_index](contour)

                if x.shape[2:] != embedded_contour.shape[2:]:
                    embedded_contour = F.interpolate(embedded_contour, size=x.shape[2:], mode='bilinear', align_corners=False)

                x += embedded_contour

                x += skip_connections.pop()
                
            x = layer(x, condition)
            
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x = self.decoder_zero_convs_out[decoder_conv_index](x)
                decoder_conv_index += 1

        x = self.output_res_block(torch.cat([x, residue], dim=1))
        x = self.output_block(x)
        return x

    
    def forward_with_contour_concat(self, x, condition=None):
        skip_connections = []
        residue = x
        
        _, contour = self.contour_extractor(condition)
        
        encoder_conv_index = 0
        for i, layer in enumerate(self.encoder):
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                embedded_contour = self.encoder_zero_convs_in[encoder_conv_index](contour)

                if x.shape[2:] != embedded_contour.shape[2:]:
                    embedded_contour = F.interpolate(embedded_contour, size=x.shape[2:], mode='bilinear', align_corners=False)

                x += embedded_contour

            x = layer(x, condition)
            
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x = self.encoder_zero_convs_out[encoder_conv_index](x)
                skip_connections.append(x)
                encoder_conv_index += 1

        for layer in self.bottle_neck:
            x = layer(x, condition)

        decoder_conv_index = 0
        for i, layer in enumerate(self.decoder):
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                embedded_contour = self.decoder_zero_convs_in[decoder_conv_index](contour)

                if x.shape[2:] != embedded_contour.shape[2:]:
                    embedded_contour = F.interpolate(embedded_contour, size=x.shape[2:], mode='bilinear', align_corners=False)

                x += embedded_contour

                x = torch.cat((x, skip_connections.pop()), dim=1)
            
            x = layer(x, condition)
            
            if not isinstance(layer[0], (DownSample, UpSample, DownSampleDepth)):
                x = self.decoder_zero_convs_out[decoder_conv_index](x)
                decoder_conv_index += 1

        x = self.output_res_block(torch.cat([x, residue], dim=1))
        x = self.output_block(x)
        return x
