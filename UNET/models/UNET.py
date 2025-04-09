import sys
import os
import kornia

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UNET.models.unet_modules import *
from UNET.models.unet_core import *

class UNET(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store the config for later use
        self.config = config  

        # Adjust decoder block arguments to account for skip connections
        self.skip_connection_type = config['model']['unet']['skip_connection_type']
        if self.skip_connection_type == 'concat':
            print('adjusting decoder block args')
            self._adjust_decoder_block_args()

        # Initialize components based on (potentially) modified config
        self.init_conv_args = config['model']['unet']['init_conv']
        self.init_conv = nn.Conv2d(*self.init_conv_args)

        self.unet = UnetCore(self.config)
        
        ## image condition
        self.image_cond = config['model']['unet']['condition']['image_condition']

        if self.image_cond:
            self.cond_image_input_channel = config['model']['unet']['condition']['d_condition']
            self.cond_image_target_channel = config['model']['unet']['encoder_block_args'][0][0][0]  # first input channel for unet
            
            self.cond_image_input_conv = nn.Conv2d(self.cond_image_input_channel, self.cond_image_target_channel, kernel_size=1, stride=1, padding=0)
            self.cond_image_concat_conv = nn.Conv2d(self.cond_image_target_channel*2, self.cond_image_target_channel, kernel_size=1, stride=1, padding=0)

    def _get_output_channels(self, args):
        if isinstance(args[-1], list):
            if len(args[-1]) == 1: # if down or up sample
                return args[-1][-1] 
            else:
                return args[-1][1]
        else:
            return args[-1]

    def _adjust_decoder_block_args(self):
        encoder_output_channels = [self._get_output_channels(args) for args in self.config['model']['unet']['encoder_block_args']]
        
        decoder_block_types = self.config['model']['unet']['decoder_block_types']
        decoder_block_args = self.config['model']['unet']['decoder_block_args']

        for i in range(len(decoder_block_args)):
            if (decoder_block_types[i] != 'upsample'):
                block_args = decoder_block_args[i] # 0, 1, 2, ...
                skip_channels = encoder_output_channels[-i-1] # -1, -2, -3, ...

                if isinstance(block_args[0], list):
                    block_args[0][0] += skip_channels # Add encoder output channels to decoder input channels
                else:
                    # If the first argument is not a list, update the input channels directly
                    block_args[0] += skip_channels
        
        # Modify config
        self.config['model']['unet']['decoder_block_args'] = decoder_block_args
    
    def forward(self, x):
        x = self.init_conv(x)
        return self.unet(x)
