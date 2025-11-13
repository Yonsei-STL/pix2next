import sys
import os
import numpy as np

from safetensors.torch import load_file

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import *
from UNET.models.UNET import *
from UNET.tester.unet_test_modules_rgb import *
from accelerate import Accelerator, DistributedDataParallelKwargs

config_path = '../../config_gan_base_internimage.yaml'
config = load_config(config_path)
loaders = build_loader(config)
print('test_loader len:', len(loaders['test']))


# Acclereator 설정
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
#accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
#device = accelerator.device
device = 'cuda:0'
torch.cuda.set_device(device) # change allocation of current GPU

G = UNET(config).to(device)
print(G)

safetensor_path = '/home/SHB/Mul-Spec/pix2pix_swin/UNET/trainer/train_20241003_160551/checkpoints/checkpoint_epoch_600.pt'
state_dict = torch.load(safetensor_path,map_location='cuda:0')
G.load_state_dict(state_dict)

test_loader=loaders['test']

test(config, G, test_loader, device)
