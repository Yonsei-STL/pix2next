import sys
import os
import numpy as np

import torch.nn as nn
from safetensors.torch import load_file

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import *
from UNET.models.UNET import *
from UNET.models.discriminator import *
from UNET.trainer.train_modules_rgb import *



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

config_path = '../../config_gan_base_internimage_pyramid.yaml'
config = load_config(config_path)
epochs = config['loader']['train']['epoch']
loaders = build_loader(config)

print('train_loader len:', len(loaders['train']))
print('test_loader len:', len(loaders['test']))

# Acclereator 설정

#device = accelerator.device
device = 'cuda:0'
torch.cuda.set_device(device) # change allocation of current GPU

G = UNET(config).to(device)
D = Discriminator().to(device)
print(G)

# 가중치 초기화 적용
G.apply(weights_init)
# D.apply(weights_init)

# safetensor_path = '/home/SHB/Mul-Spec/pix2pix_swin/UNET/trainer/train_20240702_173500/checkpoints/checkpoint_epoch_660.pt'
# state_dict = torch.load(safetensor_path)
# G.load_state_dict(state_dict)

optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-04, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-04, betas=(0.5, 0.999))

from diffusers.optimization import get_cosine_schedule_with_warmup

lr_scheduler_G = get_cosine_schedule_with_warmup(
    optimizer=optimizer_G,
    num_warmup_steps=1000,
    num_training_steps=(len(loaders['train']) * epochs),
)

lr_scheduler_D = get_cosine_schedule_with_warmup(
    optimizer=optimizer_D,
    num_warmup_steps=1000,
    num_training_steps=(len(loaders['train']) * epochs),
)


train(config, G, D, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, loaders['train'], loaders['test'], device)
