import os
import sys
import yaml
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.utils as vutils
from datetime import datetime
from torchvision.transforms import Compose, Normalize
from PIL import Image
import kornia

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

torch.autograd.set_detect_anomaly(True)

from common.data_utils import *

def denormalize(tensor):
    return tensor * 0.5 + 0.5

# 이미지 생성 함수
def generate_images(base_path, model, device, epoch, num_images_per_class, rgb_nir_pairs):
    model.eval()
    latents = []
    original_rgbs = []
    original_nirs = []

    for rgb_img, nir_img in rgb_nir_pairs:
        rgb_img = rgb_img.unsqueeze(0).to(device)
        nir_img = nir_img.unsqueeze(0).to(device)
    
        with torch.no_grad():
            generated_nir = model(rgb_img)

        latents.append(generated_nir)
        original_rgbs.append(rgb_img)
        original_nirs.append(nir_img)

    # Concatenate latents and original NIRs from single GPU
    gathered_latents = [latent.to("cpu") for latent in latents]
    gathered_original_rgbs = [rgb.to("cpu") for rgb in original_rgbs]
    gathered_original_nirs = [nir.to("cpu") for nir in original_nirs]
    concatenated_latents = torch.cat(gathered_latents, dim=0)
    concatenated_original_rgbs = torch.cat(gathered_original_rgbs, dim=0)
    concatenated_original_nirs = torch.cat(gathered_original_nirs, dim=0)

    reverse_transform_tensor = Compose([Normalize(mean=(-1, -1, -1), std=(2, 2, 2))])
    sample_images = reverse_transform_tensor(concatenated_latents)
    original_images = reverse_transform_tensor(concatenated_original_rgbs)
    original_images2 = reverse_transform_tensor(concatenated_original_nirs)

    # Concatenate generated images and original images for comparison
    comparison_images = torch.cat((original_images, original_images2, sample_images), dim=0)
    vutils.save_image(comparison_images, f"{base_path}/image_results/epoch_{epoch+1}.png", nrow=num_images_per_class)

def train(config, G, D, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D, train_loader, test_loader, device):
    print(count_parameters(G))

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "train_" + current_time
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/image_results", exist_ok=True)
    # os.makedirs(base_path + "/weights", exist.ok=True)

    with open(base_path + '/train_config.yaml', 'w') as f:
        f.write(f"# {count_parameters(G)}\n")
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print('base path:', base_path)

    scaler = GradScaler()

    train_epochs = config['loader']['train']['epoch']
    num_images_per_class = 5
    img_size = config['data']['resize'][0]  # 이미지 크기 설정

    # Keeping a record of the losses for later viewing
    with torch.autograd.set_detect_anomaly(False):
        with open(f"{base_path}/epoch_losses.txt", "w") as file:
            for epoch in range(train_epochs):
                G.train()
                D.train()

                progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{train_epochs}]", unit="batch")
                epoch_loss = 0
                epoch_ssim = 0

                for rgb, nir in progress_bar:
                    rgb = rgb.to(device)
                    nir = nir.to(device)

                    #####################################
                    ### forward pass ###
                    generated_nir = G(rgb)
                    losses = D(rgb, nir, generated_nir)

                    # calculate final loss scalar
                    loss_D_fake = losses['loss_D_fake']
                    loss_D_real = losses['loss_D_real']
                    loss_G_GAN = losses['loss_G_GAN']
                    loss_G_GAN_Feat = losses.get('loss_G_GAN_Feat', 0)
                    loss_G_VGG = losses.get('loss_G_VGG', 0)
                    loss_G_SSIM = losses.get('loss_G_SSIM', 0)
                    loss_transform = losses.get('transform_loss', 0)

                    loss_D = (loss_D_fake + loss_D_real) * 0.5
                    loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG + loss_transform #+ loss_G_SSIM

                    ### backward pass ###
                    # update generator weights
                    optimizer_G.zero_grad()
                    loss_G.backward()
                    optimizer_G.step()

                    # update discriminator weights
                    optimizer_D.zero_grad()
                    loss_D.backward()
                    optimizer_D.step()
                    #####################################

                    progress_bar.set_postfix({
                        "Loss D Fake": loss_D_fake.item(),
                        "Loss D Real": loss_D_real.item(),
                        "Loss G GAN": loss_G_GAN.item(),
                        "Loss G GAN Feat": loss_G_GAN_Feat.item() if isinstance(loss_G_GAN_Feat, torch.Tensor) else 0,
                        "Loss G VGG": loss_G_VGG.item() if isinstance(loss_G_VGG, torch.Tensor) else 0,
                        "Loss G SSIM": loss_G_SSIM.item() if isinstance(loss_G_SSIM, torch.Tensor) else 0,
                        "Loss transform": loss_transform.item() if isinstance(loss_transform, torch.Tensor) else 0,
                    })

                    epoch_loss += loss_G.item()
                    epoch_ssim += loss_G_SSIM.item()

                # Write epoch loss to file
                avg_loss = epoch_loss / len(train_loader)
                avg_ssim = epoch_ssim / len(train_loader)
                file.write(f"Epoch {epoch+1}/{train_epochs}, Loss: {avg_loss}, SSIM_Loss: {avg_ssim}\n")
                file.flush()

                if (epoch + 1) % 10 == 0:
                    sample_indices = random.sample(range(len(test_loader.dataset)), num_images_per_class)
                    sample_rgb_nir_pairs = [test_loader.dataset[i] for i in sample_indices]
                    generate_images(base_path, G, device, epoch, num_images_per_class, sample_rgb_nir_pairs)

                if (epoch + 1) % 100 == 0:
                    checkpoint_dir = os.path.dirname(f"{base_path}/checkpoints/")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    checkpoint_path = f"{base_path}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
                    torch.save(G.state_dict(), checkpoint_path)

                # Step the learning rate schedulers
                lr_scheduler_G.step()
                lr_scheduler_D.step()
