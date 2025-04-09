import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torchvision.utils as vutils
from datetime import datetime
from torchvision.transforms import Compose, Normalize
from PIL import Image

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils import *

def denormalize(tensor):
    return tensor * 0.5 + 0.5

def generate_class_images(model, scheduler, device, target_ch, class_labels, num_images_per_class, img_size):
    model.eval()
    class_images = []
    
    for label in class_labels:
        label_tensor = torch.tensor([label] * num_images_per_class, device=device)
        latent = torch.randn((num_images_per_class, target_ch, img_size, img_size), device=device)
        
        for _, t in tqdm(enumerate(scheduler.timesteps)):
            with torch.no_grad():
                timestep = torch.tensor([t], device=device).expand(latent.shape[0])
                pred_noise = model(latent, timestep, label_tensor)
                
                latent = scheduler.step(pred_noise, t, latent).prev_sample
        
        class_images.append(latent)
    
    class_images = torch.cat(class_images, dim=0)
    return class_images

def generate_images(base_path, model, scheduler, device, target_ch, epoch, num_images_per_class, img_size, accelerator):
    model.eval()
    latents = []

    latent = torch.randn((num_images_per_class, target_ch, img_size, img_size), device=device)
    
    for _, t in tqdm(enumerate(scheduler.timesteps)):
        with torch.no_grad():
            timestep = torch.tensor([t], device=device).expand(latent.shape[0])
            pred_noise = model(latent, timestep)

        latent = scheduler.step(pred_noise, t, latent).prev_sample
            
    latents.append(latent)

    # Gather latents from all GPUs
    gathered_latents = accelerator.gather(latents)

    if accelerator.is_main_process:
        # Concatenate latents from all GPUs
        gathered_latents = [latent.to("cpu") for latent in gathered_latents]  # Move tensors to CPU
        concatenated_latents = torch.cat(gathered_latents, dim=0)

        reverse_transform_tensor = Compose([
            Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        ])

        sample_images = reverse_transform_tensor(concatenated_latents)
        vutils.save_image(sample_images, f"{base_path}/image_results/epoch_{epoch+1}.png", nrow=num_images_per_class)

def train_unet(config, accelerator, model, optimizer, scheduler, lr_scheduler, train_loader, test_loader, device, use_labels=True):
    print(count_parameters(model))
    scheduler.set_timesteps(1000)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "train_" + current_time
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/image_results", exist_ok=True)
    os.makedirs(base_path + "/weights", exist_ok=True)

    with open(base_path + '/train_config.yaml', 'w') as f:
        f.write(f"# {count_parameters(model)}\n")
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print('base path:', base_path)

    scaler = GradScaler()
    loss_fn = nn.MSELoss()

    train_epochs = config['loader']['train']['epoch']
    best_loss = float('inf')
    
    class_labels = list(range(10))  # STL-10 데이터셋의 클래스 레이블 (0부터 9까지)
    num_images_per_class = 5
    img_size = config['data']['resize'][0]  # 이미지 크기 설정

    # Keeping a record of the losses for later viewing
    with open(f"{base_path}/epoch_losses.txt", "w") as file:
        for epoch in range(train_epochs):
            losses = []

            model.train()
            epoch_noise_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{train_epochs}]", unit="batch")
            for batch in progress_bar:
                if use_labels:
                    images, labels = batch
                    labels = labels.to(device)
                else:
                    images = batch

                images = images.to(device)
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, int(scheduler.config.num_train_timesteps), (images.shape[0],), device=device)
                noisy_x = scheduler.add_noise(images, noise, timesteps)

                with accelerator.accumulate(model):
                    if use_labels:
                        pred_noise = model(noisy_x, timesteps, labels)
                    else:
                        pred_noise = model(noisy_x, timesteps)

                    loss = loss_fn(pred_noise, noise)
                    losses.append(loss.item())

                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(optimizer)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    epoch_noise_loss += loss.item()
                    progress_bar.set_postfix({"Loss": loss.item()})

            avg_loss = sum(losses[-100:]) / 100 if len(losses) >= 100 else sum(losses) / len(losses)
            avg_epoch_loss = epoch_noise_loss / len(train_loader)
            file.write(f"Epoch {epoch+1}: Epoch Loss {avg_epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}\n")
            file.flush()

            if (epoch + 1) % 20 == 0:  # Generate and save images every 10 epochs
                target_ch = images.shape[1]
                generate_images(base_path, model, scheduler, device, target_ch, epoch, num_images_per_class, img_size, accelerator)

            # Save model weights every 100 epochs
            if (epoch + 1) % 100 == 0:
                #torch.save(model.state_dict(), f"{base_path}/weights/model_epoch_{epoch+1}.pth")
                checkpoint_path = f"{base_path}/checkpoints/checkpoint_epoch_{epoch+1}.pt"
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save_state(checkpoint_path, model=unwrapped_model, optimizer=optimizer, scheduler=lr_scheduler, epoch=epoch)