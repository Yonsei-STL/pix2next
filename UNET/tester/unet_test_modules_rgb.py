import os
import sys
import yaml
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
import torchvision.utils as vutils
from datetime import datetime
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

torch.autograd.set_detect_anomaly(True)

from common.data_utils import *

# 이미지 생성 함수
def generate_image(base_path, model, device, image_index, rgb_img, nir_img):
    model.eval()
    rgb_img = rgb_img.to(device)
    nir_img = nir_img.to(device)

    with torch.no_grad():
        generated_nir = model(rgb_img)

    reverse_transform_tensor = Compose([Normalize(mean=(-1, -1, -1), std=(2, 2, 2))])
    sample_image = reverse_transform_tensor(generated_nir)
    original_rgb = reverse_transform_tensor(rgb_img)
    original_nir = reverse_transform_tensor(nir_img)

    # Concatenate generated images and original images for comparison
    # comparison_images = torch.cat((original_rgb, original_nir, sample_image), dim=0)

    vutils.save_image(original_rgb, f"{base_path}/image_results/{image_index+1}_real_A.png", nrow=1)
    vutils.save_image(original_nir, f"{base_path}/image_results/{image_index+1}_real_B.png", nrow=1)
    vutils.save_image(sample_image, f"{base_path}/image_results/{image_index+1}_fake_B.png", nrow=1)
    #vutils.save_image(comparison_images, f"{base_path}/image_results/image_{image_index+1}.png", nrow=1)

def test(config, G, test_loader, device):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "test_" + current_time
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_path + "/image_results", exist_ok=True)

    with open(base_path + '/test_config.yaml', 'w') as f:
        f.write(f"# {count_parameters(G)}\n")
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

    print('base path:', base_path)

    # 모델 평가 모드로 전환
    G.eval()

    for i, (rgb, nir) in enumerate(tqdm(test_loader, desc="Testing", unit="batch")):
        generate_image(base_path, G, device, i, rgb, nir)
