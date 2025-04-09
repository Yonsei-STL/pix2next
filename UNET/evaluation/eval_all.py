import os
import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from pytorch_fid import fid_score
import tempfile
import lpips
from DISTS_pytorch import DISTS

# CUDA 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LPIPS와 DISTS 모델을 전역 변수로 정의
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
dists_model = DISTS().to(device)

def calculate_psnr_ssim(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))

    if image1 is None or image2 is None:
        raise FileNotFoundError(f"Could not open {image1_path} or {image2_path}")

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Convert NIR image to RGB by replicating the single channel
    if len(image2.shape) == 2 or image2.shape[2] == 1:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    psnr_value = psnr(image1, image2)
    ssim_value = ssim(image1, image2, channel_axis=2)  # Use 'channel_axis' instead of 'multichannel'

    return psnr_value, ssim_value

def calculate_rmse(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    rmse_per_channel = np.sqrt(np.mean((image1 - image2) ** 2, axis=(0, 1)))
    return np.mean(rmse_per_channel)

def calculate_sam(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    sam_value = np.mean(np.arccos(np.clip(np.sum(image1 * image2, axis=-1) / 
                            (np.linalg.norm(image1, axis=-1) * np.linalg.norm(image2, axis=-1) + 1e-10), -1, 1)))
    return sam_value

def calculate_fid(real_images_folder, fake_images_folder):
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, fake_images_folder], batch_size=50, device='cuda:0', dims=2048)
    return fid_value

def calculate_lpips(image1, image2):
    image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        return loss_fn_lpips(image1, image2).item()

def calculate_dists(image1, image2):
    image1 = torch.from_numpy(image1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    image2 = torch.from_numpy(image2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        return dists_model(image1, image2).item()

def main():
    images_folder = './images'
    test_B_folder = './images'
    result_file = './result.txt'
    html_file = './index.html'

    psnr_scores = []
    ssim_scores = []
    rmse_scores = []
    sam_scores = []
    lpips_scores = []
    dists_scores = []

    # Create temporary directories to store images for FID calculation
    with tempfile.TemporaryDirectory() as real_images_folder, tempfile.TemporaryDirectory() as fake_images_folder:

        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Comparison</title>
            <style>
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid black; padding: 10px; text-align: center; }
                img { width: 100%; max-width: 300px; }
            </style>
        </head>
        <body>
            <h1>Image Comparison</h1>
            <table>
                <tr>
                    <th>Input Label</th>
                    <th>Synthesized Image</th>
                    <th>NIR Image</th>
                    <th>PSNR</th>
                    <th>SSIM</th>
                    <th>RMSE</th>
                    <th>SAM</th>
                    <th>LPIPS</th>
                    <th>DISTS</th>
                </tr>
        """

        with open(result_file, 'w') as f:
            f.write("File, PSNR, SSIM, RMSE, SAM, LPIPS, DISTS\n")
            for image_name in os.listdir(images_folder):
                if image_name.endswith('_fake_B.png'):
                    base_name = image_name.replace('_fake_B.png', '')
                    print(base_name)
                    input_label_name = base_name + '_real_A.png'
                    synthesized_image_name = base_name + '_fake_B.png'
                    
                    nir_image_name = base_name + '_real_B.png'
                    input_label_path = os.path.join(images_folder, input_label_name)
                    synthesized_image_path = os.path.join(images_folder, synthesized_image_name)
                    nir_image_path = os.path.join(test_B_folder, nir_image_name)
                # if image_name.endswith('_synthesized_image.jpg'):
                #     base_name = image_name.replace('_synthesized_image.jpg', '')
                #     print(base_name)
                #     input_label_name = base_name + '_real_A.png'
                #     #synthesized_image_name = base_name + '_fake_B.png'
                #     synthesized_image_name = base_name + '_synthesized_image.jpg'
                #     nir_image_name = base_name + '_real_B.png'
                #     input_label_path = os.path.join(images_folder, input_label_name)
                #     synthesized_image_path = os.path.join(images_folder, synthesized_image_name)
                #     nir_image_path = os.path.join(test_B_folder, nir_image_name)

                    if os.path.exists(input_label_path) and os.path.exists(nir_image_path):
                        image1 = cv2.imread(synthesized_image_path)
                        image2 = cv2.imread(nir_image_path)

                        # Resize images to 256x256
                        image1 = cv2.resize(image1, (256, 256))
                        image2 = cv2.resize(image2, (256, 256))

                        # Save images to temporary directories for FID calculation
                        fake_image_temp_path = os.path.join(fake_images_folder, synthesized_image_name)
                        real_image_temp_path = os.path.join(real_images_folder, nir_image_name)
                        cv2.imwrite(fake_image_temp_path, image1)
                        cv2.imwrite(real_image_temp_path, image2)

                        psnr_value, ssim_value = calculate_psnr_ssim(synthesized_image_path, nir_image_path)
                        rmse_value = calculate_rmse(image1, image2)
                        sam_value = calculate_sam(image1, image2)
                        lpips_value = calculate_lpips(image1, image2)
                        dists_value = calculate_dists(image1, image2)

                        psnr_scores.append(psnr_value)
                        ssim_scores.append(ssim_value)
                        rmse_scores.append(rmse_value)
                        sam_scores.append(sam_value)
                        lpips_scores.append(lpips_value)
                        dists_scores.append(dists_value)

                        f.write(f"{base_name}, {psnr_value}, {ssim_value}, {rmse_value}, {sam_value}, {lpips_value}, {dists_value}\n")
                        
                        html_content += f"""
                        <tr>
                            <td>
                                <p>{input_label_name}</p>
                                <img src="{input_label_path}" alt="{input_label_name}">
                            </td>
                            <td>
                                <p>{synthesized_image_name}</p>
                                <img src="{synthesized_image_path}" alt="{synthesized_image_name}">
                            </td>
                            <td>
                                <p>{nir_image_name}</p>
                                <img src="{nir_image_path}" alt="{nir_image_name}">
                            </td>
                            <td>{psnr_value:.2f}</td>
                            <td>{ssim_value:.4f}</td>
                            <td>{rmse_value:.2f}</td>
                            <td>{sam_value:.4f}</td>
                            <td>{lpips_value:.4f}</td>
                            <td>{dists_value:.4f}</td>
                        </tr>
                        """
                    else:
                        print(f"File {input_label_path} or {nir_image_path} does not exist.")

            fid_value = calculate_fid(real_images_folder, fake_images_folder)
            mean_psnr = np.mean(psnr_scores)
            mean_ssim = np.mean(ssim_scores)
            mean_rmse = np.mean(rmse_scores)
            mean_sam = np.mean(sam_scores)
            mean_lpips = np.mean(lpips_scores)
            mean_dists = np.mean(dists_scores)

            f.write(f"\nMean PSNR: {mean_psnr}\n")
            f.write(f"Mean SSIM: {mean_ssim}\n")
            f.write(f"Mean RMSE: {mean_rmse}\n")
            f.write(f"Mean SAM: {mean_sam}\n")
            f.write(f"Mean LPIPS: {mean_lpips}\n")
            f.write(f"Mean DISTS: {mean_dists}\n")
            f.write(f"FID: {fid_value}\n")

            html_content += f"""
            </table>
            <h2>Mean PSNR: {mean_psnr:.2f}</h2>
            <h2>Mean SSIM: {mean_ssim:.4f}</h2>
            <h2>Mean RMSE: {mean_rmse:.2f}</h2>
            <h2>Mean SAM: {mean_sam:.4f}</h2>
            <h2>Mean LPIPS: {mean_lpips:.4f}</h2>
            <h2>Mean DISTS: {mean_dists:.4f}</h2>
            <h2>FID: {fid_value:.2f}</h2>
        </body>
        </html>
        """

        with open(html_file, 'w') as f:
            f.write(html_content)

if __name__ == '__main__':
    main()