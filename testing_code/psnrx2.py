import os
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def calculate_psnr(original, contrast):
    return compare_psnr(original, contrast, data_range=contrast.max() - contrast.min())

def load_images_from_folder(folder, suffix):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(suffix):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images[filename] = img
    return images

# Directories
datasets = ['B100', 'Set5', 'Set14', 'Urban100']
base_path = '..'  # Adjust as needed

# Iterate through each dataset
for subdir in datasets:
    dataset_path = os.path.join(base_path, 'Benchmark', subdir, 'x4')
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} not found. Skipping.")
        continue

    print(f"Processing {subdir}...")

    # Load HR and SR images
    hr_suffix = "HR.png"  # Adjust suffix if needed
    sr_suffix = "SR_x4.png"  # Adjust suffix if needed
    hr_images = load_images_from_folder(dataset_path, hr_suffix)
    sr_images = load_images_from_folder(dataset_path, sr_suffix)

    # Calculate and print PSNR for each pair
    psnr_values = []
    for filename, sr_image in sr_images.items():
        hr_filename = filename.replace(sr_suffix, hr_suffix)
        if hr_filename in hr_images:
            hr_image = hr_images[hr_filename]
            psnr = calculate_psnr(hr_image, sr_image)
            psnr_values.append(psnr)
            #print(f"PSNR for {filename}: {psnr:.2f} dB")

    # Calculate average PSNR
    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR for {subdir}: {average_psnr:.2f} dB")
    else:
        print(f"No SR and HR image pairs found in {subdir} for PSNR calculation.")

print("PSNR calculation for all datasets complete.")
