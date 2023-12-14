import os
import cv2
import matplotlib.pyplot as plt
import random

def show_random_image_pair(image_folder):
    lr_images = [f for f in os.listdir(image_folder) if f.endswith('LR.png')]  # Modify if the naming convention is different
    if not lr_images:
        print("No LR images found.")
        return

    # Select a random LR image
    random_lr_image = random.choice(lr_images)
    lr_path = os.path.join(image_folder, random_lr_image)

    # Corresponding SR image path
    sr_filename = random_lr_image.replace('LR', 'SR')  # Modify if the naming convention is different
    sr_path = os.path.join(image_folder, sr_filename)

    lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    sr_img = cv2.imread(sr_path, cv2.IMREAD_COLOR)

    if lr_img is None or sr_img is None:
        print(f"Images not found: {lr_path} or {sr_path}")
        return

    # Convert from BGR to RGB
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)

    # Display images side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title('Low Resolution')
    axs[0].axis('off')

    axs[1].imshow(sr_img)
    axs[1].set_title('Super Resolved')
    axs[1].axis('off')

    plt.show(block=True)

# Example usage
image_folder = "../Benchmark/Urban100/x2/"  # Replace with the path to your images

show_random_image_pair(image_folder)
