import os
import cv2
import numpy as np
import tensorflow as tf
from skimage import img_as_ubyte
import imageio
import time
import re

# Load the new autoencoder model (assumed to upscale by 4x)
autoencoder_quad = tf.keras.models.load_model('autoencoder_x4.h5')

# Function definitions
def checkSize(size):
    return size % 16 == 0  # For 4x upscaling, the size should be divisible by 16

def paddingImage(img, target_height, target_width):
    height, width, channels = img.shape
    padded_img = np.zeros((target_height, target_width, channels), dtype=img.dtype)
    padded_img[:height, :width, :channels] = img
    return padded_img

def processImage(imagelow):
    height, width, channels = imagelow.shape
    target_height = ((height // 16) + (height % 16 > 0)) * 16
    target_width = ((width // 16) + (width % 16 > 0)) * 16

    if height != target_height or width != target_width:
        imagelow = paddingImage(imagelow, target_height, target_width)

    image = imagelow.astype(np.float32) / 255               
    sr_image = np.clip(autoencoder_quad.predict(np.array([image])), 0.0, 1.0)[0]
    sr_image = sr_image[:height*4, :width*4]  # Cropping for 4x upscaled image
    return sr_image

# Directories to process
subdirs = ['B100', 'Set5', 'Set14', 'Urban100']

# Process images in each subdirectory of the Benchmark folder
for subdir in subdirs:
    path = os.path.join('..', 'Benchmark', subdir, 'x4')  # Adjusted for 4x upscaled images
    if not os.path.exists(path):
        print(f"Directory {path} not found. Skipping.")
        continue

    print(f"Processing {subdir}...")

    for filename in os.listdir(path):
        if filename.endswith('LR.png'):
            filepath = os.path.join(path, filename)
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            sr_image = processImage(image)
            elapsed_time = time.time() - start_time

            sr_filename = filename.replace('LR.png', 'SR_x4.png')  # Adjusted filename for 4x upscaled image
            sr_filepath = os.path.join(path, sr_filename)
            imageio.imwrite(sr_filepath, img_as_ubyte(sr_image))

            print(f"Processed {filename} in {elapsed_time:.2f} seconds.")

    print(f"Completed processing for {subdir}.")

print("Super-resolution processing for all directories complete.")
