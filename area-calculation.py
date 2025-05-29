import numpy as np
import pandas as pd
# import torch
from bioio import BioImage
import cv2
from tifffile import imwrite

input_path = "/allen/aics/assay-dev/users/Sandi/emt-tracking/emt-data-013025/raw/3500006851/B3_P0/movie/last-200.tif"

# read in image 
img = BioImage(input_path) 
print("full image shape:", img.shape)
IMG = img.data

# print("img type:", type(img))
# print("IMG type:", type(IMG))

# variable that has shape of full movie
img_dimensions = IMG.shape # (T,C,Z,Y,X)

# list of images to calculate area of individual objects
images = [] # (C,Y,X) for T*Z

# for loop that separates movie into images in every timepoint X slice
for timepoint in range(20,31):
    for zslice in range(0,10):
        tps = IMG[timepoint,:,zslice,:,:]
        images.append(tps)

test_img = images[0]
print("test_img shape:", test_img.shape)
print("unique values in test_img:", np.unique(test_img))

# save the file
imwrite("test_img.tiff", test_img)

# # Threshold the image (if needed) to create a binary mask
# _, binary_mask = cv2.threshold(test_img, 127, 255, cv2.THRESH_BINARY)

# print(np.unique(binary_mask))

# # Find contours of individual objects
# contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Compute areas of individual objects
# object_areas = [cv2.contourArea(cnt) for cnt in contours]