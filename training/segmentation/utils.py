import os
from glob import glob

def get_pre_disaster_image_mask_pairs(path):
    satellite_images = sorted(glob(os.path.join(path, 'images', '*_pre_disaster.png')))
    mask_images = sorted(glob(os.path.join(path, 'masks', '*_pre_disaster.png')))
    return list(zip(satellite_images, mask_images))