import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import median_filter
from torchvision import transforms


class CoherentNoiseFilter:
    def __init__(self, filter_size=3, tolerance=30):
        self.filter_size = filter_size
        self.tolerance = tolerance

    def __call__(self, img):
        img_np = np.array(img)

        if img_np.ndim == 2:
            mask = (img_np <= self.tolerance) | (img_np >= 255 - self.tolerance)
            filtered = median_filter(img_np, size=self.filter_size)
            final_img = np.where(mask, filtered, img_np)
            mode = 'L'
        else:
            mask_r = (img_np[:, :, 0] <= self.tolerance) | (img_np[:, :, 0] >= 255 - self.tolerance)
            mask_g = (img_np[:, :, 1] <= self.tolerance) | (img_np[:, :, 1] >= 255 - self.tolerance)
            mask_b = (img_np[:, :, 2] <= self.tolerance) | (img_np[:, :, 2] >= 255 - self.tolerance)

            global_mask = mask_r | mask_g | mask_b
            global_mask_3d = np.repeat(global_mask[:, :, np.newaxis], 3, axis=2)

            filtered_full = np.zeros_like(img_np)
            for c in range(3):
                filtered_full[:, :, c] = median_filter(img_np[:, :, c], size=self.filter_size)

            final_img = np.where(global_mask_3d, filtered_full, img_np)
            mode = 'RGB'

        restored_pil = Image.fromarray(final_img.astype(np.uint8), mode=mode)
        return restored_pil


class MedianBlur:
    def __init__(self, size=3):
        self.size = size
    
    def __call__(self, img):
        return img.filter(ImageFilter.MedianFilter(size=self.size))


def get_train_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CoherentNoiseFilter(filter_size=3, tolerance=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])


def get_val_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        CoherentNoiseFilter(filter_size=3, tolerance=30),
        transforms.ToTensor(),
    ])
