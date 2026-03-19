import numpy as np
import torch
from PIL import ImageEnhance
import PIL
import PIL.Image as Image


def min_max_scale_image(mask: Image, output_dtype: type = np.float32) -> Image:
    image_array = np.array(mask).astype(output_dtype) / 255.0

    min_val = np.min(image_array)
    max_val = np.max(image_array)
    scaled_array = (image_array - min_val) / (max_val - min_val)
    # Ensure the scaled array is only 0s and 1s
    binary_array = torch.round(torch.tensor(scaled_array)).numpy()

    scaled_image = Image.fromarray((binary_array*255).astype(output_dtype))
    return scaled_image

class IncreaseContrast:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.factor)