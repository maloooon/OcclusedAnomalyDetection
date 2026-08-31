import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import re

def overlay_raspberries(folder_path, output_path=None):
    """
    Overlay all raspberry images from a folder, preserving their original positions.
    Black background becomes transparent, and grade numbers are drawn over each raspberry.
    This is a helper function to check whether the extracted grades from the GT are correctly matched.
    
    Args:
        folder_path: Path to folder containing .png files
        output_path: Where to save the combined image (None = show only)
    
    Returns:
        PIL Image object
    """
    # Get all .png files and sort them
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    
    if not image_files:
        print(f"No .png files found in {folder_path}")
        return None
    
    # Load first image to get dimensions
    first_img = Image.open(os.path.join(folder_path, image_files[0]))
    width, height = first_img.size
    
    # Create blank canvas with transparent background
    canvas = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    # Process each image
    for img_file in image_files:
        # Extract grade from filename
        match = re.search(r'grade(\d+)', img_file)
        grade = match.group(1) if match else '?'
        
        # Load image
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')
        
        # Convert to RGBA
        img_rgba = img.copy()
        img_rgba = img_rgba.convert('RGBA')
        
        # Make black pixels transparent
        img_array = np.array(img_rgba)
        # Create mask where all RGB channels are close to black (threshold for safety)
        black_mask = (img_array[:, :, 0] < 10) & \
                     (img_array[:, :, 1] < 10) & \
                     (img_array[:, :, 2] < 10)
        img_array[black_mask, 3] = 0  # Set alpha to 0 for black pixels
        
        # Convert back to PIL
        img_transparent = Image.fromarray(img_array)
        
        # Overlay onto canvas
        canvas = Image.alpha_composite(canvas, img_transparent)
    
    # Convert to RGB for drawing (easier to work with)
    canvas_rgb = Image.new('RGB', (width, height), color='black')
    canvas_rgb.paste(canvas, mask=canvas.split()[3])  # Use alpha channel as mask
    
    # Draw grade numbers
    draw = ImageDraw.Draw(canvas_rgb)
    
    for img_file in image_files:

        # Skip image that has 'overlayed' string in its name
        if 'overlayed' in img_file:
            continue

        # Extract grade
        match = re.search(r'grade(\d+)', img_file)
        grade = match.group(1) if match else '?'
        
        # Load image to find raspberry position
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # Find non-black pixels (the raspberry)
        non_black = (img_array[:, :, 0] > 10) | \
                    (img_array[:, :, 1] > 10) | \
                    (img_array[:, :, 2] > 10)
        
        if not non_black.any():
            continue
        
        # Find centroid of raspberry
        coords = np.argwhere(non_black)
        center_y, center_x = coords.mean(axis=0).astype(int)
        
        # Get text size
        bbox = draw.textbbox((0, 0), str(grade), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Position text at centroid
        text_x = center_x - text_width // 2
        text_y = center_y - text_height // 2
        
        # Draw text with outline
        for offset_x in [-2, -1, 0, 1, 2]:
            for offset_y in [-2, -1, 0, 1, 2]:
                draw.text((text_x + offset_x, text_y + offset_y), 
                         str(grade), font=font, fill='black')
        draw.text((text_x, text_y), str(grade), font=font, fill='yellow')
    
    # Save if output path provided
    if output_path:
        canvas_rgb.save(output_path)
       # print(f"Saved combined image to {output_path}")
    
    return canvas_rgb

def apply_random_transform(img, mask, depth, mask_unfiltered=None,
                           scale_range=(0.5, 0.8),
                           rotation_range=(-180, 180)):
    """
    Apply random rotation + scale to img, mask, depth, and optionally
    mask_unfiltered. A single affine matrix is computed and reused for all arrays so
    they stay perfectly aligned.

    Args:
        img: HxWx3 uint8
        mask: HxW bool
        depth: HxW float32
        mask_unfiltered: HxW bool (optional)
        scale_range: (min, max) uniform scale factor
        rotation_range: (min_deg, max_deg) uniform rotation in degrees

    Returns:
        (img_out, mask_out, depth_out) when mask_unfiltered is None, else
        (img_out, mask_out, depth_out, mask_unfiltered_out)
    """
    angle = random.uniform(*rotation_range)
    scale = random.uniform(*scale_range)

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    img_out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask_out = cv2.warpAffine(
        mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    ).astype(bool)
    depth_out = cv2.warpAffine(
        depth.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    if mask_unfiltered is not None:
        mask_unfiltered_out = cv2.warpAffine(
            mask_unfiltered.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
        ).astype(bool)
        return img_out, mask_out, depth_out, mask_unfiltered_out

    return img_out, mask_out, depth_out


def main():
    folder = "dataset_single_objects/GT/img001"
    output = "overlayed_raspberries_img001_GT.png"
    
    img = overlay_raspberries(folder, output)


#if __name__ == "__main__":
#    main()