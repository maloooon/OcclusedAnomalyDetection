import cv2 
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import random 

def edge_smoothing(image, mask, ksize=(5, 5), sigma=3, thickness=5):
    """
    Smooth out edges of segmented raspberries by applying gaussian filtering
    only on the contour area, then overlay back on the original image.
    """
    mask_uint8 = mask.astype(np.uint8)

    # Find contours and draw them as a thick band
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=thickness)

    # Blur the original image
    blurred = cv2.GaussianBlur(image, ksize, sigma)

    # Swap in blurred pixels only where the contour is
    result = image.copy()
    result[contour_mask > 0] = blurred[contour_mask > 0]
    # TODO : possibly also need to adjust mask ???

    cv2.imwrite("edge_smoothed_image.png", result)
    

    return result, mask


def apply_cutout(image, mask, n_holes=3, hole_size_range=(32, 64)):
    """
    Apply random rectangular cutouts ONLY within the raspberry mask region.
    Returns the masked images and a binary tensor indicating which pixels
    were cut out (for computing loss only on the masked region).
    
    Args:
        images: (B, 3, H, W) tensor
        mask: (B, 1, H, W) binary mask of raspberry region
        n_holes: number of cutout rectangles
        hole_size_range: (min_size, max_size) for each rectangle edge
    """
    H, W, C = image.shape
    mask = np.expand_dims(mask, axis=2)  # ensure (H, W, 1)
    cutout_mask = np.ones((H, W, 1))

    print(H)
    print(W)


    for _ in range(n_holes):
        h = random.randint(hole_size_range[0], hole_size_range[1])
        w = random.randint(hole_size_range[0], hole_size_range[1])
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        cutout_mask[y:y+h, x:x+w] = 0.0

    

    # Only cut out within the raspberry region
    cutout_mask = cutout_mask + (1.0 - mask)  # preserve background (i.e. make sure we only cut out where we have the raspberry mask)
    cutout_mask = cutout_mask.clip(0, 1)



    masked_image = image * cutout_mask
    # The "holes" are where cutout_mask is 0 AND mask is 1
    holes = (1.0 - cutout_mask) * mask

    cv2.imwrite("masked_image.png", masked_image)
    cv2.imwrite("cutout_mask.png", cutout_mask.squeeze(2) * 255)

    return masked_image, holes

def find_contour(image,mask):

    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_mask, contours, -1, 255, thickness=8)

    cv2.imwrite("contour.png", contour_mask)

def apply_specular_suppression(image, mask, visualize=True, inpainting=True):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # NOTE:  might be RGB2HSV 
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    
    s_thresh = 100
    v_thresh = 220
    max_blob_area = 150

    raw_highlights = ((S < s_thresh) & (V > v_thresh) & (mask > 0)).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_highlights)
    filtered = np.zeros_like(raw_highlights)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= max_blob_area:
            filtered[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.dilate(filtered, kernel, iterations=1) & (mask > 0).astype(np.uint8) * 255

    if inpainting:
        result = cv2.inpaint(image, dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
       # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_title = "Inpainted"
    else:
        result = image.copy()
        result[filtered > 0] = 0
       # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_title = "Zeroed out"

    if visualize:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title in zip(axes,
            [cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             raw_highlights,
             filtered,
             result],
            ["Original", "Raw highlights", "Filtered highlights", result_title]):
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            ax.set_title(title); ax.axis('off')
        plt.tight_layout()
        plt.savefig("specular_suppression_visualization.png", dpi=100, bbox_inches='tight')

    return result, mask

def clean_protrusions(img, mask, depth, kernel_size=5):
    """Remove thin protrusions via morphological opening, then clean small fragments.
    
    Args:
        img: numpy array (H, W, 3) RGB image
        mask: numpy array (H, W) segmentation mask
        depth: numpy array (H, W) float32 depth map
        kernel_size: size of elliptical structuring element for opening
    
    Returns:
        clean_img: image with removed regions zeroed out
        clean_mask: cleaned binary mask
        clean_depth: depth with removed regions zeroed out
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    img = np.asarray(img, dtype=np.uint8)

    # drop any small fragments created by the opening
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        clean_depth = depth.copy()
        clean_depth[opened == 0] = 0
        return img, opened, clean_depth

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    keep = labels == largest

    clean_mask = keep.astype(np.uint8)
    clean_img = img.copy()
    clean_img[~keep] = 0
    clean_depth = depth.copy()
    clean_depth[~keep] = 0

    # save clean img
   # cv2.imwrite("cleaned_image.png", clean_img)

    return clean_img, clean_mask, clean_depth

def normalize_distribution(image,mask,target_mean = 128, target_std = 40):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    roi = v[mask > 0]
   # target_mean = target_mean.mean()
   # target_std = target_std.mean()
    v[mask > 0] = np.clip((roi - roi.mean()) / (roi.std() + 1e-6) * target_std + target_mean, 0, 255).astype(np.uint8)
    result = cv2.merge([h, s, v])
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    cv2.imwrite("normalized_image.png", result)

    return result, mask

def filter_holes_batched(score_maps, batch, mask, depth,
                 depth_threshold_percentile=20,
                 brightness_threshold_percentile=30, width = None, height = None):
    """
    Zero out anomaly scores in detected hole regions (deep AND dark pixels).
    For AD pipelines
    
    Args:
        score_maps: torch.Tensor (B, 1, H, W) anomaly score maps
        batch: torch.Tensor (B, 3, H, W) original images (normalized)
        mask: torch.Tensor or np.ndarray (B, H, W) or (B, 1, H, W) segmentation masks
        depth: torch.Tensor or np.ndarray (B, H, W) or (B, 1, H, W) depth maps
        depth_threshold_percentile: bottom X% of depth = deep
        brightness_threshold_percentile: bottom X% of brightness = dark
    
    Returns:
        score_maps: filtered score maps with hole regions zeroed out
    """
    B, _, H, W = score_maps.shape
    device = score_maps.device

    for i in range(B):
        # Get image as numpy HWC uint8 RGB
        img = batch[i]  # (3, H, W)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.shape[0] == 3:  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Get mask as numpy HW
        m = mask[i]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        binary_mask = (m > 0).astype(np.uint8)

        if binary_mask.sum() == 0:
            continue

        # Get depth as numpy HW float32
        d = depth[i]
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        if d.ndim == 3:
            d = d.squeeze(0)
        d = d.astype(np.float32)

        # Depth thresholding
        masked_depths = d[binary_mask > 0]
        depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
        deep_mask = (d <= depth_thresh) & (binary_mask > 0)

        # Brightness thresholding
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        masked_brightness = brightness[binary_mask > 0]
        bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
        dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)

        # Hole = deep AND dark
        hole_mask = deep_mask & dark_mask

        # Zero out score map in hole regions
        hole_tensor = torch.from_numpy(hole_mask).to(device)
        score_maps[i, 0][hole_tensor] = 0.0

    return score_maps

def find_holes_fix(image, mask, depth, save_folder=None, filename=None,
               depth_threshold_percentile=50,
               brightness_threshold_percentile=20,
               small_hole_max_area=100,
               surrounding_threshold=0.95,
               min_hole_area=100,
               dilation_radius=8,
               visualize_bool=False):
    """
    Combined depth + brightness thresholding to find hole regions.
    Only pixels that are both deep AND dark are considered holes.

    After the initial thresholding, CCA is run on the filtered-out regions.
    Each component is restored if either:
      - its border ring is less than `surrounding_threshold` fraction covered
        by non-hole raspberry pixels (it touches the background), or
      - its area in pixels is below `min_hole_area` (too small to be a hole).

    Surviving hole components can optionally be dilated outward by
    `dilation_radius` pixels (constrained to the raspberry mask).

    Args:
        image: numpy array (H, W, 3) RGB
        mask: numpy array (H, W) segmentation mask
        depth: numpy array (H, W) float32 depth map
        save_folder: path to save visualization
        filename: filename for saved figure
        depth_threshold_percentile: bottom X% of depth values = deep
        brightness_threshold_percentile: bottom X% of brightness values = dark
        small_hole_max_area: internal holes smaller than this (in pixels) get
                            filled back in with original pixel values
        surrounding_threshold: fraction [0, 1] of a hole component's border
                               that must be covered by non-hole raspberry pixels
                               for it to be kept as a hole; components below
                               this threshold are restored (default 0.99)
        min_hole_area: hole components with fewer pixels than this are always
                       restored, regardless of surrounding fraction (default 0
                       = disabled)
        dilation_radius: if > 0, each surviving hole component is dilated by
                         this many pixels in all directions, constrained to
                         the raspberry mask (default 0 = no dilation)

    Returns:
        image_cleaned: image with hole pixels zeroed out
        mask_cleaned: mask with hole pixels removed
        depth_cleaned: depth with hole pixels zeroed out
    """
    binary_mask = (mask > 0).astype(np.uint8)

    if binary_mask.sum() == 0:
        return image.copy(), mask.copy(), depth.copy()

    # Depth thresholding
    masked_depths = depth[binary_mask > 0]
    depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
    deep_mask = (depth <= depth_thresh) & (binary_mask > 0)

    # Brightness thresholding (V channel of HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2].astype(np.float32)
    masked_brightness = brightness[binary_mask > 0]
    bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
    dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)

    # Raw hole mask: deep AND dark
    raw_hole_mask = (deep_mask & dark_mask).astype(np.uint8)

    # CCA-based recovery: restore components that are not sufficiently
    # surrounded by non-hole raspberry pixels (they are edge artifacts).
    # "non-hole raspberry" = raspberry pixels not flagged as holes.
    non_hole_rasp = (binary_mask > 0) & (raw_hole_mask == 0)
    border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_hole_mask, connectivity=8)
    final_hole_mask = np.zeros_like(raw_hole_mask)
    for i in range(1, num_labels):
        comp = (labels == i).astype(np.uint8)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_hole_area:
            continue  # too small, restore
        # 1-pixel ring around the component
        border = cv2.dilate(comp, border_kernel) - comp
        border_bool = border > 0
        total_border = int(border_bool.sum())
        if total_border == 0:
            final_hole_mask[comp > 0] = 1
            continue
        surrounded_frac = float(non_hole_rasp[border_bool].sum()) / total_border
        if surrounded_frac >= surrounding_threshold:
            final_hole_mask[comp > 0] = 1  # truly interior hole, keep filtered

    hole_mask = final_hole_mask * 255

    # Apply hole mask to all three outputs
    image_cleaned = image.copy()
    image_cleaned[hole_mask > 0] = 0

    mask_cleaned = mask.copy()
    mask_cleaned[hole_mask > 0] = 0

    depth_cleaned = depth.copy()
    depth_cleaned[hole_mask > 0] = 0

    # Remove small fragments left after hole removal (keep largest component)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_cleaned.astype(np.uint8), connectivity=8
    )
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep = labels == largest
        mask_cleaned = keep.astype(mask.dtype)
        image_cleaned[~keep] = 0
        depth_cleaned[~keep] = 0

    # Fill small internal holes with original pixel values
    filled = mask_cleaned.astype(np.uint8).copy()
    inverted = cv2.bitwise_not(filled * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        touches_border = (
            x == 0 or y == 0 or
            x + w >= filled.shape[1] or
            y + h >= filled.shape[0]
        )
        if touches_border:
            continue
        if area < small_hole_max_area:
            filled[labels == i] = 1

    restore = (filled > 0) & (mask_cleaned == 0)
    mask_cleaned = filled.astype(mask.dtype)
    image_cleaned[restore] = image[restore]
    depth_cleaned[restore] = depth[restore]

    # Dilation applied last, after all restoration, constrained to raspberry mask
    if dilation_radius > 0:
        dil_size = 2 * dilation_radius + 1
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size, dil_size))
        # hole pixels = raspberry pixels that are still zeroed out after restoration
        current_hole = (binary_mask > 0) & (mask_cleaned == 0)
        expanded_hole = cv2.dilate(current_hole.astype(np.uint8), dil_kernel).astype(bool)
        expanded_hole = expanded_hole & (binary_mask > 0)
        newly_removed = expanded_hole & (mask_cleaned > 0)
        mask_cleaned[newly_removed] = 0
        image_cleaned[newly_removed] = 0
        depth_cleaned[newly_removed] = 0
        final_hole_mask = current_hole.astype(np.uint8) | newly_removed.astype(np.uint8)
        hole_mask = final_hole_mask * 255

    if visualize_bool:
        d_min, d_max = masked_depths.min(), masked_depths.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        depth_norm[binary_mask == 0] = 0
        depth_colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth_colored[binary_mask == 0] = 0

        b_min, b_max = masked_brightness.min(), masked_brightness.max()
        bright_norm = (brightness - b_min) / (b_max - b_min + 1e-8)
        bright_norm[binary_mask == 0] = 0

        overlay_depth = depth_colored.copy()
        overlay_depth[deep_mask] = [255, 0, 0]

        overlay_bright = np.stack([bright_norm] * 3, axis=-1)
        overlay_bright = (overlay_bright * 255).astype(np.uint8)
        overlay_bright[binary_mask == 0] = 0
        overlay_bright_marked = overlay_bright.copy()
        overlay_bright_marked[dark_mask] = [255, 0, 0]

        # raw hole before CCA recovery
        overlay_raw = depth_colored.copy()
        overlay_raw[raw_hole_mask > 0] = [255, 0, 0]

        # final hole after CCA recovery + dilation
        overlay_final = depth_colored.copy()
        overlay_final[hole_mask > 0] = [255, 0, 0]

        fig, axes = plt.subplots(1, 7, figsize=(35, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(depth_colored)
        axes[1].set_title("Depth Map")
        axes[1].axis('off')

        axes[2].imshow(overlay_depth)
        axes[2].set_title(f"Deep ({depth_threshold_percentile}th pct)")
        axes[2].axis('off')

        axes[3].imshow(overlay_bright_marked)
        axes[3].set_title(f"Dark ({brightness_threshold_percentile}th pct)")
        axes[3].axis('off')

        axes[4].imshow(overlay_raw)
        axes[4].set_title(f"Raw holes ({raw_hole_mask.sum()} px)")
        axes[4].axis('off')

        axes[5].imshow(overlay_final)
        axes[5].set_title(f"After CCA+dilation ({(hole_mask > 0).sum()} px)")
        axes[5].axis('off')

        axes[6].imshow(image_cleaned)
        axes[6].set_title("Cleaned")
        axes[6].axis('off')

        plt.tight_layout()
       # plt.savefig(os.path.join(str(save_folder), filename), dpi=100, bbox_inches='tight')
        plt.savefig("hole_detection_visualization.png", dpi=100, bbox_inches='tight')
        plt.close()

    return image_cleaned, mask_cleaned, depth_cleaned

def find_holes(image, mask, depth, save_folder = None, filename = None,
               depth_threshold_percentile=50,
               brightness_threshold_percentile=50, visualize_bool=False):
    """
    Combined depth + brightness thresholding to find hole regions.
    Only pixels that are both deep AND dark are considered holes.

    Currently, this will remove more than just the holes.
    It will also remove parts of the raspberry that are deep and dark.
    This might even be beneficial, because it seems like we remove
    parts that the AD models might struggle with ; meanwhile,
    we will not remove mold/white parts that are seen often
    in grade 4+5 !

    More importantly, we do this as post-processing, meaning
    the AD model will run over the normal raspberry, but for calculation
    of the heatmap, we will ignore the hole/dark regions.

    Args:
        image: numpy array (H, W, 3) RGB
        mask: numpy array (H, W) segmentation mask
        depth: numpy array (H, W) float32 depth map
        save_folder: path to save visualization
        filename: filename for saved figure
        depth_threshold_percentile: bottom X% of depth values = deep
        brightness_threshold_percentile: bottom X% of brightness values = dark

    Returns:
        image_cleaned: image with hole pixels zeroed out
        mask_cleaned: mask with hole pixels removed
        depth_cleaned: depth with hole pixels zeroed out
        hole_mask: binary mask of detected hole pixels (255 = hole)
    """
    binary_mask = (mask > 0).astype(np.uint8)

    if binary_mask.sum() == 0:
        return image.copy(), mask.copy(), depth.copy(), np.zeros_like(mask, dtype=np.uint8)

    # Depth thresholding
    masked_depths = depth[binary_mask > 0]
    depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
    deep_mask = (depth <= depth_thresh) & (binary_mask > 0)

    # Brightness thresholding (V channel of HSV)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = hsv[:, :, 2].astype(np.float32)
    masked_brightness = brightness[binary_mask > 0]
    bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
    dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)

    # Hole = deep AND dark
    hole_mask = (deep_mask & dark_mask).astype(np.uint8) * 255

    # Clean all three outputs
    image_cleaned = image.copy()
    image_cleaned[hole_mask > 0] = 0

    mask_cleaned = mask.copy()
    mask_cleaned[hole_mask > 0] = 0

    depth_cleaned = depth.copy()
    depth_cleaned[hole_mask > 0] = 0

    if visualize_bool:
        # --- Visualization ---
        d_min, d_max = masked_depths.min(), masked_depths.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        depth_norm[binary_mask == 0] = 0

        depth_colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        depth_colored[binary_mask == 0] = 0

        b_min, b_max = masked_brightness.min(), masked_brightness.max()
        bright_norm = (brightness - b_min) / (b_max - b_min + 1e-8)
        bright_norm[binary_mask == 0] = 0

        overlay_depth = depth_colored.copy()
        overlay_depth[deep_mask] = [255, 0, 0]

        overlay_bright = np.stack([bright_norm] * 3, axis=-1)
        overlay_bright = (overlay_bright * 255).astype(np.uint8)
        overlay_bright[binary_mask == 0] = 0
        overlay_bright_marked = overlay_bright.copy()
        overlay_bright_marked[dark_mask] = [255, 0, 0]

        overlay_combined = depth_colored.copy()
        overlay_combined[hole_mask > 0] = [255, 0, 0]

        fig, axes = plt.subplots(1, 6, figsize=(30, 5))

        # convert image to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(depth_colored)
        axes[1].set_title("Depth Map")
        axes[1].axis('off')

        axes[2].imshow(overlay_depth)
        axes[2].set_title(f"Deep ({depth_threshold_percentile}th pct)")
        axes[2].axis('off')

        axes[3].imshow(overlay_bright_marked)
        axes[3].set_title(f"Dark ({brightness_threshold_percentile}th pct)")
        axes[3].axis('off')

        axes[4].imshow(overlay_combined)
        axes[4].set_title(f"Deep AND Dark ({(hole_mask > 0).sum()} px)")
        axes[4].axis('off')

        # iamge cleaned to rgb
        image_cleaned = cv2.cvtColor(image_cleaned, cv2.COLOR_BGR2RGB)
        axes[5].imshow(image_cleaned)
        axes[5].set_title("Removed")
        axes[5].axis('off')

        plt.tight_layout()
        plt.savefig('hole_detection_visualization.png', dpi=100, bbox_inches='tight')
       # plt.savefig(os.path.join(str(save_folder), filename), dpi=100, bbox_inches='tight')
        plt.close()

    return image_cleaned, mask_cleaned, depth_cleaned

def overlay_mask_on_image(image, mask):
    # Create a red mask
    red_mask = np.zeros_like(image)
    green_mask = np.zeros_like(image)
    green_mask[mask > 0] = [0, 255, 0]  # Green color

    # Blend the original image with the red mask
    blended = cv2.addWeighted(image, 0.8, green_mask, 0.2, 0)

    cv2.imwrite("overlayed_image.png", blended)


def main():

    # Load the dataset (anomalous and non-anomalous samples)
    dataset_path = Path("../../nvme1/dataset_single_objects/GT/filtered_darkness_80_0.3_seed_42_256/processed")
    normal_path = dataset_path / 'normal' / 'normal_samples.pkl'
    anomalous_path = dataset_path / 'anomalous' / 'anomalous_samples.pkl'

    # Get examplary normal and anomalous sample
    with open(normal_path, 'rb') as f:
        normal_data = pickle.load(f)
    with open(anomalous_path, 'rb') as f:
        anomalous_data = pickle.load(f)

    # Concatenate normal and anomalous data
    all_images = [item['image'] for item in normal_data + anomalous_data]
    all_masks = [item['mask'] for item in normal_data + anomalous_data]
    all_depths = [item['depth'] for item in normal_data + anomalous_data]

    # Convert all images to hsv and only look at the v channel for mean and std calculation
    all_hsv_images = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV) for image in all_images]
    all_v_channels = [hsv[:,:,2] for hsv in all_hsv_images]

    all_pixels = np.concatenate([v[mask > 0] for v, mask in zip(all_v_channels, all_masks)])
    target_mean = all_pixels.mean(axis=0)
    target_std = all_pixels.std(axis=0) 

    print(f"Calculated target mean: {target_mean}, target std: {target_std}")

    
   # normal_sample = normal_data[0]['image']
   # anomalous_sample = anomalous_data[0]['image']
    print(len(normal_data), len(anomalous_data))

    # Create anomalous_depth and normal_depth folders 
   # path_normal_depth = dataset_path / 'normal' / 'normal_depth'
   # path_anomalous_depth = dataset_path / 'anomalous' / 'anomalous_depth'
   # path_normal_depth.mkdir(parents=True, exist_ok=True)
   # path_anomalous_depth.mkdir(parents=True, exist_ok=True)

    path_tests = dataset_path / 'normal' / 'tests'
    import shutil
    if path_tests.exists():
        shutil.rmtree(path_tests)
    path_tests.mkdir(parents=True, exist_ok=True)

    

    
    for i in range(len(normal_data)):

        normal_sample = normal_data[i]['image']
        normal_mask = normal_data[i]['mask']
        normal_depth = normal_data[i]['depth']
        path = normal_data[i]['img_path']
        stem_path = (Path(path)).stem + '.png'

        if stem_path == "img004_obj15_grade2.png": #"": #"img002_obj19_grade2.png": :
    

            # Convert sample to rgb
            
            normal_sample = cv2.cvtColor(normal_sample, cv2.COLOR_BGR2RGB)
            # Visualize normal img
            cv2.imwrite("original_normal_image.png", normal_sample)

           # find_contour(normal_sample, normal_mask)

        # Turn to RGB for visualization
      #  anomalous_sample = cv2.cvtColor(anomalous_sample, cv2.COLOR_BGR2RGB)
          #  cv2.imwrite("original_anomalous_image.png", anomalous_sample)

        #  edge_smoothing(anomalous_sample, anomalous_mask)
        #  overlay_mask_on_image(anomalous_sample, anomalous_mask)
        #  normalize_distribution(anomalous_sample, anomalous_mask, target_mean, target_std)
         #   find_holes(normal_sample, normal_mask, normal_depth, path_tests, stem_path, visualize_bool = True)
           # find_contour(normal_sample, normal_mask)
           # apply_cutout(normal_sample, normal_mask)
          #  apply_specular_suppression(normal_sample, normal_mask, visualize=True, inpainting=True)
           
            find_holes_fix(normal_sample, normal_mask, normal_depth, path_tests, stem_path, visualize_bool = True)
          #  clean_protrusions(normal_sample, normal_mask, normal_depth)
            break








if __name__ == "__main__":
    main()
