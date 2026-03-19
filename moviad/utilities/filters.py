import numpy as np
import cv2 
import torch
import torch.nn.functional as F


def filter_holes_batched(score_maps, batch, batch_og, mask_og, depth_og,
                         depth_threshold_percentile=50,
                         brightness_threshold_percentile=30):
    """
    Detect holes at original resolution, resize hole mask to score_map size,
    zero out anomaly scores in hole regions.

    Idea here is that AD models seem to struggle with the raspberry hole areas (often anomalous region detected there, but not really anomalous). So we want to filter out these hole areas from the score maps. We can detect holes by depth and brightness thresholding at the original resolution, 
    then upsample the hole mask to the score map size and zero out scores in these hole regions.
    The reason for the upsampling here (and that we need to also use as parameters all og imgs) is that
    we cannot downsample the depth mask to (224,224) like the score map, since the depth map loses a lot of information that way.


    
    Args:
        score_maps: torch.Tensor (B, 1, H, W) anomaly score maps (e.g. 224x224)
        batch: torch.Tensor (B, 3, H, W) or list of numpy arrays, input images RGB
        batch_og: torch.Tensor (B, 3, H_og, W_og) or list of numpy arrays, original images RGB
        mask_og: torch.Tensor (B, 1, H_og, W_og) or list of numpy arrays, segmentation masks
        depth_og: torch.Tensor (B, 1, H_og, W_og) or list of numpy arrays, depth maps
    
    Returns:
        score_maps: filtered score maps
    """


    B = score_maps.shape[0]
    _, _, out_h, out_w = score_maps.shape
    device = score_maps.device





    for i in range(B):
        # Extract image as numpy HWC uint8 RGB
        img = batch_og[i]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Extract mask as numpy HW
        m = mask_og[i]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        binary_mask = (m > 0).astype(np.uint8)

        if binary_mask.sum() == 0:
            continue

        # Extract depth as numpy HW float32
        d = depth_og[i]
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        if d.ndim == 3:
            d = d.squeeze(0)
        d = d.astype(np.float32)


        # Depth thresholding at original resolution
        masked_depths = d[binary_mask > 0]
        depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
        deep_mask = (d <= depth_thresh) & (binary_mask > 0)

        # Brightness thresholding at original resolution
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        masked_brightness = brightness[binary_mask > 0]
        bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
        dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)

        # Hole = deep AND dark
        hole_mask = (deep_mask & dark_mask).astype(np.uint8) * 255

        if hole_mask.sum() > 0:
            og_h, og_w = hole_mask.shape[:2]
            
            # Upscale score map to og resolution
            score_map_up = F.interpolate(
                score_maps[i:i+1], size=(og_h, og_w), mode='bilinear', align_corners=False
            )
            
            # Zero out holes at og resolution
            hole_tensor = torch.from_numpy(hole_mask > 0).to(device)
            score_map_up[0, 0][hole_tensor] = 0.0
            
            # Downscale back to original score map size
            score_maps[i:i+1] = F.interpolate(
                score_map_up, size=(out_h, out_w), mode='bilinear', align_corners=False
            )
        


    return score_maps








