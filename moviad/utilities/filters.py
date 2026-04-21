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


'''
def filter_holes_batched(score_maps, batch, batch_og, mask_og, depth_og,
                         depth_threshold_percentile=50,
                         brightness_threshold_percentile=30,
                         patch_grid_size=(16, 16),
                         patch_hole_fraction=0.3):
    """
    Detect holes at original resolution, aggregate the hole mask to
    patch-grid resolution, then zero out whole patches whose hole fraction
    exceeds patch_hole_fraction.
 
    Idea: AD models struggle with raspberry hole areas (often flagged as
    anomalous but not truly anomalous). We detect holes via depth +
    brightness thresholding at the original resolution, where the signal
    is still fine-grained (downsampling depth/brightness to 224x224 or the
    patch grid first would wash out thin gaps between drupelets).
 
    Suppression is applied at patch resolution because score maps are
    produced by upsampling a coarse patch grid (e.g. DINOv2: 16x16 -> 224x224).
    Zeroing the score map at pixel resolution and then averaging back would
    only dim patches proportionally to their hole fraction rather than
    removing them; operating at patch resolution suppresses contaminated
    patches as blocks, which matches the actual structure of the score map.
 
    Args:
        score_maps: torch.Tensor (B, 1, H, W) anomaly score maps (e.g. 224x224)
        batch: torch.Tensor (B, 3, H, W) input images RGB (kept for API
            compatibility; not used inside this function)
        batch_og: torch.Tensor (B, 3, H_og, W_og) or list, original-resolution RGB
        mask_og: torch.Tensor (B, 1, H_og, W_og) or list, original-resolution masks
        depth_og: torch.Tensor (B, 1, H_og, W_og) or list, original-resolution depth
        depth_threshold_percentile: bottom X% of depth (within raspberry) = deep
        brightness_threshold_percentile: bottom X% of brightness (within raspberry) = dark
        patch_grid_size: (gh, gw) of the embedding grid. Must match the
            backbone's patch grid (e.g. (16, 16) for DINOv2 at 224 input).
        patch_hole_fraction: a patch is suppressed if at least this fraction
            of its og-resolution pixels are flagged as hole. Lower = more
            aggressive. Sweep {0.2, 0.3, 0.4} on a validation split.
 
    Returns:
        score_maps: filtered score maps, same shape as input.
    """
    B = score_maps.shape[0]
    _, _, out_h, out_w = score_maps.shape
    device = score_maps.device
 
    for i in range(B):
        # --- image to HWC uint8 RGB at og resolution ---
        img = batch_og[i]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
 
        # --- mask to HW binary at og resolution ---
        m = mask_og[i]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        binary_mask = (m > 0).astype(np.uint8)
 
        if binary_mask.sum() == 0:
            continue
 
        # --- depth to HW float32 at og resolution ---
        d = depth_og[i]
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        if d.ndim == 3:
            d = d.squeeze(0)
        d = d.astype(np.float32)
 
        # --- hole detection at og resolution ---
        masked_depths = d[binary_mask > 0]
        depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
        deep_mask = (d <= depth_thresh) & (binary_mask > 0)
 
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        masked_brightness = brightness[binary_mask > 0]
        bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
        dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)
 
        hole_mask = (deep_mask & dark_mask).astype(np.float32)  # (H_og, W_og)
 
        if hole_mask.sum() == 0:
            continue
 
        # --- aggregate hole mask: og resolution -> patch grid ---
        # adaptive_avg_pool2d gives the fraction of hole pixels per patch cell.
        hole_t = torch.from_numpy(hole_mask).unsqueeze(0).unsqueeze(0).to(device)
        patch_hole_frac = F.adaptive_avg_pool2d(hole_t, patch_grid_size)  # (1,1,gh,gw)
        patch_hole = (patch_hole_frac >= patch_hole_fraction).float()
 
        # --- upsample back to score-map resolution, block-wise ---
        hole_at_scoremap_res = F.interpolate(
            patch_hole, size=(out_h, out_w), mode='nearest'
        )  # (1, 1, H, W)
 
        score_maps[i, 0][hole_at_scoremap_res[0, 0].bool()] = 0.0
 
    return score_maps
'''


def compute_hole_mask_patchgrid(
    batch_og, mask_og, depth_og,
    patch_h, patch_w,
    depth_threshold_percentile=50,
    brightness_threshold_percentile=30,
):
    """
    Returns: torch.Tensor (B, 1, patch_h, patch_w), binary, 1 = hole (to be suppressed).
    Hole detection runs at original resolution; downsampled to patch grid via nearest.
    """
    B = len(batch_og) if isinstance(batch_og, list) else batch_og.shape[0]
    hole_masks_patch = []

    for i in range(B):
        img = batch_og[i]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        m = mask_og[i]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        binary_mask = (m > 0).astype(np.uint8)

        d = depth_og[i]
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        if d.ndim == 3:
            d = d.squeeze(0)
        d = d.astype(np.float32)

        if binary_mask.sum() == 0:
            hole_masks_patch.append(np.zeros((patch_h, patch_w), dtype=np.uint8))
            continue

        masked_depths = d[binary_mask > 0]
        depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
        deep_mask = (d <= depth_thresh) & (binary_mask > 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        masked_brightness = brightness[binary_mask > 0]
        bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
        dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)

        hole_mask_og = (deep_mask & dark_mask).astype(np.uint8)  # 0/1

        # Downsample to patch grid (nearest — binary)
        hole_mask_patch = cv2.resize(
            hole_mask_og, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST
        )
        hole_masks_patch.append(hole_mask_patch)

    return torch.from_numpy(np.stack(hole_masks_patch)).unsqueeze(1).float()


def suppress_removed_mask_regions(
    score_maps: torch.Tensor,
    mask: torch.Tensor,
    mask_unfiltered: torch.Tensor,
    influence_radius: int = 20,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Zero out anomaly scores in regions removed by mask post-processing
    (clean_protrusions / specular suppression / find_holes_fix), then smoothly
    attenuate scores in surrounding pixels based on their distance from the
    removed boundary.

    Removed region = pixels in mask_unfiltered that are absent in mask.
    Within influence_radius pixels of that region, scores are multiplied by
    (dist / influence_radius)^gamma — 0 at the boundary, rising to 1.0 at the
    radius edge. Beyond influence_radius, scores are unchanged.

    Args:
        score_maps:       (B, 1, H, W) anomaly score tensor.
        mask:             (B, 1|H, W) filtered mask.
        mask_unfiltered:  (B, 1|H, W) original unfiltered mask.
        influence_radius: Pixels out from removed region over which attenuation
                          is applied. 0 means only the removed region is zeroed.
        gamma:            Exponent of the distance ramp. 1 = linear taper;
                          <1 = stronger suppression further out;
                          >1 = more abrupt recovery to full score.

    Returns:
        score_maps modified in-place.
    """
    B, _, H, W = score_maps.shape
    device = score_maps.device

    def _to_hw_binary(m, h, w):
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        m = (m > 0).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        return m

    for i in range(B):
        m_filt = _to_hw_binary(mask[i], H, W)
        m_orig = _to_hw_binary(mask_unfiltered[i], H, W)

        removed = ((m_orig > 0) & (m_filt == 0)).astype(np.uint8)

        if removed.sum() == 0:
            continue

        # Distance of each pixel from the nearest removed pixel.
        # distanceTransform measures distance from non-zero pixels to nearest
        # zero pixel, so inverting removed gives us what we need.
        dist = cv2.distanceTransform(
            (1 - removed).astype(np.uint8), cv2.DIST_L2, maskSize=5
        ).astype(np.float32)

        if influence_radius > 0:
            weight = np.where(
                removed > 0,
                0.0,
                np.where(
                    dist <= influence_radius,
                    (dist / influence_radius) ** gamma,
                    1.0,
                ),
            ).astype(np.float32)
        else:
            weight = np.where(removed > 0, 0.0, 1.0).astype(np.float32)

        weight_tensor = torch.from_numpy(weight).unsqueeze(0).to(device)  # (1, H, W)
        score_maps[i] *= weight_tensor

    return score_maps
