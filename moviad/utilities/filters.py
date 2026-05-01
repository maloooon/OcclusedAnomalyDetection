import numpy as np
import cv2
import torch


def filter_holes_batched(batch_og, mask_og, depth_og,
                         depth_threshold_percentile=15,
                         brightness_threshold_percentile=40,
                         surrounding_threshold=0.80,
                         min_hole_area=200,
                         dilation_radius=15,
                         border_exclusion_width=30):
    """
    Detect holes at original resolution using depth + brightness thresholding,
    CCA-based recovery, and optional dilation.

    Args:
        batch_og: torch.Tensor (B, 3, H_og, W_og) or list of arrays, original RGB images
        mask_og: torch.Tensor (B, 1, H_og, W_og) or list of arrays, segmentation masks
        depth_og: torch.Tensor (B, 1, H_og, W_og) or list of arrays, depth maps
        depth_threshold_percentile: bottom X% of depth within mask = deep
        brightness_threshold_percentile: bottom X% of brightness within mask = dark
        surrounding_threshold: fraction of a hole component's border that must be
            covered by non-hole raspberry pixels to keep it as a hole
        min_hole_area: hole components smaller than this are always discarded
        dilation_radius: expand surviving holes outward by this many pixels,
            constrained to the raspberry mask; 0 disables dilation
        border_exclusion_width: inward ring from the raspberry contour excluded from
            hole detection before CCA runs; 0 disables exclusion

    Returns:
        torch.Tensor (B, 1, H_og, W_og) float, 1 = hole pixel
    """
    B = len(batch_og) if isinstance(batch_og, list) else batch_og.shape[0]
    hole_masks = []

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

        # Extract mask as numpy HW binary
        m = mask_og[i]
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        binary_mask = (m > 0).astype(np.uint8)

        og_h, og_w = binary_mask.shape

        if binary_mask.sum() == 0:
            hole_masks.append(np.zeros((og_h, og_w), dtype=np.float32))
            continue

        # Extract depth as numpy HW float32
        d = depth_og[i]
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        if d.ndim == 3:
            d = d.squeeze(0)
        d = d.astype(np.float32)

        # Depth thresholding
        masked_depths = d[binary_mask > 0]
        depth_thresh = np.percentile(masked_depths, depth_threshold_percentile)
        deep_mask = (d <= depth_thresh) & (binary_mask > 0)

        # Brightness thresholding (V channel of HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        masked_brightness = brightness[binary_mask > 0]
        bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
        dark_mask = (brightness <= bright_thresh) & (binary_mask > 0)

        # Raw hole mask: deep AND dark
        raw_hole_mask = (deep_mask & dark_mask).astype(np.uint8)

        # Border exclusion: remove detections within border_exclusion_width pixels
        # of the raspberry contour (edge darkness/shadow, not genuine holes)
        if border_exclusion_width > 0:
            excl_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * border_exclusion_width + 1, 2 * border_exclusion_width + 1),
            )
            eroded_mask = cv2.erode(binary_mask, excl_kernel)
            border_ring = (binary_mask > 0) & (eroded_mask == 0)
            raw_hole_mask[border_ring] = 0

        # CCA-based recovery: discard components that are too small or not
        # sufficiently surrounded by non-hole raspberry pixels
        non_hole_rasp = (binary_mask > 0) & (raw_hole_mask == 0)
        border_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_hole_mask, connectivity=8)
        final_hole_mask = np.zeros_like(raw_hole_mask)
        for j in range(1, num_labels):
            comp = (labels == j).astype(np.uint8)
            area = int(stats[j, cv2.CC_STAT_AREA])
            if area < min_hole_area:
                continue
            border = cv2.dilate(comp, border_kernel) - comp
            border_bool = border > 0
            total_border = int(border_bool.sum())
            if total_border == 0:
                final_hole_mask[comp > 0] = 1
                continue
            surrounded_frac = float(non_hole_rasp[border_bool].sum()) / total_border
            if surrounded_frac >= surrounding_threshold:
                final_hole_mask[comp > 0] = 1

        # Dilation constrained to raspberry mask
        if dilation_radius > 0:
            dil_size = 2 * dilation_radius + 1
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size, dil_size))
            expanded_hole = cv2.dilate(final_hole_mask, dil_kernel).astype(bool)
            final_hole_mask = (expanded_hole & (binary_mask > 0)).astype(np.uint8)

        hole_masks.append(final_hole_mask.astype(np.float32))

    return torch.from_numpy(np.stack(hole_masks)).unsqueeze(1).float()  # (B, 1, H_og, W_og)


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


def compute_darkness_mask(
    batch_og,
    mask_og,
    brightness_threshold_percentile: int = 30,
    min_component_area: int = 200,
) -> torch.Tensor:
    """
    Brightness-threshold filter at original image resolution, with CCA to
    discard small dark regions.

    After thresholding, connected components whose area is below
    min_component_area are removed from the dark mask (i.e. kept in the score
    map). Only large dark chunks are suppressed.

    Args:
        batch_og: (B, 3, H, W) tensor or list of HxWx3 uint8 RGB arrays.
        mask_og:  (B, 1, H, W) tensor or list of HxW arrays, segmentation masks.
        brightness_threshold_percentile: V-channel percentile within the mask
            below which pixels are flagged as dark.
        min_component_area: connected components strictly smaller than this
            (in pixels) are kept — only components >= this size are suppressed.

    Returns:
        (B, 1, H, W) float tensor, 1 = large dark region to suppress.
    """
    B = len(batch_og) if isinstance(batch_og, list) else batch_og.shape[0]
    dark_masks = []

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
        binary_mask = (m > 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32)
        masked_brightness = brightness[binary_mask]

        if masked_brightness.size == 0:
            dark_masks.append(np.zeros(m.shape, dtype=np.float32))
            continue

        bright_thresh = np.percentile(masked_brightness, brightness_threshold_percentile)
        dark_mask = ((brightness <= bright_thresh) & binary_mask).astype(np.uint8)

        # CCA: keep only components large enough to be genuine dark chunks
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
        filtered = np.zeros_like(dark_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                filtered[labels == label] = 1

        dark_masks.append(filtered.astype(np.float32))

    return torch.from_numpy(np.stack(dark_masks)).unsqueeze(1).float()  # (B,1,H,W)


def compute_protrusion_weight_patchgrid(
    mask: torch.Tensor,
    mask_unfiltered: torch.Tensor,
    patch_h: int,
    patch_w: int,
    influence_radius: int = 20,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Soft weight map at patch-grid resolution for protrusion-removed regions.

    Computes the distance-transform taper at original resolution (same logic as
    suppress_removed_mask_regions), then resizes to (patch_h, patch_w) via
    INTER_NEAREST — matching the pattern of compute_hole_mask_patchgrid.

    Returns: (B, 1, patch_h, patch_w) float tensor in [0, 1].
    """
    B = mask.shape[0]
    weight_maps = []

    def _to_hw_binary(m):
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        return (m > 0).astype(np.uint8)

    for i in range(B):
        m_filt = _to_hw_binary(mask[i])
        m_orig = _to_hw_binary(mask_unfiltered[i])

        removed = ((m_orig > 0) & (m_filt == 0)).astype(np.uint8)

        if removed.sum() == 0:
            weight_maps.append(np.ones((patch_h, patch_w), dtype=np.float32))
            continue

        dist = cv2.distanceTransform(
            (1 - removed).astype(np.uint8), cv2.DIST_L2, maskSize=5
        ).astype(np.float32)

        if influence_radius > 0:
            weight = np.where(
                removed > 0,
                0.0,
                np.where(dist <= influence_radius,
                         (dist / influence_radius) ** gamma,
                         1.0),
            ).astype(np.float32)
        else:
            weight = np.where(removed > 0, 0.0, 1.0).astype(np.float32)

        weight_patch = cv2.resize(weight, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST)
        weight_maps.append(weight_patch)

    return torch.from_numpy(np.stack(weight_maps)).unsqueeze(1).float()  # (B,1,ph,pw)


def suppress_removed_mask_regions(
    score_maps: torch.Tensor,
    mask: torch.Tensor = None,
    mask_unfiltered: torch.Tensor = None,
    influence_radius: int = 20,
    gamma: float = 1.0,
    removed_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Zero out anomaly scores in a removed region, then smoothly attenuate scores
    in surrounding pixels based on distance from that region.

    The removed region can be supplied in two ways:
      - ``mask`` / ``mask_unfiltered`` are always used: the removed region is
        derived as pixels present in mask_unfiltered but absent in mask.
      - ``removed_mask`` (optional): a (B, 1, H, W) binary tensor of additional
        pixels to suppress (from compute_darkness_mask currently). Unioned with the mask-diff region.

    Within influence_radius pixels of the removed region, scores are multiplied
    by (dist / influence_radius)^gamma — 0 at the boundary, rising to 1.0 at
    the radius edge. Beyond influence_radius, scores are unchanged.

    Args:
        score_maps:       (B, 1, H, W) anomaly score tensor.
        mask:             (B, 1|H, W) filtered mask (used when removed_mask is None).
        mask_unfiltered:  (B, 1|H, W) original unfiltered mask (used when removed_mask is None).
        influence_radius: Pixels out from removed region over which attenuation
                          is applied. 0 means only the removed region is zeroed.
        gamma:            Exponent of the distance ramp. 1 = linear taper;
                          <1 = stronger suppression further out;
                          >1 = more abrupt recovery to full score.
        removed_mask:     (B, 1, H, W) pre-computed binary mask of the region to
                          suppress. When given, mask/mask_unfiltered are ignored.

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

        if removed_mask is not None:
            removed = np.maximum(removed, _to_hw_binary(removed_mask[i], H, W))

        if removed.sum() == 0:
            continue

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
