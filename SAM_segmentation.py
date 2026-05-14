from datasets import load_dataset
from ultralytics import SAM, YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.models.fastsam import FastSAMPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics.models.sam.amg import build_point_grid, remove_small_regions
import cv2
import os
from time import time 
from huggingface_hub import hf_hub_download
import pickle 

import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import requests
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


#from Depth_module.depth_anything_v2.dpt import DepthAnythingV2




@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


# F2 ALGORITHM
'''
def filter_overlapping_masks_extended(
    masks: np.ndarray,
    masks_depth_values: np.ndarray,
    overlap_threshold: int = 30,
    containment_threshold: float = 0.98
) -> dict:
    """
    Mask filtering with overlap analysis and containment checking.
    
    Args:
        masks: np.ndarray of shape (k, H, W) - boolean masks
        masks_depth_values: np.ndarray of shape (k, N_pixels_in_mask) containing depth values for each mask's pixels
        overlap_threshold: Minimum number of overlapping pixels to count as overlap
        containment_threshold: Fraction of mask that must be contained to count as "entirely within"
                              (e.g., 0.95 means 95% of pixels must be inside)
    
    Returns:
        dict with keys:
            - 'masks': filtered masks
            - 'kept_indices': original indices of kept masks
            - 'removed_indices': original indices of removed masks
            - 'overlap_map': canvas showing overlap counts (0, 1, 2, 3, ...)
            - 'overlap_dict': dict showing which masks overlap each mask
            - 'removal_reasons': dict explaining why each mask was removed
    
    Algorithm:
        1. Paint all masks onto canvas (sorted by size)
        2. Build overlap dictionary
        3. Apply removal rules based on overlap patterns
    """
    if len(masks) == 0:
        return {
            'masks': masks,
            'kept_indices': np.array([], dtype=int),
            'removed_indices': np.array([], dtype=int),
            'overlap_map': np.zeros((0, 0), dtype=np.int32),
            'overlap_dict': {},
            'removal_reasons': {}
        }
    
    masks = masks.astype(bool)
    k, H, W = masks.shape
    
    # Sort masks by size (largest first)
    mask_sizes = masks.sum(axis=(1, 2))
    sorted_indices = np.argsort(mask_sizes)[::-1]
    sorted_masks = masks[sorted_indices]
    
    # Phase 1: Paint all masks and build overlap map
    canvas = np.zeros((H, W), dtype=np.int32)
    overlap_dict = {i: [] for i in range(k)}  # mask_i: [masks that overlap with mask_i]
    
    for i, mask in enumerate(sorted_masks):
        # Paint this mask
        canvas[mask] += 1
    
    # Phase 2: Analyze overlaps - check each mask against the canvas
    for i, mask_i in enumerate(sorted_masks):
        # Get overlap counts for this mask's pixels
        overlap_counts = canvas[mask_i]
        
        # Find pixels where overlap > 1 (this mask + at least one other)
        overlapping_pixels = overlap_counts > 1
        num_overlapping_pixels = overlapping_pixels.sum()
        
        if num_overlapping_pixels >= overlap_threshold:
            # This mask has significant overlap
            # Find which OTHER masks it overlaps with
            # (need to check each other mask individually)
            for j, mask_j in enumerate(sorted_masks):
                if i == j:
                    continue
                
                # Check overlap between mask_i and mask_j
                overlap_ij = mask_i & mask_j
                num_overlap_ij = overlap_ij.sum()
                
                if num_overlap_ij >= overlap_threshold:
                    # mask_j overlaps with mask_i
                    # Since we're processing in size order (largest first),
                    # if i < j, mask_i is larger and mask_j overlaps it
                    # if i > j, mask_j is larger and overlaps mask_i
                    if i < j:
                        # mask_i (larger) is overlapped by mask_j (smaller)
                        overlap_dict[i].append(j)
    
    # Phase 3: Apply removal rules
    flags_for_removal = set()
    removal_reasons = {}

    # Sort the overlap_dict by longest to shortest overlapping lists
    sorted_overlap_items = sorted(overlap_dict.items(), key=lambda x: len(x[1]), reverse=True)
    overlap_dict = dict(sorted_overlap_items)

    for mask_idx, overlapping_masks in overlap_dict.items():
        num_overlaps = len(overlapping_masks)

        # Visualize current mask for debugging
   #     plt.figure()
    #    plt.imshow(sorted_masks[mask_idx], cmap='gray')
    #    plt.close()

        #Visualize overlapping masks for debugging
     #   for overlapping_idx in overlapping_masks:
     #       plt.figure()
     #       plt.imshow(sorted_masks[overlapping_idx], cmap='gray')
     #       plt.close()

        

        
        if num_overlaps == 0:
            # Rule 3: No overlap, keep mask
            continue
        

        elif num_overlaps >= 1:
            # Rule 1 & 2: Check containment for all overlapping masks
            all_contained = []
            contained_masks = []
            
            for overlapping_idx in overlapping_masks:
                is_cont = is_contained(sorted_masks[overlapping_idx], sorted_masks[mask_idx], containment_threshold)
                all_contained.append(is_cont)
                
                if is_cont:
                    contained_masks.append(overlapping_idx)
            
            # If ANY overlapping masks are NOT contained
            # then current mask likely spans multiple masks - remove it
            if not all(all_contained):
                flags_for_removal.add(mask_idx)
                removal_reasons[mask_idx] = f"Spans across multiple masks: {overlapping_masks}"
            else:
                # ALL overlapping masks are contained within current mask
                # These are likely small artifacts/noise - flag them for removal
                for overlapping_idx in contained_masks:
                    flags_for_removal.add(overlapping_idx)
                    removal_reasons[overlapping_idx] = f"Contained within mask {mask_idx}"
    
    # Create final mask selection
    keep_flags = np.array([i not in flags_for_removal for i in range(k)])
    
    # Map back to original indices
    kept_indices = sorted_indices[keep_flags]
    removed_indices = sorted_indices[~keep_flags]
    
    return {
        'masks': sorted_masks[keep_flags],
        'kept_indices': kept_indices,
        'removed_indices': removed_indices,
        'overlap_map': canvas,
        'overlap_dict': {sorted_indices[k]: [sorted_indices[v] for v in vs] 
                        for k, vs in overlap_dict.items()},  # Map back to original indices
        'removal_reasons': {sorted_indices[k]: reason 
                           for k, reason in removal_reasons.items()}
    }
'''


# OLD , DIDNT WORK THAT WELL
'''
def filter_overlapping_masks_extended(
    masks: np.ndarray,
    masks_depth_values: np.ndarray,
    overlap_threshold: int = 30,
    containment_threshold: float = 0.98,
    depth_range_threshold: float = 0.7,  # Threshold for bucket size ratio (e.g., 0.7 means one bucket must be >= 70% to skip depth check)
    enclosure_threshold: float = 0.7,  # Fraction of one bucket that must be surrounded by the other
    area_fill_threshold: float = 0.5  # Fraction of deleted mask area that must be filled by other masks to keep it deleted
) -> dict:
    """
    Mask filtering with overlap analysis, containment checking, and depth-based filtering.
    
    Args:
        masks: np.ndarray of shape (k, H, W) - boolean masks
        masks_depth_values: np.ndarray of shape (k, H, W) containing depth values for each mask
        overlap_threshold: Minimum number of overlapping pixels to count as overlap
        containment_threshold: Fraction of mask that must be contained to count as "entirely within"
        depth_range_threshold: Bucket size ratio threshold (0-1). If one bucket is >= this fraction, skip depth filtering
        enclosure_threshold: Fraction of pixels that must be enclosed to count as "surrounded"
        area_fill_threshold: Fraction of deleted mask area that must be filled by remaining masks
    
    Returns:
        dict with keys:
            - 'masks': filtered masks
            - 'kept_indices': original indices of kept masks
            - 'removed_indices': original indices of removed masks
            - 'overlap_map': canvas showing overlap counts
            - 'overlap_dict': dict showing which masks overlap each mask
            - 'removal_reasons': dict explaining why each mask was removed
    """
    if len(masks) == 0:
        return {
            'masks': masks,
            'kept_indices': np.array([], dtype=int),
            'removed_indices': np.array([], dtype=int),
            'overlap_map': np.zeros((0, 0), dtype=np.int32),
            'overlap_dict': {},
            'removal_reasons': {}
        }
    
    masks = masks.astype(bool)
    k, H, W = masks.shape
    
    # Sort masks by size (largest first)
    mask_sizes = masks.sum(axis=(1, 2))
    sorted_indices = np.argsort(mask_sizes)[::-1]
    sorted_masks = masks[sorted_indices]
    sorted_depths = masks_depth_values[sorted_indices]
    
    # Phase 1: Paint all masks and build overlap map
    canvas = np.zeros((H, W), dtype=np.int32)
    overlap_dict = {i: [] for i in range(k)}
    
    for i, mask in enumerate(sorted_masks):
        canvas[mask] += 1
    
    # Phase 2: Analyze overlaps
    for i, mask_i in enumerate(sorted_masks):
        overlap_counts = canvas[mask_i]
        overlapping_pixels = overlap_counts > 1
        num_overlapping_pixels = overlapping_pixels.sum()
        
        if num_overlapping_pixels >= overlap_threshold:
            for j, mask_j in enumerate(sorted_masks):
                if i == j:
                    continue
                
                overlap_ij = mask_i & mask_j
                num_overlap_ij = overlap_ij.sum()
                
                if num_overlap_ij >= overlap_threshold:
                    if i < j:
                        overlap_dict[i].append(j)
    
    # Calculate bucket size ratios for sorting
    bucket_ratios = {}
    for mask_idx in overlap_dict.keys():
        depth_vals = sorted_depths[mask_idx]
        depth_vals_valid = depth_vals[depth_vals > 0]
        
        if len(depth_vals_valid) > 0:
            median_depth = np.median(depth_vals_valid)
            mask_pixels = sorted_masks[mask_idx]
            
            bucket1_mask = (depth_vals > 0) & (depth_vals <= median_depth) & mask_pixels
            bucket2_mask = (depth_vals > 0) & (depth_vals > median_depth) & mask_pixels
            
            bucket1_size = bucket1_mask.sum()
            bucket2_size = bucket2_mask.sum()
            total_size = bucket1_size + bucket2_size
            
            if total_size > 0:
                # Calculate the ratio of the larger bucket
                bucket_ratios[mask_idx] = max(bucket1_size, bucket2_size) / total_size
            else:
                bucket_ratios[mask_idx] = 1.0
        else:
            bucket_ratios[mask_idx] = 1.0
    
    # Sort by smallest bucket ratio first (most balanced = potentially problematic)
    sorted_overlap_items = sorted(overlap_dict.items(), key=lambda x: bucket_ratios.get(x[0], 1.0))
    overlap_dict = dict(sorted_overlap_items)
    
    # Phase 3: Apply removal rules with immediate deletion
    removed_indices_set = set()
    removal_reasons = {}
    
    # Keep iterating until no more deletions occur
    while True:
        deleted_this_iteration = False
        
        for mask_idx in list(overlap_dict.keys()):
            if mask_idx in removed_indices_set:
                continue
                
            overlapping_masks = [idx for idx in overlap_dict[mask_idx] if idx not in removed_indices_set]
            num_overlaps = len(overlapping_masks)
            
            if num_overlaps == 0:
                continue

            # Visualize current mask 
        #    plt.figure()
        #    plt.imshow(sorted_masks[mask_idx], cmap='gray')
        #    plt.title(f"Current mask idx {mask_idx} with bucket ratio {bucket_ratios.get(mask_idx, 1.0):.3f}")
        #    plt.close()
            
            # Check depth-based removal first
            depth_vals = sorted_depths[mask_idx]
            depth_vals_valid = depth_vals[depth_vals > 0] # 0 values are black background (i.e. not part of the "depth mask" of the rasperry), so we filter them out
            
            if len(depth_vals_valid) > 0:
                # Create two depth buckets (binary split at median)
                median_depth = np.median(depth_vals_valid)
                
                # Create masks for each bucket
                mask_pixels = sorted_masks[mask_idx]
                bucket1_mask = (depth_vals > 0) & (depth_vals <= median_depth) & mask_pixels
                bucket2_mask = (depth_vals > 0) & (depth_vals > median_depth) & mask_pixels
                
                bucket1_size = bucket1_mask.sum()
                bucket2_size = bucket2_mask.sum()
                total_size = bucket1_size + bucket2_size
                
                if total_size > 0:
                    bucket1_ratio = bucket1_size / total_size
                    bucket2_ratio = bucket2_size / total_size
                    max_bucket_ratio = max(bucket1_ratio, bucket2_ratio)
                    
                    # Only check enclosure if buckets are relatively balanced (not dominated by one)
                    if max_bucket_ratio < depth_range_threshold: # Not needed per-se, but higher efficiency (i.e. we do not need to check everything)
                        # Check if one bucket is NOT enclosed by the other
                        bucket1_enclosed = is_enclosed_by(bucket1_mask, bucket2_mask, enclosure_threshold)
                        bucket2_enclosed = is_enclosed_by(bucket2_mask, bucket1_mask, enclosure_threshold)
                        
                        # visualize the buckets for debugging ; draw buckets in colors red and blue
                     #   plt.figure(figsize=(12, 6))

                        # Create RGB composite for both subplots
                      #  composite1 = np.zeros((bucket1_mask.shape[0], bucket1_mask.shape[1], 3))
                        #composite1[bucket1_mask, 2] = 1  # Bucket 1 in blue
                       # composite1[bucket2_mask, 0] = 1  # Bucket 2 in red

                        #composite2 = composite1.copy()  # Same composite for both

                     #   plt.subplot(1, 2, 1)
                     #   plt.imshow(composite1)
                     #   plt.title(f"Bucket 1 (blue, <= median depth {median_depth:.3f})\nEnclosed by Bucket 2: {bucket1_enclosed}")

                      #  plt.subplot(1, 2, 2)
                       # plt.imshow(composite2)
                       # plt.title(f"Bucket 2 (red, > median depth {median_depth:.3f})\nEnclosed by Bucket 1: {bucket2_enclosed}")

                        #plt.suptitle(f"Depth-based filtering for mask idx {mask_idx} with bucket ratio {max_bucket_ratio:.3f}", fontsize=16)
                        #plt.tight_layout()
                       # plt.show()
                      #  plt.close()
                        

                        if not bucket1_enclosed and not bucket2_enclosed:
                            # Neither bucket encloses the other - delete this mask
                            removed_indices_set.add(mask_idx)
                            removal_reasons[mask_idx] = f"Balanced depth buckets ({bucket1_ratio:.2f}/{bucket2_ratio:.2f}) with non-enclosed regions"
                            
                            # Remove from overlap_dict as key
                            del overlap_dict[mask_idx]
                            
                            # Remove from all values in overlap_dict
                            for key in overlap_dict:
                                if mask_idx in overlap_dict[key]:
                                    overlap_dict[key].remove(mask_idx)
                            
                            deleted_this_iteration = True
                            continue
            
            # Standard containment logic
            all_contained = []
            contained_masks = []
            
            for overlapping_idx in overlapping_masks:
                is_cont = is_contained(sorted_masks[overlapping_idx], sorted_masks[mask_idx], containment_threshold)
                all_contained.append(is_cont)
                
                if is_cont:
                    contained_masks.append(overlapping_idx)


        
            
          #  if not all(all_contained):
                # Current mask spans multiple masks - remove it
          #      removed_indices_set.add(mask_idx)
           #     removal_reasons[mask_idx] = f"Spans across multiple masks: {overlapping_masks}"
                
                # Remove from overlap_dict
           #     del overlap_dict[mask_idx]
             #   for key in overlap_dict:
            #        if mask_idx in overlap_dict[key]:
              #          overlap_dict[key].remove(mask_idx)
                
              #  deleted_this_iteration = True
            #else:
            # Remove fully contained masks
            for overlapping_idx in contained_masks:
                if overlapping_idx not in removed_indices_set:
                    removed_indices_set.add(overlapping_idx)
                    removal_reasons[overlapping_idx] = f"Contained within mask {mask_idx}"
                    
                    # Remove from overlap_dict
                    if overlapping_idx in overlap_dict:
                        del overlap_dict[overlapping_idx]
                    for key in overlap_dict:
                        if overlapping_idx in overlap_dict[key]:
                            overlap_dict[key].remove(overlapping_idx)
                    
                    deleted_this_iteration = True
        
        # If no deletions occurred this iteration, we're done
        if not deleted_this_iteration:
            break
    
    # Phase 4: Check if deleted masks should be restored based on area coverage
    
    masks_to_restore = set()
    
    for removed_idx in list(removed_indices_set):
        removed_mask = sorted_masks[removed_idx]
        removed_mask_area = removed_mask.sum()
        
        # Calculate how much of this area is now covered by remaining masks
        covered_area = 0
        for keep_idx in range(k):
            if keep_idx not in removed_indices_set:
                overlap = removed_mask & sorted_masks[keep_idx]
                covered_area += overlap.sum()
        
        # Calculate coverage fraction
        if removed_mask_area > 0:
            coverage_fraction = covered_area / removed_mask_area
        else:
            coverage_fraction = 0
        
        # If coverage is below threshold, restore the mask
        if coverage_fraction < area_fill_threshold:
            masks_to_restore.add(removed_idx)
            removal_reasons[removed_idx] = removal_reasons[removed_idx] + f" [RESTORED: only {coverage_fraction:.2%} coverage]"
    
    # Restore masks
    for restore_idx in masks_to_restore:
        removed_indices_set.remove(restore_idx)
    
    # Create final mask selection
    keep_flags = np.array([i not in removed_indices_set for i in range(k)])
    
    # Map back to original indices
    kept_indices = sorted_indices[keep_flags]
    removed_indices = sorted_indices[~keep_flags]
    
    return {
        'masks': sorted_masks[keep_flags],
        'kept_indices': kept_indices,
        'removed_indices': removed_indices,
        'overlap_map': canvas,
        'overlap_dict': {sorted_indices[k]: [sorted_indices[v] for v in vs] 
                        for k, vs in overlap_dict.items() if k not in removed_indices_set},
        'removal_reasons': {sorted_indices[k]: reason 
                           for k, reason in removal_reasons.items()}
    }


def is_enclosed_by(inner_mask: np.ndarray, outer_mask: np.ndarray, threshold: float = 0.7) -> bool:
    """
    Check if inner_mask pixels are enclosed by outer_mask by testing if each pixel
    can "escape" to the image boundary without crossing the outer mask.
    
    Args:
        inner_mask: Boolean mask of the potentially enclosed region
        outer_mask: Boolean mask of the potentially enclosing region
        threshold: Fraction of inner mask pixels that must be enclosed
    
    Returns:
        True if at least 'threshold' fraction of inner_mask pixels are enclosed
    """
    from scipy.ndimage import label
    
    # Get all inner mask pixel coordinates
    inner_coords = np.argwhere(inner_mask)
    
    if len(inner_coords) == 0:
        return False
    
    # Create a "free space" map: regions where you can move (not outer mask)
    # Include inner mask itself as free space
    free_space = ~outer_mask
    
    # Label connected components in free space
    # Each component is a region of connected free space
    labeled_free_space, num_components = label(free_space)
    
    # Find which components touch the image boundary (these are "escape routes")
    H, W = inner_mask.shape
    boundary_labels = set()
    
    # Check all four edges
    boundary_labels.update(labeled_free_space[0, :])      # Top edge
    boundary_labels.update(labeled_free_space[-1, :])     # Bottom edge
    boundary_labels.update(labeled_free_space[:, 0])      # Left edge
    boundary_labels.update(labeled_free_space[:, -1])     # Right edge
    
    # Remove 0 (background label)
    boundary_labels.discard(0)
    
    # Count how many inner pixels can escape
    enclosed_count = 0
    
    for y, x in inner_coords:
        pixel_label = labeled_free_space[y, x]
        
        # If this pixel's free space component doesn't touch the boundary, it's enclosed
        if pixel_label not in boundary_labels:
            enclosed_count += 1
    
    # Calculate fraction of enclosed pixels
    total_inner_pixels = len(inner_coords)
    enclosed_fraction = enclosed_count / total_inner_pixels
    
    return enclosed_fraction >= threshold
'''


 # F2 DEPTH ALGORITHM
def filter_overlapping_masks_extended(
    masks: np.ndarray,
    masks_depth_values: np.ndarray,
    overlap_threshold: int = 30,
    containment_threshold: float = 0.95,
    depth_difference_threshold: float = 10.0,  # Minimum depth difference to consider masks separate
    debug: bool = False
) -> dict:
    """
    Overlap mask filtering based on containment and depth analysis.
    
    
    Args:
        masks: np.ndarray of shape (k, H, W) - boolean masks
        masks_depth_values: np.ndarray of shape (k, H, W) containing depth values for each mask
        overlap_threshold: Minimum number of overlapping pixels to count as overlap
        containment_threshold: Fraction of mask that must be contained to count as "fully contained"
        depth_difference_threshold: Minimum depth difference to remove key mask (when 2 contained masks)
        debug: Whether to print debug information
    
    Returns:
        dict with keys:
            - 'masks': filtered masks
            - 'kept_indices': original indices of kept masks
            - 'removed_indices': original indices of removed masks
            - 'overlap_map': canvas showing overlap counts
            - 'overlap_dict': dict showing which masks overlap each mask
            - 'removal_reasons': dict explaining why each mask was removed
    """
    if len(masks) == 0:
        return {
            'masks': masks,
            'kept_indices': np.array([], dtype=int),
            'removed_indices': np.array([], dtype=int),
            'overlap_map': np.zeros((0, 0), dtype=np.int32),
            'overlap_dict': {},
            'removal_reasons': {}
        }
    
    masks = masks.astype(bool)
    k, H, W = masks.shape
    
    # Sort masks by size (largest first)
    mask_sizes = masks.sum(axis=(1, 2))
    sorted_indices = np.argsort(mask_sizes)[::-1]
    sorted_masks = masks[sorted_indices]
    sorted_depths = masks_depth_values[sorted_indices]
    
    # Phase 1: Build overlap map and find overlapping pairs
    canvas = np.zeros((H, W), dtype=np.int32)
    overlap_dict = {i: [] for i in range(k)}
    
    for i, mask in enumerate(sorted_masks):
        canvas[mask] += 1
    
    # Phase 2: Analyze overlaps and build overlap dictionary
    for i, mask_i in enumerate(sorted_masks):
        overlap_counts = canvas[mask_i]
        overlapping_pixels = overlap_counts > 1
        num_overlapping_pixels = overlapping_pixels.sum()
        
        if num_overlapping_pixels >= overlap_threshold:
            for j, mask_j in enumerate(sorted_masks):
                if i == j:
                    continue
                
                overlap_ij = mask_i & mask_j
                num_overlap_ij = overlap_ij.sum()
                
                if num_overlap_ij >= overlap_threshold:
                    if j not in overlap_dict[i]:  # Avoid duplicates
                        overlap_dict[i].append(j)
    
    # Phase 3: Apply removal logic
    removed_indices_set = set()
    removal_reasons = {}
    
    for mask_idx in range(k):
        if mask_idx in removed_indices_set:
            continue

        # Plot current mask 
        if debug:
            plt.figure()
            plt.imshow(sorted_masks[mask_idx], cmap='gray')
            plt.title(f"Analyzing mask idx {mask_idx}")
            plt.close()
        
        overlapping_masks = [idx for idx in overlap_dict[mask_idx] if idx not in removed_indices_set]
        
        if len(overlapping_masks) == 0:
            continue
        
        # Find which overlapping masks are fully contained within current mask
        contained_masks = []
        for overlap_idx in overlapping_masks:
            if is_contained(sorted_masks[overlap_idx], sorted_masks[mask_idx], containment_threshold):
                contained_masks.append(overlap_idx)
        
        num_contained = len(contained_masks)
        
        if debug:
            print(f"\nMask {mask_idx}: {num_contained} fully contained masks")
        
        # Apply rules based on number of contained masks
        if num_contained == 1:
            # Remove the single contained mask
            contained_idx = contained_masks[0]
            removed_indices_set.add(contained_idx)
            removal_reasons[contained_idx] = f"Single mask contained within mask {mask_idx}"
            
            if debug:
                print(f"  → Removing mask {contained_idx} (single contained)")
        
        elif num_contained == 2:
            # Check depth difference in facing region
            mask1_idx, mask2_idx = contained_masks[0], contained_masks[1]
            
      
            
            # Get depth values in facing region for each mask
            depth_key = sorted_depths[mask_idx]
            depth_mask1 = sorted_depths[mask1_idx]
            depth_mask2 = sorted_depths[mask2_idx]
            
            # Calculate average depth for each mask 
            # Only consider pixels where depth > 0 (valid depth values)
            mask1_region = sorted_masks[mask1_idx]
            mask2_region = sorted_masks[mask2_idx]
            
            depth1_values = depth_mask1[mask1_region & (depth_mask1 > 0)]
            depth2_values = depth_mask2[mask2_region & (depth_mask2 > 0)]
            
     
            
            avg_depth1 = np.mean(depth1_values)
            avg_depth2 = np.mean(depth2_values)
            depth_diff = abs(avg_depth1 - avg_depth2)
            
            if debug:
                print(f"  → Mask {mask1_idx} avg depth: {avg_depth1:.2f}")
                print(f"  → Mask {mask2_idx} avg depth: {avg_depth2:.2f}")
                print(f"  → Depth difference: {depth_diff:.2f} (threshold: {depth_difference_threshold})")
            
            if depth_diff >= depth_difference_threshold:
                # Depth difference is significant - remove the key mask
                removed_indices_set.add(mask_idx)
                removal_reasons[mask_idx] = (
                    f"Contains 2 masks ({mask1_idx}, {mask2_idx}) with depth difference "
                    f"{depth_diff:.2f} >= {depth_difference_threshold}"
                )
            
                
                if debug:
                    print(f"  → Removing key mask {mask_idx} (sufficient depth difference)")
            else:
                # Remove the contained masks instead (they are likely noise/artifacts)
                for contained_idx in contained_masks:
                    removed_indices_set.add(contained_idx)
                    removal_reasons[contained_idx] = (
                        f"Contained within mask {mask_idx} with insufficient depth difference "
                        f"({depth_diff:.2f} < {depth_difference_threshold})"
                    )
                if debug:
                    print(f"  → Removing contained masks {contained_masks} (insufficient depth difference)")
        
        elif num_contained >= 3:
            # For now, just remove all contained masks (could be improved with more complex logic)
            for contained_idx in contained_masks:
                removed_indices_set.add(contained_idx)
                removal_reasons[contained_idx] = f"Multiple ({num_contained}) masks contained within mask {mask_idx}"
    
    # Phase 4: Create final output
    keep_flags = np.array([i not in removed_indices_set for i in range(k)])
    
    # Map back to original indices
    kept_indices = sorted_indices[keep_flags]
    removed_indices = sorted_indices[~keep_flags]
    
    return {
        'masks': sorted_masks[keep_flags],
        'kept_indices': kept_indices,
        'removed_indices': removed_indices,
        'overlap_map': canvas,
        'overlap_dict': {sorted_indices[k]: [sorted_indices[v] for v in vs] 
                        for k, vs in overlap_dict.items() if k not in removed_indices_set},
        'removal_reasons': {sorted_indices[k]: reason 
                           for k, reason in removal_reasons.items()}
    }


def is_contained(mask_inner: np.ndarray, mask_outer: np.ndarray, threshold: float = 0.95) -> bool:
    """
    Check if mask_inner is (mostly) contained within mask_outer.
    
    Args:
        mask_inner: The mask to check if contained
        mask_outer: The mask to check containment within
        threshold: Fraction of mask_inner that must overlap with mask_outer
    
    Returns:
        True if at least `threshold` fraction of mask_inner overlaps with mask_outer
    
    Examples:
        >>> inner = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        >>> outer = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
        >>> is_contained(inner, outer, threshold=0.95)
        True
    """
    inner_size = mask_inner.sum()
    
    if inner_size == 0:
        return True  # Empty mask is contained in anything
    
    # Count how many pixels of inner overlap with outer
    overlap = (mask_inner & mask_outer).sum()
    
    # Calculate fraction contained
    fraction_contained = overlap / inner_size
    
    return fraction_contained >= threshold

def _filter_masks_by_darkness(masks, img_lab,img_array, plot_boxplot = False, plot_removed = False):
    """
    Filter masks based on LAB L* (lightness) channel.
    Lower L* = darker objects.
    
    Args:
        masks: numpy array of shape (N, H, W) containing binary masks
        img_lab: LAB image as numpy array of shape (H, W, 3)
        img_array : Original RGB image as numpy array of shape (H, W, 3)
        plot_boxplot: whether to plot boxplot of mean lightness values for all masks
        plot_removed: whether to plot the removed masks for visual inspection
    
    Returns:
        valid_idx: numpy array of indices of masks to keep
    """
    
    valid_idx = []
    mean_lightness_values = []
    
    for i, mask in enumerate(masks):
        # Get L* values within the mask
        masked_lab = img_lab[mask > 0]
        
        if len(masked_lab) == 0:
            continue
            
        # Calculate mean L* (lightness)
        mean_L = np.mean(masked_lab[:, 0])
        mean_lightness_values.append(mean_L)
    
    # Also get mask sizes
    mask_sizes = [mask.sum() for mask in masks]

    # Remove masks that are too dark based on whether they are a 'flier' in the boxplot (i.e. below Q1 - 1.5*IQR)
    # AND that are very small (e.g. 'flier' in boxplot of mask sizes)
    Q1_darkness = np.percentile(mean_lightness_values, 25)
    Q3_darkness = np.percentile(mean_lightness_values, 75)
    IQR_darkness = Q3_darkness - Q1_darkness
    lower_bound_darkness = Q1_darkness - 1.5 * IQR_darkness

    Q1_sizes = np.percentile(mask_sizes, 25)
    Q3_sizes = np.percentile(mask_sizes, 75)
    IQR_sizes = Q3_sizes - Q1_sizes
    lower_bound_size = Q1_sizes - 1.5 * IQR_sizes

        
    valid_idx = [i for i, mean_L in enumerate(mean_lightness_values) if mean_L >= lower_bound_darkness and mask_sizes[i] >= lower_bound_size]

    valid_idx = np.array(valid_idx)

    if plot_boxplot:
        # Plot both boxplots in one figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Boxplot of light values
        axes[0].boxplot(mean_lightness_values, vert=False)
        axes[0].set_xlabel('Mean L* (Lightness) - Lower = Darker', fontsize=12)
        axes[0].set_title('Boxplot of Mean Lightness per Mask', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        # Boxplot of mask sizes
        axes[1].boxplot(mask_sizes, vert=False)
        axes[1].set_xlabel('Mask Size (Number of Pixels)', fontsize=12)
        axes[1].set_title('Boxplot of Mask Sizes', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    if plot_removed:
        # Plot original image + removed masks for visual inspection
        removed_indices = [i for i in range(len(masks)) if i not in valid_idx]
        num_removed = len(removed_indices)
        cols = 5
        total_plots = num_removed + 1  # +1 for original image
        rows = (total_plots + cols - 1) // cols
        plt.figure(figsize=(15, 3 * rows))
        # Plot the original image in the first subplot
        plt.subplot(rows, cols, 1)
        plt.imshow(img_array)
        plt.title('Original Image')
        plt.axis('off')
        # Plot removed masks
        for idx, mask_idx in enumerate(removed_indices):
            plt.subplot(rows, cols, idx + 2)
            plt.imshow(masks[mask_idx], cmap='gray')
            plt.title(f'Mask {mask_idx}\nMean L*: {mean_lightness_values[mask_idx]:.1f}\nSize: {mask_sizes[mask_idx]}')
            plt.axis('off')
        plt.suptitle('Original Image and Removed Masks Based on Darkness and Size Filtering', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    # Print statistics
    if len(mean_lightness_values) > 0:
        print(f"Darkness filtering: {len(valid_idx)}/{len(masks)} masks kept")
        print(f"  L* values - min: {np.min(mean_lightness_values):.1f}, max: {np.max(mean_lightness_values):.1f}, mean: {np.mean(mean_lightness_values):.1f}")
        print(f"  Threshold: L* >= {lower_bound_darkness}")
        print(f"  Mask sizes - min: {np.min(mask_sizes):.1f}, max: {np.max(mask_sizes):.1f}, mean: {np.mean(mask_sizes):.1f}")
        print(f"  Threshold: Size >= {lower_bound_size}")
    
    return valid_idx

# Visualization helper
def visualize_overlap_map(overlap_map: np.ndarray, title: str = "Overlap Map"):
    """Visualize the overlap map with color coding."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(overlap_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Number of overlapping masks')
    plt.title(title)
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # Add text showing unique values
    unique_vals = np.unique(overlap_map)
    plt.text(0.02, 0.98, f'Values: {unique_vals}', 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def extract_bonnet_bounding_box(sample, visualize_bool = False):
    """
    When we run SAM on the raspberry images, we want to focus only on the bonnet area.
    
    """

    img = sample['image'] # Get PIL image
    labels = sample['labels'] # Get label and segmentation mask points

    img_array = np.array(img)
    width, height = img.size

    # Extract labels of bonnet (class 0)
    bonnet_label = None
    for label_data in labels:
        class_id = int(label_data[0])
        if class_id == 0:
            bonnet_data = label_data
            break

    coords = bonnet_data[1:]

    coords = np.array(coords).reshape(-1, 2)
    polygon_points = coords * [width, height]

    x_coords = polygon_points[:, 0]
    y_coords = polygon_points[:, 1]

    min_max_bbox = np.array([x_coords.min(), y_coords.min(), 
                             x_coords.max(), y_coords.max()])
    
    if visualize_bool:

        fig, ax = plt.subplots(1)
        ax.imshow(img_array)

        # Create a Rectangle patch
        rect = patches.Rectangle((min_max_bbox[0], min_max_bbox[1]), 
                                 min_max_bbox[2] - min_max_bbox[0], 
                                 min_max_bbox[3] - min_max_bbox[1], 
                                 linewidth=2, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        plt.show()

    return list(min_max_bbox)

def model_SAM3(samples, save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes = (True, 0.2, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None), prompt = "raspberry", filter_red=(True, 0.4), filter_holes_islands=False, filter_overlap_masks=False):
    """
    SAM3 segmentation model inference and mask saving.   
    
  
    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process (i.e. processing takes a long time, so for testing we can limit this).
    
    
    """


    if type(samples) is not list:
        samples = [samples]


    overrides = dict(
        conf=0.75, 
        task="segment",
        mode="predict",
        model="../../disk/pretrained_models/sam3.pt",
        imgsz = 1008,
        half=True, 
        save=False,
        device = 2 if torch.cuda.is_available() else "cpu"
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # Load depth pipeline only when needed
    depth_pipe = None
    if filter_overlap_masks:
        depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=2 if torch.cuda.is_available() else 'cpu')

    masks_list = []
    img_ids_list = []
    xyn_list = []
    sample_imgs = []
    conf_scores_list = []
    # Let us only look at the first k samples for testing
    if type(testing_samples) is int:
        samples = samples[:testing_samples]

    times_image_encoding = []
    times_inference = []
    times_red = []
    times_bbox = []
    times_mask_shape = []
    times_mask_size = []
    times_holes_islands = []
    times_depth = []
    times_overlap = []

    for loop_idx, sample in enumerate(samples):

        sample_img = sample['image']
        sample_idx = sample['image_id']
        start_time_0 = time()
        predictor.set_image(sample_img)
        end_time_0 = time()
        print(f"Image encoding time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        if loop_idx > 0:
            times_image_encoding.append(end_time_0 - start_time_0)
        start_time = time()
        results = predictor(text=[prompt])
        end_time = time()
        print(f"Inference (prompt encoding and mask decoding) time for sample {sample_idx}: {end_time - start_time:.2f} seconds")
        if loop_idx > 0:
            times_inference.append(end_time - start_time)

        # 1. Red filter — discard non-raspberry masks before any other filtering
        if results[0].masks is not None and filter_red[0]:
            _t0 = time()
            masks_np = results[0].masks.data.cpu().numpy()
            red_valid_idx = _filter_red_masks(masks_np, np.array(sample_img), min_red_fraction=filter_red[1])
            results[0].boxes = results[0].boxes[red_valid_idx]
            results[0].masks = results[0].masks[red_valid_idx]
            _t1 = time()
            print(f"Red filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_red.append(_t1 - _t0)

        # 2. Calculate median area and filter outliers
        boxes = results[0].boxes.xyxy.cpu().numpy()

        if len(boxes) > 0 and filter_bboxes[0]:
            _t0 = time()
            valid_idx, median_area = _filter_bbox_sizes(
                boxes,
                upper_multiplier= filter_bboxes[2],
                lower_multiplier= filter_bboxes[1],
                save_distribution=save_imgs_bool,
                filename='area_distribution.png'
            )
            results[0].boxes = results[0].boxes[valid_idx]
            _t1 = time()
            print(f"Bbox size filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_bbox.append(_t1 - _t0)

            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]

                if filter_masks_shapes[0]:
                    _t0 = time()
                    # Filter by mask shape (rectangularity)
                    masks_np = results[0].masks.data.cpu().numpy()
                    boxes_np = results[0].boxes.xyxy.cpu().numpy()
                    shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
                    results[0].boxes = results[0].boxes[shape_valid_idx]
                    results[0].masks = results[0].masks[shape_valid_idx]
                    _t1 = time()
                    print(f"Mask shape filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
                    if loop_idx > 0:
                        times_mask_shape.append(_t1 - _t0)

                if filter_masks_sizes[0]:
                    _t0 = time()
                    # Filter by actual mask size
                    masks_np = results[0].masks.data.cpu().numpy()
                    size_valid_idx, median_mask_area = _filter_mask_sizes(
                        masks_np,
                        upper_multiplier= filter_masks_sizes[2], # No upper limit, as we have filtered out via large bboxes already
                        lower_multiplier= filter_masks_sizes[1],  # Filter small masks
                        save_distribution=save_imgs_bool,
                        filename='mask_size_distribution.png'
                    )
                    results[0].boxes = results[0].boxes[size_valid_idx]
                    results[0].masks = results[0].masks[size_valid_idx]
                    _t1 = time()
                    print(f"Mask size filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
                    if loop_idx > 0:
                        times_mask_size.append(_t1 - _t0)

        # 5. Holes-and-islands cleanup — refine each mask's shape morphologically
        if filter_holes_islands and results[0].masks is not None:
            start_time_0 = time()
            masks = results[0].masks.data.cpu().numpy()
            islands_threshold = int(min(m.astype(np.bool_).sum() for m in masks)) - 1
            refined_masks = []
            for mask in masks:
                mask = mask.astype(np.bool_)
                refined_mask, _ = remove_small_regions(mask, islands_threshold, mode='islands')
                refined_mask, _ = remove_small_regions(refined_mask, 20000, mode='holes')
                refined_masks.append(refined_mask)
            refined_masks = np.array(refined_masks)
            results[0].masks.data = torch.from_numpy(refined_masks).float()
            end_time_0 = time()
            print(f"Holes/islands filter time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_holes_islands.append(end_time_0 - start_time_0)

        # 6. Overlap filter — keep one mask when two masks overlap/contain each other, guided by depth
        if filter_overlap_masks and results[0].masks is not None:
            _t0 = time()
            depth_array = np.array(depth_pipe(sample_img)["depth"])
            _t1 = time()
            print(f"Depth estimation time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_depth.append(_t1 - _t0)
            start_time_0 = time()
            masks = results[0].masks.data.cpu().numpy()
            masks_before_overlap = masks.copy()
            masks_depth_values = np.array([depth_array * mask for mask in masks])
            masks_filtered_dict = filter_overlapping_masks_extended(
                masks,
                masks_depth_values,
                overlap_threshold=50,
                containment_threshold=0.95,
                depth_difference_threshold=40,
                debug=False
            )
            results[0].masks = results[0].masks[masks_filtered_dict['kept_indices']]
            results[0].boxes = results[0].boxes[masks_filtered_dict['kept_indices']]
            end_time_0 = time()
            print(f"Overlap filter time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_overlap.append(end_time_0 - start_time_0)

            if save_imgs_bool:
                img_array = np.array(sample_img)
                for mask in masks_filtered_dict['masks']:
                    color = np.random.randint(0, 255, 3).tolist()
                    color_mask = np.zeros_like(img_array)
                    color_mask[mask == True] = color
                    img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
                cv2.imwrite('output_kept_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

                non_kept_masks = np.array([masks_before_overlap[i] for i in masks_filtered_dict['removed_indices']])
                img_array_non_kept = np.array(sample_img)
                for mask in non_kept_masks:
                    color = np.random.randint(0, 255, 3).tolist()
                    color_mask = np.zeros_like(img_array_non_kept)
                    color_mask[mask == True] = color
                    img_array_non_kept = cv2.addWeighted(img_array_non_kept, 1, color_mask, 0.9, 0)
                cv2.imwrite('output_removed_masks.jpg', cv2.cvtColor(img_array_non_kept, cv2.COLOR_RGB2BGR))

#        plotted_img = results[0].plot(
#            boxes=False,
#            masks=True,
#        )
        
 #       plotted_img_w_boxes = results[0].plot(
 #           boxes=True,
 #           masks=True,
 #       )

        # Draw masks with different colors
        img_array = np.array(sample_img)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):

                color = np.random.randint(0, 255, 3).tolist()
                color_mask = np.zeros_like(img_array)
                color_mask[mask == True] = color
                img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
        
        if save_imgs_bool:
            cv2.imwrite('output_colored_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
         #   cv2.imwrite('output.jpg', plotted_img)
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        # Save masks (boolean values) and also normalized xy polygon coordinates
        masks_list.append(results[0].masks.data.cpu().numpy())
        xyn_list.append(results[0].masks.xyn)
        # Save image IDs and images
        img_ids_list.append(sample_idx)
        sample_imgs.append(sample_img)
        # Save confidence scores
        conf_scores_list.append(results[0].boxes.conf.cpu().numpy())
    
    if times_image_encoding:
        def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

        avg_enc        = _avg(times_image_encoding)
        avg_inf        = _avg(times_inference)
        avg_red        = _avg(times_red)
        avg_bbox       = _avg(times_bbox)
        avg_mask_shape = _avg(times_mask_shape)
        avg_mask_size  = _avg(times_mask_size)
        avg_hi         = _avg(times_holes_islands)
        avg_depth      = _avg(times_depth)
        avg_ov         = _avg(times_overlap)

        print("\n--- Timing Summary (excluding first sample) ---")
        print(f"Avg image encoding time  : {avg_enc:.3f} s")
        print(f"Avg inference time       : {avg_inf:.3f} s")
        if times_red:
            print(f"Avg red filter           : {avg_red:.3f} s")
        if times_bbox:
            print(f"Avg bbox size filter     : {avg_bbox:.3f} s")
        if times_mask_shape:
            print(f"Avg mask shape filter    : {avg_mask_shape:.3f} s")
        if times_mask_size:
            print(f"Avg mask size filter     : {avg_mask_size:.3f} s")
        if times_holes_islands:
            print(f"Avg holes/islands filter : {avg_hi:.3f} s")
        if times_depth:
            print(f"Avg depth estimation     : {avg_depth:.3f} s")
        if times_overlap:
            print(f"Avg overlap filter       : {avg_ov:.3f} s")
        full_avg = avg_enc + avg_inf + avg_red + avg_bbox + avg_mask_shape + avg_mask_size + avg_hi + avg_depth + avg_ov
        print(f"Avg full time (sum)      : {full_avg:.3f} s")
        print("-----------------------------------------------")

    if store_masks_bool:
        filepath = '../../disk/saved_masks/SAM3' if prompt == "raspberry" else '../../disk/saved_masks/SAM3_punnet'
        store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list, sample_imgs, filepath=filepath)

def model_SAM(samples, mode = 'mobile', save_imgs_bool = False, store_masks_bool = False, testing_samples = 1, filter_bboxes = (True, 0.2, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None), filter_red = (True, 0.4)):
    """
    
    SAM segmentation model inference and mask saving.

    Args:

        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        mode (str): The mode of the SAM model ('mobile' or 'base') ; mobile is supposedly much faster.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process (i.e. processing takes a long time, so for testing we can limit this).
        filter_bboxes (tuple): (bool, lower_multiplier, upper_multiplier) to filter bounding boxes based on area relative to median.
        filter_masks_shapes (tuple): (bool, rectangularity_threshold) to filter masks based on shape.
        filter_masks_sizes (tuple): (bool, lower_multiplier, upper_multiplier) to filter masks based on size.
        filter_red (tuple): (bool, min_red_fraction) to keep only masks whose covered pixels are mostly red.

    """

    if type(samples) is not list:
        samples = [samples]

    # Load SAM model
    # We can set a higher conf score, possibly try finding some optimal one across multiple images...
    model = "../../disk/pretrained_models/mobile_sam.pt" if mode == 'mobile' else "../../disk/pretrained_models/sam_b.pt"
    overrides = dict(conf=0.70, 
                     task="segment", 
                     mode="predict", 
                     imgsz = 1024, # internal resize to (1024,1024) seems to give the best results, since SAM was trained on this size
                     model=model,
                     save = False) 
    predictor = SAMPredictor(overrides=overrides)


    masks_list = []
    img_ids_list = []
    conf_scores_list = []

    # Let us only look at the first k samples for testing
    if type(testing_samples) is int:
        samples = samples[:testing_samples]


    # Build point grid for prompt encoding
    points_grid = build_point_grid(32) # Create 32x32 grid of points (default setting for SAM if not otherwise prompted)
    # Bonnet bbox 
    bonnet_bbox = [260, 116, 1120, 707]  # x_min, y_min, x_max, y_max
    #bonnet_bbox = [0,0,1280,800]

    for sample in samples:

        sample_img = sample['image']
        sample_idx = sample['image_id']

      #  plt.imshow(sample_img)
      #  plt.axis('off')
      #  plt.show()

        H,W = sample_img.size[1], sample_img.size[0]

        points_grid_scaled = points_grid * np.array([W, H])
        x_min, y_min, x_max, y_max = bonnet_bbox

        mask_filter = (
            (points_grid_scaled[:, 0] >= x_min) & 
            (points_grid_scaled[:, 0] <= x_max) &
            (points_grid_scaled[:, 1] >= y_min) & 
            (points_grid_scaled[:, 1] <= y_max)
        )

        filtered_points = points_grid_scaled[mask_filter]


       # plt.imshow(sample_img)
       # plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1)
       # plt.axis('off')
       # plt.show()

        # Scale filtered_points to [0, 1] range as expected by SAM
        filtered_points = filtered_points / np.array([W, H])

        filtered_points = [np.array(filtered_points)]


        start_time_0 = time()
        predictor.set_image(sample_img) # IMAGE ENCODING
        end_time_0 = time()
        print(f"Image encoding time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        start_time = time()
        results = predictor() # points = filtered_points TODO : need to fix, smh doesnt work on the GPU and also sloppy, should detect the bonnet first # PROMPT ENCODING (automatic 32x32 grid of points I think ?) AND MASK DECODING (for each of these points, predict masks)
       # results = predictor()
        end_time = time()
        print(f"Inference (prompt encoding and mask decoding) time for sample {sample_idx}: {end_time - start_time:.2f} seconds")

        # 1. First ,filter solely based on redness (since raspberries will always be red or contain a certain amount of red pixels)
        if results[0].masks is not None and filter_red[0]:
            masks_np = results[0].masks.data.cpu().numpy()
            red_valid_idx = _filter_red_masks(masks_np, np.array(sample_img), min_red_fraction=filter_red[1])
            results[0].boxes = results[0].boxes[red_valid_idx]
            results[0].masks = results[0].masks[red_valid_idx]

        # 2. Bbox size filter
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0 and filter_bboxes[0]:
            valid_idx, median_area = _filter_bbox_sizes(
                boxes,
                upper_multiplier=filter_bboxes[2],
                lower_multiplier=filter_bboxes[1],
                save_distribution=save_imgs_bool,
                filename='area_distribution.png'
            )
            results[0].boxes = results[0].boxes[valid_idx]
            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]

        # 3. Mask shape filter (rectangularity)
        if results[0].masks is not None and filter_masks_shapes[0]:
            masks_np = results[0].masks.data.cpu().numpy()
            boxes_np = results[0].boxes.xyxy.cpu().numpy()
            shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
            results[0].boxes = results[0].boxes[shape_valid_idx]
            results[0].masks = results[0].masks[shape_valid_idx]

        # 4. Mask size filter
        if results[0].masks is not None and filter_masks_sizes[0]:
            masks_np = results[0].masks.data.cpu().numpy()
            size_valid_idx, median_mask_area = _filter_mask_sizes(
                masks_np,
                upper_multiplier=filter_masks_sizes[2],
                lower_multiplier=filter_masks_sizes[1],
                save_distribution=save_imgs_bool,
                filename='mask_size_distribution.png'
            )
            results[0].boxes = results[0].boxes[size_valid_idx]
            results[0].masks = results[0].masks[size_valid_idx]
        
        plotted_img = results[0].plot(
            boxes=False,
            masks=True,
        )
        
        plotted_img_w_boxes = results[0].plot(
            boxes=True,
            masks=True,
        )

        # Draw masks with different colors
        img_array = np.array(sample_img)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):

                color = np.random.randint(0, 255, 3).tolist()
                color_mask = np.zeros_like(img_array)
                color_mask[mask == True] = color
                img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
        
        if save_imgs_bool:
            cv2.imwrite('output_colored_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
           # cv2.imwrite('output.jpg', plotted_img)
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        # Save masks, confidence scores, and image IDs
        masks_list.append(results[0].masks.data.cpu().numpy())
        conf_scores_list.append(results[0].boxes.conf.cpu().numpy())
        img_ids_list.append(sample_idx)

    if store_masks_bool:
        if mode == 'mobile':
            store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list = None, sample_imgs = None, filepath= '../../disk/saved_masks/SAM_mobile')
        else:
            store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list = None, sample_imgs = None, filepath= '../../disk/saved_masks/SAM')

    return masks_list, conf_scores_list, img_ids_list


def model_SAM_extended(samples, mode='mobile', save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes=(True, 0.2, 3.0), filter_masks_shapes=(True, 0.85), filter_masks_sizes=(True, 0.2, None), filter_red=(True, 0.4), filter_holes_islands=False, filter_overlap_masks=False):
    """
    SAM segmentation model inference and mask saving — extended version of model_SAM.

    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        mode (str): The mode of the SAM model ('mobile' or 'base').
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process.
        filter_bboxes (tuple): (bool, lower_multiplier, upper_multiplier) to filter bounding boxes based on area relative to median.
        filter_masks_shapes (tuple): (bool, rectangularity_threshold) to filter masks based on shape.
        filter_masks_sizes (tuple): (bool, lower_multiplier, upper_multiplier) to filter masks based on size.
        filter_red (tuple): (bool, min_red_fraction) to keep only masks whose covered pixels are mostly red.
        filter_holes_islands (bool): Whether to remove small holes and islands from each mask via morphological cleanup.
        filter_overlap_masks (bool): Whether to remove highly overlapping/contained masks using depth-guided selection.
    """

    if type(samples) is not list:
        samples = [samples]

    # Load SAM model
    model = "../../disk/pretrained_models/mobile_sam.pt" if mode == 'mobile' else "../../disk/pretrained_models/sam_b.pt"
    overrides = dict(conf=0.70,
                     task="segment",
                     mode="predict",
                     imgsz=1024,
                     model=model,
                     save=False)
    predictor = SAMPredictor(overrides=overrides)

    # Load depth pipeline only when needed
    depth_pipe = None
    if filter_overlap_masks:
        depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=2 if torch.cuda.is_available() else 'cpu')

    # Load punnet detection model for dynamic bbox extraction
    punnet_model = YOLO("../../disk/pretrained_models/yolo26n-seg-punnet-best.pt")
    _FALLBACK_BONNET_BBOX = [260, 116, 1120, 707]  # x_min, y_min, x_max, y_max

    masks_list = []
    img_ids_list = []
    conf_scores_list = []

    if type(testing_samples) is int:
        samples = samples[:testing_samples]

    # Build point grid for prompt encoding
    points_grid = build_point_grid(32)

    times_image_encoding = []
    times_inference = []
    times_red = []
    times_bbox = []
    times_mask_shape = []
    times_mask_size = []
    times_punnet_detection = []
    times_holes_islands = []
    times_depth = []
    times_overlap = []

    for loop_idx, sample in enumerate(samples):

        sample_img = sample['image']
        sample_idx = sample['image_id']

        H, W = sample_img.size[1], sample_img.size[0]

        # Detect punnet bbox for this sample; fall back to fixed bbox if no detection
        start_time_punnet = time()
        punnet_results = punnet_model.predict(sample_img, device=2, verbose=False)
        end_time_punnet = time()
        if loop_idx > 0:
            times_punnet_detection.append(end_time_punnet - start_time_punnet)
        if punnet_results[0].boxes is not None and len(punnet_results[0].boxes) > 0:
            # If multiple punnet detections exist, calc each box center, measure euclid dist to image center and choose box with smallest dist
            boxes_xyxy = punnet_results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
            img_cx, img_cy = W / 2, H / 2
            box_centers = np.stack([(boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2,
                                    (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2], axis=1)
            dists = np.hypot(box_centers[:, 0] - img_cx, box_centers[:, 1] - img_cy)
            bonnet_bbox = boxes_xyxy[np.argmin(dists)].tolist()
        else:
            bonnet_bbox = _FALLBACK_BONNET_BBOX

        points_grid_scaled = points_grid * np.array([W, H])
        x_min, y_min, x_max, y_max = bonnet_bbox

        mask_filter = (
            (points_grid_scaled[:, 0] >= x_min) &
            (points_grid_scaled[:, 0] <= x_max) &
            (points_grid_scaled[:, 1] >= y_min) &
            (points_grid_scaled[:, 1] <= y_max)
        )

        filtered_points = points_grid_scaled[mask_filter]
        filtered_points = filtered_points / np.array([W, H])
        
        # Convert into list of lists
       # filtered_points = filtered_points.tolist()

        #print(filtered_points)
    
        filtered_points = [np.array(filtered_points)]
        

        # Save figure of the prompt points for visualisation/debugging
        if save_imgs_bool:
            plt.imshow(sample_img)
            plt.scatter(filtered_points[0][:, 0] * W, filtered_points[0][:, 1] * H, s=1)
            plt.axis('off')
            plt.title(f"Prompt points for sample {sample_idx}")
            plt.savefig(f'prompt_points_{sample_idx}.png')
            plt.close()
        
       # exit()

        start_time_0 = time()
        predictor.set_image(sample_img)
        end_time_0 = time()
        print(f"Image encoding time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        if loop_idx > 0:
            times_image_encoding.append(end_time_0 - start_time_0)
        start_time = time()
        results = predictor(points = filtered_points) # 
        end_time = time()
        print(f"Inference time for sample {sample_idx}: {end_time - start_time:.2f} seconds")
        if loop_idx > 0:
            times_inference.append(end_time - start_time)

        # 1. Red filter — discard non-raspberry masks before any other filtering
        if results[0].masks is not None and filter_red[0]:
            _t0 = time()
            masks_np = results[0].masks.data.cpu().numpy()
            red_valid_idx = _filter_red_masks(masks_np, np.array(sample_img), min_red_fraction=filter_red[1])
            results[0].boxes = results[0].boxes[red_valid_idx]
            results[0].masks = results[0].masks[red_valid_idx]
            _t1 = time()
            print(f"Red filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_red.append(_t1 - _t0)

        # 2. Bbox size filter
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) > 0 and filter_bboxes[0]:
            _t0 = time()
            valid_idx, median_area = _filter_bbox_sizes(
                boxes,
                upper_multiplier=filter_bboxes[2],
                lower_multiplier=filter_bboxes[1],
                save_distribution=save_imgs_bool,
                filename='area_distribution.png'
            )
            results[0].boxes = results[0].boxes[valid_idx]
            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]
            _t1 = time()
            print(f"Bbox size filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_bbox.append(_t1 - _t0)

        # 3. Mask shape filter (rectangularity)
        if results[0].masks is not None and filter_masks_shapes[0]:
            _t0 = time()
            masks_np = results[0].masks.data.cpu().numpy()
            boxes_np = results[0].boxes.xyxy.cpu().numpy()
            shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
            results[0].boxes = results[0].boxes[shape_valid_idx]
            results[0].masks = results[0].masks[shape_valid_idx]
            _t1 = time()
            print(f"Mask shape filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_mask_shape.append(_t1 - _t0)

        # 4. Mask size filter
        if results[0].masks is not None and filter_masks_sizes[0]:
            _t0 = time()
            masks_np = results[0].masks.data.cpu().numpy()
            size_valid_idx, median_mask_area = _filter_mask_sizes(
                masks_np,
                upper_multiplier=filter_masks_sizes[2],
                lower_multiplier=filter_masks_sizes[1],
                save_distribution=save_imgs_bool,
                filename='mask_size_distribution.png'
            )
            results[0].boxes = results[0].boxes[size_valid_idx]
            results[0].masks = results[0].masks[size_valid_idx]
            _t1 = time()
            print(f"Mask size filter time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_mask_size.append(_t1 - _t0)

        # 5. Holes-and-islands cleanup — refine each mask's shape morphologically
        if filter_holes_islands and results[0].masks is not None:
            start_time_0 = time()
            masks = results[0].masks.data.cpu().numpy()
            # Get the area of the smallest mask and set islands threshold to just below that
            islands_threshold = int(min(m.astype(np.bool_).sum() for m in masks)) - 1
            refined_masks = []
            for mask in masks:
                mask = mask.astype(np.bool_)
                refined_mask, _ = remove_small_regions(mask, islands_threshold, mode='islands')
                refined_mask, _ = remove_small_regions(refined_mask, 20000, mode='holes')
                refined_masks.append(refined_mask)
            refined_masks = np.array(refined_masks)
            results[0].masks = torch.from_numpy(refined_masks)
            end_time_0 = time()
            print(f"Holes/islands filter time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_holes_islands.append(end_time_0 - start_time_0)

        # 6. Overlap filter — keep one mask when two masks overlap/contain each other, guided by depth
        if filter_overlap_masks and results[0].masks is not None:
            _t0 = time()
            depth_array = np.array(depth_pipe(sample_img)["depth"])
            _t1 = time()
            print(f"Depth estimation time for sample {sample_idx}: {_t1 - _t0:.2f} seconds")
            if loop_idx > 0:
                times_depth.append(_t1 - _t0)
            start_time_0 = time()
            masks = results[0].masks.data.cpu().numpy()
            # snapshot before filtering, used for removed-mask visualisation below
            masks_before_overlap = masks.copy()
            masks_depth_values = np.array([depth_array * mask for mask in masks])
            masks_filtered_dict = filter_overlapping_masks_extended(
                masks,
                masks_depth_values,
                overlap_threshold=50,
                containment_threshold=0.95,
                depth_difference_threshold=40,
                debug=False
            )
            results[0].masks = results[0].masks[masks_filtered_dict['kept_indices']]
            results[0].boxes = results[0].boxes[masks_filtered_dict['kept_indices']]
            end_time_0 = time()
            print(f"Overlap filter time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_overlap.append(end_time_0 - start_time_0)

            if save_imgs_bool:
                img_array = np.array(sample_img)
                for mask in masks_filtered_dict['masks']:
                    color = np.random.randint(0, 255, 3).tolist()
                    color_mask = np.zeros_like(img_array)
                    color_mask[mask == True] = color
                    img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
                cv2.imwrite('output_kept_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

                non_kept_masks = np.array([masks_before_overlap[i] for i in masks_filtered_dict['removed_indices']])
                img_array_non_kept = np.array(sample_img)
                for mask in non_kept_masks:
                    color = np.random.randint(0, 255, 3).tolist()
                    color_mask = np.zeros_like(img_array_non_kept)
                    color_mask[mask == True] = color
                    img_array_non_kept = cv2.addWeighted(img_array_non_kept, 1, color_mask, 0.9, 0)
                cv2.imwrite('output_removed_masks.jpg', cv2.cvtColor(img_array_non_kept, cv2.COLOR_RGB2BGR))

      #  plotted_img = results[0].plot(boxes=False, masks=True)
#        plotted_img_w_boxes = results[0].plot(boxes=True, masks=True)

        # Draw masks with different colors
        img_array = np.array(sample_img)
        if results[0].masks is not None:
            for mask in results[0].masks.data.cpu().numpy():
                color = np.random.randint(0, 255, 3).tolist()
                color_mask = np.zeros_like(img_array)
                color_mask[mask == True] = color
                img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)

        if save_imgs_bool:
            cv2.imwrite('output_colored_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        masks_list.append(results[0].masks.data.cpu().numpy())
        conf_scores_list.append(results[0].boxes.conf.cpu().numpy())
        img_ids_list.append(sample_idx)

    if times_image_encoding:
        def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

        avg_enc        = _avg(times_image_encoding)
        avg_inf        = _avg(times_inference)
        avg_red        = _avg(times_red)
        avg_bbox       = _avg(times_bbox)
        avg_mask_shape = _avg(times_mask_shape)
        avg_mask_size  = _avg(times_mask_size)
        avg_hi         = _avg(times_holes_islands)
        avg_depth      = _avg(times_depth)
        avg_ov         = _avg(times_overlap)
        avg_punnet     = _avg(times_punnet_detection)

        print("\n--- Timing Summary (excluding first sample) ---")
        print(f"Avg image encoding time  : {avg_enc:.3f} s")
        print(f"Avg inference time       : {avg_inf:.3f} s")
        if times_red:
            print(f"Avg red filter           : {avg_red:.3f} s")
        if times_bbox:
            print(f"Avg bbox size filter     : {avg_bbox:.3f} s")
        if times_mask_shape:
            print(f"Avg mask shape filter    : {avg_mask_shape:.3f} s")
        if times_mask_size:
            print(f"Avg mask size filter     : {avg_mask_size:.3f} s")
        if times_holes_islands:
            print(f"Avg holes/islands filter : {avg_hi:.3f} s")
        if times_depth:
            print(f"Avg depth estimation     : {avg_depth:.3f} s")
        if times_overlap:
            print(f"Avg overlap filter       : {avg_ov:.3f} s")
        if times_punnet_detection:
            print(f"Avg punnet detection     : {avg_punnet:.3f} s")
        full_avg = avg_enc + avg_inf + avg_red + avg_bbox + avg_mask_shape + avg_mask_size + avg_hi + avg_depth + avg_ov + avg_punnet
        print(f"Avg full time (sum)      : {full_avg:.3f} s")
        print("-----------------------------------------------")

    if store_masks_bool:
        if mode == 'mobile':
            store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list=None, sample_imgs=None, filepath='../../disk/saved_masks/SAM_extended_mobile')
        else:
            store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list=None, sample_imgs=None, filepath='../../disk/saved_masks/SAM_extended')

    return masks_list, conf_scores_list, img_ids_list


def model_SAM_manipulate(samples, save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes=(True, 0.2, 3.0), filter_masks_shapes=(True, 0.85), filter_masks_sizes=(True, 0.2, None)):
    """
    SAM segmentation model using Meta's implementation (not Ultralytics).
    Provides more flexibility in hyperparameters.
    
    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process.
        filter_bboxes (tuple): (bool, lower_multiplier, upper_multiplier) to filter bounding boxes based on area relative to median.
        filter_masks_shapes (tuple): (bool, rectangularity_threshold) to filter masks based on shape.
        filter_masks_sizes (tuple): (bool, lower_multiplier, upper_multiplier) to filter masks based on size.
    """
    
    if type(samples) is not list:
        samples = [samples]
    
    # Load SAM model (Meta implementation)
    sam = sam_model_registry["vit_b"](checkpoint="pretrained_models/sam_b.pt")
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,
    )
    
    masks_list = []
    img_ids_list = []
    
    # Let us only look at the first k samples for testing
    samples = samples[:testing_samples]
    
    for sample in samples:
        sample_img = sample['image']
        sample_idx = sample['image_id']

        sample_img = np.asarray(sample_img)[110:700, 227:1105]
        
        # Generate masks (Meta SAM returns list of dicts)
        begin_time = time()
        mask_dicts = mask_generator.generate(np.array(sample_img))
        end_time = time()
        print(f"Inference time for sample {sample_idx}: {end_time - begin_time:.2f} seconds")

        # Extract masks and bounding boxes for filtering
        if len(mask_dicts) > 0:
            # Extract bounding boxes and calculate areas
            bboxes = np.array([m['bbox'] for m in mask_dicts])  # [x, y, w, h]

            # Convert to xyxy format for filtering
            boxes_xyxy = np.column_stack([
                bboxes[:, 0],
                bboxes[:, 1],
                bboxes[:, 0] + bboxes[:, 2],
                bboxes[:, 1] + bboxes[:, 3]
            ])
            
            # Filter by bounding box size
            if filter_bboxes[0]:
                valid_idx, median_area = _filter_bbox_sizes(
                    boxes_xyxy,
                    upper_multiplier=filter_bboxes[2],
                    lower_multiplier=filter_bboxes[1],
                    save_distribution=save_imgs_bool,
                    filename='area_distribution.png'
                )
                mask_dicts = [m for i, m in enumerate(mask_dicts) if valid_idx[i]]
            
            # Filter by mask shape (rectangularity)
            if filter_masks_shapes[0] and len(mask_dicts) > 0:
                # Extract masks and boxes from remaining mask_dicts
                binary_masks = np.array([m['segmentation'] for m in mask_dicts])
                bboxes = np.array([m['bbox'] for m in mask_dicts])
                boxes_xyxy = np.column_stack([
                    bboxes[:, 0],
                    bboxes[:, 1],
                    bboxes[:, 0] + bboxes[:, 2],
                    bboxes[:, 1] + bboxes[:, 3]
                ])
                
                shape_valid_idx = _filter_mask_shapes(
                    binary_masks, 
                    boxes=boxes_xyxy, 
                    rectangularity_threshold=filter_masks_shapes[1]
                )
                mask_dicts = [m for i, m in enumerate(mask_dicts) if shape_valid_idx[i]]
            
            # Filter by mask size
            if filter_masks_sizes[0] and len(mask_dicts) > 0:
                binary_masks = np.array([m['segmentation'] for m in mask_dicts])
                size_valid_idx, median_mask_area = _filter_mask_sizes(
                    binary_masks,
                    upper_multiplier=filter_masks_sizes[2],
                    lower_multiplier=filter_masks_sizes[1],
                    save_distribution=save_imgs_bool,
                    filename='mask_size_distribution.png'
                )
                mask_dicts = [m for i, m in enumerate(mask_dicts) if size_valid_idx[i]]
        
        # Extract binary masks from the filtered dictionaries
        binary_masks = np.array([m['segmentation'] for m in mask_dicts]) if len(mask_dicts) > 0 else np.array([])
        
        # Visualization with colored masks (similar to Ultralytics version)
        img_array = np.array(sample_img).copy()
        
        for mask_dict in mask_dicts:
            mask = mask_dict['segmentation']
            color = np.random.randint(0, 255, 3).tolist()
            color_mask = np.zeros_like(img_array)
            color_mask[mask == True] = color
            img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
        
        if save_imgs_bool:
            # Save colored masks version
            cv2.imwrite(f'output_colored_masks.jpg', 
                       cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            
            # Save version with bounding boxes
            img_with_boxes = img_array.copy()
            for mask_dict in mask_dicts:
                bbox = mask_dict['bbox']  # [x, y, w, h]
                x, y, w, h = bbox
                color = np.random.randint(0, 255, 3).tolist()
                cv2.rectangle(img_with_boxes, (int(x), int(y)), 
                            (int(x+w), int(y+h)), color, 2)
            
            cv2.imwrite(f'output_w_boxes.jpg', 
                       cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
            
            # Save original image
            cv2.imwrite(f'sample_img.jpg', 
                       cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))
        
        # Store masks and IDs
        masks_list.append(binary_masks)
        img_ids_list.append(sample_idx)
    
    if store_masks_bool:
        store_masks(masks_list, img_ids_list, filepath='saved_masks/SAM_manipulate')
    
def model_grounding_SAM(samples, mode = 'base', save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes=(True, 0.2, 3.0), filter_masks_shapes=(True, 0.85), filter_masks_sizes=(True, 0.2, None), filter_holes_islands = False, filter_overlap_masks = False, filter_by_darkness = False):
    """
    Grounding DINO to find bboxes based on text prompt + SAM to generate masks.
    """

    """
    Grounding DINO + SAM segmentation model.

    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process.
        filter_bboxes (tuple): (bool, lower_multiplier, upper_multiplier) to filter bounding boxes based on area relative to median.
        filter_masks_shapes (tuple): (bool, rectangularity_threshold) to filter masks based on shape.
        filter_masks_sizes (tuple): (bool, lower_multiplier, upper_multiplier) to filter masks based on size.
        filter_by_darkness (bool): Whether to filter masks based on darkness using LAB color space.
    """


    if type(samples) is not list:
        samples = [samples]

   # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # BBOX DETECTOR : Grounding DINO
    detector_id = "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=2 if torch.cuda.is_available() else 'cpu')
    labels = ["raspberry."]
    labels = [label if label.endswith(".") else label+"." for label in labels]

    # MASK GENERATOR : SAM
    segmenter_id = "../../disk/pretrained_models/sam_b.pt" if mode == 'base' else "../../disk/pretrained_models/mobile_sam.pt"
    overrides = dict(conf=0.70, 
                     task="segment", 
                     mode="predict", 
                     imgsz = 1024, # internal resize to (1024,1024) seems to give the best results, since SAM was trained on this size
                     model=segmenter_id,
                     save = False,
                     device = 2 if torch.cuda.is_available() else 'cpu')
    mask_predictor = SAMPredictor(overrides=overrides)


    # DEPTH DETECTOR
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device= 2 if torch.cuda.is_available() else 'cpu')


  #  model_configs = {
  #      'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
  #      'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
  #      'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
  #      'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
  #  }

  #  encoder = 'vitb' # or 'vits', 'vitb', 'vitg'

  #  model = DepthAnythingV2(**model_configs[encoder])
  #  model.load_state_dict(torch.load(f'Depth_module/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
  #  model = model.to(device)
  #  model = model.float()
  #  model.eval()



    masks_list = []
    img_ids_list = []
    conf_scores_list = []

    # Let us only look at the first k samples for testing
    if type(testing_samples) is int:
        samples = samples[:testing_samples]
      #  samples = [samples[3]]

    times_depth = []
    times_dino = []
    times_image_encoding = []
    times_inference = []
    times_bbox = []
    times_mask_shape = []
    times_mask_size = []
    times_holes_islands = []
    times_overlap = []

    for loop_idx, sample in enumerate(samples):

        sample_img = sample['image']
        sample_idx = sample['image_id']


        start_time_0 = time()
        depth = pipe(sample_img)["depth"]
        end_time_0 = time()
        print(f"Depth estimation time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        if loop_idx > 0:
            times_depth.append(end_time_0 - start_time_0)

      #  raw_img = np.array(sample_img)[:, :, ::-1]  # RGB to BGR
        # Convert to float32
       # raw_img = raw_img.astype(np.float32)
       # depth = model.infer_image(raw_img) # HxW raw depth map in numpy

        # Visualize depth (PIL image)
        depth_array = np.array(depth)


        # DINO
        start_time_0 = time()
        detections = object_detector(sample_img, candidate_labels = labels, threshold = 0.10)
        end_time_0 = time()
        print(f"Detection time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        if loop_idx > 0:
            times_dino.append(end_time_0 - start_time_0)
        
        detections = [DetectionResult.from_dict(det) for det in detections]
        

        bboxes_detections = get_boxes(detections)
     

        # Remove one layer of the inner list 
        bboxes_detections = np.array(bboxes_detections[0])

        # NOTE : idea was that when we segment based on point prompts, we might get better results than with bbox prompts, but in practice it seems worse...
        # Turn bbox detections into point prompts by getting center points of each bbox
        # bboxes format is [x_min, y_min, x_max, y_max]
       # center_points = []
       # for bbox in bboxes_detections:
       #     x_min, y_min, x_max, y_max = bbox
       #     center_x = (x_min + x_max) / 2
       #     center_y = (y_min + y_max) / 2
       #     center_points.append([center_x, center_y])
       # center_points_to_draw = np.array(center_points).copy()
        # Scale center points to [0, 1] range as expected by SAM
       # H,W = sample_img.size[1], sample_img.size[0]
       # center_points = center_points / np.array([W, H])
       # center_points = [np.array(center_points)]
        # Visualize center points on image
       # plt.imshow(sample_img)
       # plt.scatter(center_points_to_draw[:, 0], center_points_to_draw[:, 1], s=10, c='red')
       # plt.axis('off')
       # plt.show()

        print("Total detections from DINO:", len(bboxes_detections))

        # SAM
        start_time_1 = time()
        mask_predictor.set_image(sample_img) # IMAGE ENCODING
        end_time_1 = time()
        print(f"Image encoding time for sample {sample_idx}: {end_time_1 - start_time_1:.2f} seconds")
        if loop_idx > 0:
            times_image_encoding.append(end_time_1 - start_time_1)
        start_time_2 = time()
        results = mask_predictor(bboxes = bboxes_detections) # PROMPT ENCODING AND MASK DECODING
        end_time_2 = time()
        print(f"Inference (prompt encoding and mask decoding) time for sample {sample_idx}: {end_time_2 - start_time_2:.2f} seconds")
        if loop_idx > 0:
            times_inference.append(end_time_2 - start_time_2)

        print(f"Full inference time DINO + SAM: {end_time_2 - start_time_2 + end_time_1 - start_time_1 + end_time_0 - start_time_0} seconds")


        

        # Calculate median area and filter outliers
        boxes = results[0].boxes.xyxy.cpu().numpy()

     #   mask_depth_values_list = []
        # Extract depth values for each mask 
      #  for i, mask in enumerate(results[0].masks.data.cpu().numpy()):
       #     mask_depth_values = depth_array[mask]
        #    mask_depth_values_list.append(mask_depth_values)

        #mask_depth_values = np.array(mask_depth_values_list)
        print("Total masks from SAM before filtering:", len(boxes))
        
        if len(boxes) > 0 and filter_bboxes[0]:
            start_time_0 = time()
            valid_idx, median_area = _filter_bbox_sizes(
                boxes, 
                upper_multiplier=filter_bboxes[2],
                lower_multiplier=filter_bboxes[1],
                save_distribution=save_imgs_bool,
                filename='area_distribution.png'
            )
            end_time_0 = time()
            print(f"Filtering bounding boxes time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_bbox.append(end_time_0 - start_time_0)

            results[0].boxes = results[0].boxes[valid_idx]
          #  mask_depth_values = mask_depth_values[valid_idx]
            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]

                if filter_masks_shapes[0]:
                    start_time_0 = time()
                    # Filter by mask shape (rectangularity)
                    masks_np = results[0].masks.data.cpu().numpy()
                    boxes_np = results[0].boxes.xyxy.cpu().numpy()
                    shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
                    results[0].boxes = results[0].boxes[shape_valid_idx]
                    results[0].masks = results[0].masks[shape_valid_idx]
                  #  mask_depth_values = mask_depth_values[shape_valid_idx]


                    end_time_0 = time()
                    print(f"Filtering masks by shape time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
                    if loop_idx > 0:
                        times_mask_shape.append(end_time_0 - start_time_0)

                if filter_masks_sizes[0]:
                    start_time_0 = time()
                    # Filter by actual mask size
                    masks_np = results[0].masks.data.cpu().numpy()
                    size_valid_idx, median_mask_area = _filter_mask_sizes(
                        masks_np,
                        upper_multiplier= filter_masks_sizes[2], # No upper limit, as we have filtered out via large bboxes already
                        lower_multiplier= filter_masks_sizes[1],  # Filter small masks
                        save_distribution=save_imgs_bool,
                        filename='mask_size_distribution.png'
                    )
                    results[0].boxes = results[0].boxes[size_valid_idx]
                    results[0].masks = results[0].masks[size_valid_idx]
                 #   mask_depth_values = mask_depth_values[size_valid_idx]
                    end_time_0 = time()
                    print(f"Filtering masks by size time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
                    if loop_idx > 0:
                        times_mask_size.append(end_time_0 - start_time_0)
        
      #  plotted_img = results[0].plot(
      #      boxes=False,
      #      masks=True,
      #  )
        
      #  plotted_img_w_boxes = results[0].plot(
      #      boxes=True,
      #      masks=True,
      #  )

        '''
        if filter_by_darkness:
            start_time_0 = time()
            
            # Convert PIL image to numpy array and then to LAB
            img_array = np.array(sample_img)
            img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            masks_np = results[0].masks.data.cpu().numpy()
            darkness_valid_idx = _filter_masks_by_darkness(
                masks_np,
                img_array,
                img_lab,
                plot_boxplot= True,
                plot_removed = True
            )
            
            results[0].boxes = results[0].boxes[darkness_valid_idx]
            results[0].masks = results[0].masks[darkness_valid_idx]
            mask_depth_values = mask_depth_values[darkness_valid_idx]
            
            end_time_0 = time()
            print(f"Filtering masks by darkness time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            print(f"Masks remaining after darkness filtering: {len(darkness_valid_idx)}")
        '''
        

        if filter_holes_islands:
            start_time_0 = time()
            # Finally, iterate over all masks and use remove_small_regions
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                # Get the area of the smallest mask and set islands threshold to just below that
                islands_threshold = int(min(m.astype(np.bool_).sum() for m in masks)) - 1
                refined_masks = []
                for mask in masks:
                    mask = mask.astype(np.bool_)
                    # Plot the mask
                #  plt.figure(figsize=(5, 5))
                #  plt.imshow(mask, cmap='gray')
                #  plt.axis('off')
                #  plt.show()
                    # Erode the mask first
                  #  kernel = np.ones((1, 1), np.uint8)
                  #  mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                  #  mask = mask.astype(np.bool_)
                    refined_mask, _ = remove_small_regions(mask, islands_threshold, mode = 'islands')
                    refined_mask, _ = remove_small_regions(refined_mask, 20000, mode = 'holes')
                    # Plot the refined mask
                #  plt.figure(figsize=(5, 5))
                #  plt.imshow(refined_mask, cmap='gray')
                #  plt.axis('off')
                #  plt.show()
                    refined_masks.append(refined_mask)
                refined_masks = np.array(refined_masks)

                results[0].masks = torch.from_numpy(refined_masks)

            
            end_time_0 = time()
            print(f"Filtering holes and islands time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_holes_islands.append(end_time_0 - start_time_0)

        masks_depth_values_list = []
        # Extract depth values for each mask 
        for i, mask in enumerate(results[0].masks.data.cpu().numpy()):
            mask_depth_values = depth_array * mask
            masks_depth_values_list.append(mask_depth_values)   
        masks_depth_values = np.array(masks_depth_values_list)

        if filter_overlap_masks:
            start_time_0 = time()
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
            
            masks_filtered_dict = filter_overlapping_masks_extended(
                masks,
                masks_depth_values,
                overlap_threshold=50, # 50
                containment_threshold=0.95, # 0.95
                depth_difference_threshold = 40,
                debug = False
             #   enclosure_threshold= 0.2,
             #   area_fill_threshold= 0.8
                
            )
            results[0].masks = results[0].masks[masks_filtered_dict['kept_indices']]
            results[0].boxes = results[0].boxes[masks_filtered_dict['kept_indices']]
            end_time_0 = time()
            print(f"Filtering overlapping masks time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
            if loop_idx > 0:
                times_overlap.append(end_time_0 - start_time_0)


        

            if save_imgs_bool:
                # visualize_overlap_map(masks_filtered_dict['overlap_map'])
                # Visualize kept masks after overlap filtering
                img_array = np.array(sample_img)
                for i, mask in enumerate(masks_filtered_dict['masks']):
                    color = np.random.randint(0, 255, 3).tolist()
                    color_mask = np.zeros_like(img_array)
                    color_mask[mask == True] = color
                    img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)

                # Visualize image 
                cv2.imwrite('output_kept_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

                # Get non-kept masks
                non_kept_masks = []
                for i in masks_filtered_dict['removed_indices']:
                    non_kept_masks.append(refined_masks[i])

                non_kept_masks = np.array(non_kept_masks)

                # Visualize non-kept masks
                img_array_non_kept = np.array(sample_img)
                for i, mask in enumerate(non_kept_masks):
                    color = np.random.randint(0, 255, 3).tolist()
                    color_mask = np.zeros_like(img_array_non_kept)
                    color_mask[mask == True] = color
                    img_array_non_kept = cv2.addWeighted(img_array_non_kept, 1, color_mask, 0.9, 0)

                # Visualize image 
                cv2.imwrite('output_removed_masks.jpg', cv2.cvtColor(img_array_non_kept, cv2.COLOR_RGB2BGR))



        # Draw masks with different colors
        img_array = np.array(sample_img)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):

                color = np.random.randint(0, 255, 3).tolist()
                color_mask = np.zeros_like(img_array)
                color_mask[mask == True] = color
                img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
        
        if save_imgs_bool:
            plotted_img_w_boxes = results[0].plot(boxes=True, masks=True)

   
            # Write depth visualization
            cv2.imwrite('output_depth.jpg', cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=255.0/depth_array.max()), cv2.COLORMAP_PLASMA))
            cv2.imwrite('output_colored_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
           # cv2.imwrite('output.jpg', plotted_img)
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        # Save masks
        masks_list.append(results[0].masks.data.cpu().numpy())
        # Save image IDs
        img_ids_list.append(sample_idx)
        # Save confidence scores 
        conf_scores_list.append(results[0].boxes.conf.cpu().numpy())
    
    if times_image_encoding:
        def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

        avg_depth      = _avg(times_depth)
        avg_dino       = _avg(times_dino)
        avg_enc        = _avg(times_image_encoding)
        avg_inf        = _avg(times_inference)
        avg_bbox       = _avg(times_bbox)
        avg_mask_shape = _avg(times_mask_shape)
        avg_mask_size  = _avg(times_mask_size)
        avg_hi         = _avg(times_holes_islands)
        avg_ov         = _avg(times_overlap)

        print("\n--- Timing Summary (excluding first sample) ---")
        print(f"Avg depth estimation     : {avg_depth:.3f} s")
        print(f"Avg DINO detection       : {avg_dino:.3f} s")
        print(f"Avg SAM image encoding   : {avg_enc:.3f} s")
        print(f"Avg SAM inference        : {avg_inf:.3f} s")
        if times_bbox:
            print(f"Avg bbox size filter     : {avg_bbox:.3f} s")
        if times_mask_shape:
            print(f"Avg mask shape filter    : {avg_mask_shape:.3f} s")
        if times_mask_size:
            print(f"Avg mask size filter     : {avg_mask_size:.3f} s")
        if times_holes_islands:
            print(f"Avg holes/islands filter : {avg_hi:.3f} s")
        if times_overlap:
            print(f"Avg overlap filter       : {avg_ov:.3f} s")
        full_avg = avg_depth + avg_dino + avg_enc + avg_inf + avg_bbox + avg_mask_shape + avg_mask_size + avg_hi + avg_ov
        print(f"Avg full time (sum)      : {full_avg:.3f} s")
        print("-----------------------------------------------")

    if store_masks_bool:
        if mode == 'mobile':
            store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list = None, sample_imgs = None, filepath= '../../disk/saved_masks/DINO_SAM_mobile')
        else:
            store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list = None, sample_imgs = None, filepath= '../../disk/saved_masks/DINO_SAM')

def model_FastSAM(samples, save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes=(True, 0.2, 3.0), filter_masks_shapes=(True, 0.85), filter_masks_sizes=(True, 0.2, None)):
    """
    FastSAM segmentation model. This would be a model that could be good for the inference task

    Notes : Segmentation seems to be much worse, so not a good model choice...
    
    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process.
        filter_bboxes (tuple): (bool, lower_multiplier, upper_multiplier) to filter bounding boxes based on area relative to median.
        filter_masks_shapes (tuple): (bool, rectangularity_threshold) to filter masks based on shape.
        filter_masks_sizes (tuple): (bool, lower_multiplier, upper_multiplier) to filter masks based on size.
    """


    if type(samples) is not list:
        samples = [samples]

    # Load SAM model
    # We can set a higher conf score, possibly try finding some optimal one across multiple images...
    overrides = dict(conf=0.60, 
                     task="segment", 
                     mode="predict", 
                     imgsz= 1024,
                     model="pretrained_models/FastSAM-s.pt",
                     save = False) 
    #predictor = SAMPredictor(overrides=overrides)
    predictor = FastSAMPredictor(overrides=overrides)

    masks_list = []
    img_ids_list = []

    # Let us only look at the first k samples for testing
    samples = samples[:testing_samples]

    for sample in samples:

        sample_img = sample['image']
        sample_idx = sample['image_id']
        # Resize image to (1024,1024) for FastSAM
        sample_img = cv2.resize(np.asarray(sample_img), (1024, 1024))
        start_time = time()
        results = predictor(sample_img)
     #   results = predictor.prompt(results, texts = 'raspberry')
        end_time = time()
        print(f"Inference time for sample {sample_idx}: {end_time - start_time:.2f} seconds")

        # Calculate median area and filter outliers
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0 and filter_bboxes[0]:
            valid_idx, median_area = _filter_bbox_sizes(
                boxes, 
                upper_multiplier=filter_bboxes[2],
                lower_multiplier=filter_bboxes[1],
                save_distribution=save_imgs_bool,
                filename='area_distribution.png'
            )
            
            results[0].boxes = results[0].boxes[valid_idx]
            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]

                if filter_masks_shapes[0]:
                    # Filter by mask shape (rectangularity)
                    masks_np = results[0].masks.data.cpu().numpy()
                    boxes_np = results[0].boxes.xyxy.cpu().numpy()
                    shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
                    results[0].boxes = results[0].boxes[shape_valid_idx]
                    results[0].masks = results[0].masks[shape_valid_idx]



                if filter_masks_sizes[0]:
                    # Filter by actual mask size
                    masks_np = results[0].masks.data.cpu().numpy()
                    size_valid_idx, median_mask_area = _filter_mask_sizes(
                        masks_np,
                        upper_multiplier= filter_masks_sizes[2], # No upper limit, as we have filtered out via large bboxes already
                        lower_multiplier= filter_masks_sizes[1],  # Filter small masks
                        save_distribution=save_imgs_bool,
                        filename='mask_size_distribution.png'
                    )
                    results[0].boxes = results[0].boxes[size_valid_idx]
                    results[0].masks = results[0].masks[size_valid_idx]
        
        plotted_img = results[0].plot(
            boxes=False,
            masks=True,
        )
        
        plotted_img_w_boxes = results[0].plot(
            boxes=True,
            masks=True,
        )

        # Draw masks with different colors
        img_array = np.array(sample_img)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):

                color = np.random.randint(0, 255, 3).tolist()
                color_mask = np.zeros_like(img_array)
                color_mask[mask == True] = color
                img_array = cv2.addWeighted(img_array, 1, color_mask, 0.9, 0)
        
        if save_imgs_bool:
            cv2.imwrite('output_colored_masks.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
           # cv2.imwrite('output.jpg', plotted_img)
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        # Save masks
        masks_list.append(results[0].masks.data.cpu().numpy())
        # Save image IDs
        img_ids_list.append(sample_idx)
    
    if store_masks_bool:
        store_masks(masks_list, img_ids_list, filepath= 'saved_masks/FastSAM') 



def _filter_bbox_sizes(boxes, upper_multiplier=4.0, lower_multiplier=None, save_distribution=False, filename='area_distribution.png'):
    """
    Filter bounding boxes based on area relative to median.
    
    Args:
        boxes: numpy array of bounding boxes in xyxy format [x1, y1, x2, y2]
        upper_multiplier: Maximum area as multiple of median (default 4.0)
        lower_multiplier: Minimum area as multiple of median (default None = no lower bound)
        save_distribution: Whether to save histogram of areas
        filename: Filename for the distribution plot
        
    Returns:
        valid_idx: Boolean array indicating which boxes to keep
        median_area: The median area value
    """
    if len(boxes) == 0:
        return np.array([], dtype=bool), 0
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    
    if save_distribution:
        plt.figure()
        plt.hist(areas, bins=30, color='blue', alpha=0.7)
        plt.axvline(np.median(areas), color='red', linestyle='dashed', linewidth=1)
        plt.title('Distribution of Detected Object Areas')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.savefig(filename)
        plt.close()
    
    median_area = np.median(areas)
    
    # Build filter conditions
    valid_idx = areas <= median_area * upper_multiplier
    if lower_multiplier is not None:
        valid_idx &= (areas >= median_area * lower_multiplier)
    
    print(f"Median area: {median_area:.0f} pixels")
    print(f"Filtered: {len(boxes)} -> {valid_idx.sum()} detections")
    
    return valid_idx, median_area

def _filter_mask_shapes(masks, boxes, rectangularity_threshold=0.85):
    """
    Filter out masks that are too rectangular (likely container edges, not raspberries).
    
    Args:
        masks: numpy array of binary masks [N, H, W]
        boxes: Optional numpy array of bounding boxes in xyxy format [N, 4].
               If None, boxes will be computed from masks.
        rectangularity_threshold: Ratio of mask area to bounding box area (default 0.85)
                                 Higher = more rectangular. Raspberries typically < 0.8
    
    Returns:
        valid_idx: Boolean array indicating which masks to keep
    """
    if len(masks) == 0:
        return np.array([], dtype=bool)
    
    valid_idx = np.ones(len(masks), dtype=bool)
    
    for i, mask in enumerate(masks):
        # Calculate mask area
        mask_area = mask.sum()
        
        if mask_area == 0:
            valid_idx[i] = False
            continue
        
        # Get bounding box area
        x1, y1, x2, y2 = boxes[i]
        bbox_area = (x2 - x1) * (y2 - y1)

        
        # Calculate rectangularity
        rectangularity = mask_area / bbox_area if bbox_area > 0 else 0
        
        # Filter if too rectangular
        if rectangularity > rectangularity_threshold:
            valid_idx[i] = False
            print(f"Filtered mask {i}: rectangularity={rectangularity:.3f}")


    print("Filtered {} masks based on shape (rectangularity threshold={})".format(len(masks) - valid_idx.sum(), rectangularity_threshold))
    
    return valid_idx

def _filter_mask_sizes(masks, upper_multiplier=None, lower_multiplier=0.3, save_distribution=False, filename='mask_area_distribution.png'):
    """
    Filter masks based on actual segmentation area (not bounding box).
    Useful for filtering out small artifacts or container edges with large bboxes but small masks.
    
    Args:
        masks: numpy array of binary masks [N, H, W]
        upper_multiplier: Maximum area as multiple of median (default 4.0)
        lower_multiplier: Minimum area as multiple of median (default None = no lower bound)
        save_distribution: Whether to save histogram of areas
        filename: Filename for the distribution plot
        
    Returns:
        valid_idx: Boolean array indicating which masks to keep
        median_area: The median mask area value
    """
    if len(masks) == 0:
        return np.array([], dtype=bool), 0
    
    # Calculate actual mask areas (number of pixels in each mask)
    areas = np.array([mask.sum() for mask in masks])
    
    if save_distribution:
        plt.figure()
        plt.hist(areas, bins=30, color='blue', alpha=0.7)
        plt.axvline(np.median(areas), color='red', linestyle='dashed', linewidth=1)
        plt.title('Distribution of Mask Areas')
        plt.xlabel('Area (pixels)')
        plt.ylabel('Frequency')
        plt.savefig(filename)
        plt.close()
    
    median_area = np.median(areas)
    
    # Build filter conditions
    valid_idx = areas >= median_area * lower_multiplier
    if upper_multiplier is not None:
        valid_idx &= (areas <= median_area * upper_multiplier)

    print(f"Median mask area: {median_area:.0f} pixels")
    print(f"Filtered: {len(masks)} -> {valid_idx.sum()} detections")
    
    return valid_idx, median_area

def _filter_red_masks(masks, img_array, min_red_fraction=0.4):
    """
    Keep only masks whose covered pixels are predominantly red (raspberry hue).

    Uses HSV hue ranges [0°-10°] and [170°-180°] (red wraps around in OpenCV's
    0-180 hue scale) to identify red pixels.

    Args:
        masks: numpy array of binary masks [N, H, W]
        img_array: RGB image as numpy array [H, W, 3]
        min_red_fraction: Minimum fraction of mask pixels that must be red to keep the mask.

    Returns:
        valid_idx: Boolean array indicating which masks to keep
    """
    if len(masks) == 0:
        return np.array([], dtype=bool)

    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Red occupies two hue bands in OpenCV HSV (H in [0,180])
    red_mask = (
        ((img_hsv[:, :, 0] <= 10) | (img_hsv[:, :, 0] >= 170)) &
        (img_hsv[:, :, 1] >= 60) &   # minimum saturation to exclude near-grey/white
        (img_hsv[:, :, 2] >= 25)      # minimum value to exclude near-black
    )

    valid_idx = np.zeros(len(masks), dtype=bool)
    for i, mask in enumerate(masks):
        pixel_count = mask.sum()
        if pixel_count == 0:
            continue
        red_fraction = red_mask[mask.astype(bool)].sum() / pixel_count
        valid_idx[i] = red_fraction >= min_red_fraction
      #  print(f"  Mask {i}: red fraction = {red_fraction:.2f} -> {'kept' if valid_idx[i] else 'removed'}")

    print(f"Red filter: {len(masks)} -> {valid_idx.sum()} detections")
    return valid_idx


def store_masks(masks, image_ids, conf_scores_list,xyn_list, sample_imgs, filepath):
 
    os.makedirs(filepath, exist_ok=True)

    
    
    if xyn_list is not None:
        # Since we have different amounts of polygon xyn coordinates per object, we cannot store everything in a single numpy array
        # xyn polygon coordinates are needed for YOLO lightweight model training
        output_file = os.path.join(filepath, 'masks.pkl')
        save_data = {i: [mask, xyn, conf_score, sample_img] for i, mask, xyn, conf_score, sample_img in zip(image_ids, masks, xyn_list, conf_scores_list, sample_imgs)}
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
    else:
        output_file = os.path.join(filepath, 'masks.pkl')
        save_data = {i: [mask, conf_score] for i, mask, conf_score in zip(image_ids, masks, conf_scores_list)}
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
   
    print(f"Saved {len(masks)} mask sets to {output_file}")


def main():

    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

   # example_sample = ds['train'][0]
   # bonnet_bbox = extract_bonnet_bounding_box(example_sample, visualize_bool=False)
   # example_img = ds['train'][1]['image']
    # Crop to bonnet area
  #  example_img = example_img.crop((bonnet_bbox[0], bonnet_bbox[1], bonnet_bbox[2], bonnet_bbox[3])) # left, upper, right, lower

    MODE = "SAM3"

    full_data = list(ds['valid']) # list(ds['train']) +
    

   # if MODE == "SAM_manipulate":
   #     model_SAM_manipulate(list(ds['train']), save_imgs_bool = True, store_masks_bool = False, testing_samples = 1, filter_bboxes = (False, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (False, 0.2, None))
        
  #  elif MODE == "SAM":
  #      model_SAM(full_data, mode = 'base', save_imgs_bool = False, store_masks_bool = True, testing_samples = 'all', filter_bboxes = (False, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (False, 0.2, None), filter_red = (False, 0.2))
    
    if MODE == "SAM_extended":
        # extended : added island + overlap filters
        model_SAM_extended(full_data, mode = 'base', save_imgs_bool = False, store_masks_bool = True, testing_samples = 'all', filter_bboxes = (True, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (True, 0.2, None), filter_red = (True, 0.3), filter_holes_islands = True, filter_overlap_masks = True)
    
    elif MODE == "grounding_SAM":
        model_grounding_SAM(full_data, mode = 'base', save_imgs_bool = True, store_masks_bool = False, testing_samples = 1, filter_bboxes = (False, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (False, 0.2, None), filter_holes_islands = False, filter_overlap_masks = False, filter_by_darkness = False)

  #  elif MODE == "FastSAM":
  #      model_FastSAM(list(ds['train']), save_imgs_bool = False, store_masks_bool = False, testing_samples = 15, filter_bboxes = (True, None, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None))

    elif MODE == "SAM3":
        model_SAM3(full_data, save_imgs_bool = False, store_masks_bool = True, testing_samples = 'all', filter_masks_shapes = (False, 0.95), filter_bboxes = (False, 0.2, 3.0), filter_masks_sizes = (False, 0.2, None), prompt = "raspberry", filter_red = (True, 0.3), filter_holes_islands = True, filter_overlap_masks = True)




if __name__ == "__main__":
    main()