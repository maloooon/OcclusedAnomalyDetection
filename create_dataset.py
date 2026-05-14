## Create the dataset of single object masks from occluded objects

import numpy as np
from datasets import load_dataset
import os
from PIL import Image
from evaluation_segmentation import _extract_masks, match_instances, _hungarian_matching
from helper import overlay_raspberries
import matplotlib.pyplot as plt
import pickle
from image_manipulation import edge_smoothing, normalize_distribution, find_holes_fix, apply_specular_suppression, clean_protrusions
import pickle
import random
import shutil
from pathlib import Path
from transformers import pipeline
import torch
import matplotlib.cm as cm
import cv2








def _center_object(masked_img, mask):
    """
    Center the non-zero object in the image without changing image size.
    Also update the mask accordingly.
    """
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return masked_img, mask

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    object_crop = masked_img[y_min:y_max+1, x_min:x_max+1].copy()
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1].copy()
    
    img_h, img_w = masked_img.shape[:2]
    img_center_y, img_center_x = img_h // 2, img_w // 2
    
    obj_h, obj_w = object_crop.shape[:2]
    obj_center_y, obj_center_x = obj_h // 2, obj_w // 2
    
    paste_y = img_center_y - obj_center_y
    paste_x = img_center_x - obj_center_x
    
    centered_img = np.zeros_like(masked_img)
    centered_mask = np.zeros_like(mask)
    
    src_y_start = max(0, -paste_y)
    src_x_start = max(0, -paste_x)
    src_y_end = min(obj_h, img_h - paste_y)
    src_x_end = min(obj_w, img_w - paste_x)
    
    dst_y_start = max(0, paste_y)
    dst_x_start = max(0, paste_x)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    
    centered_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        object_crop[src_y_start:src_y_end, src_x_start:src_x_end]
    centered_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        mask_crop[src_y_start:src_y_end, src_x_start:src_x_end]

    return centered_img, centered_mask

def _crop_image(masked_img, mask):
    """Crop the image to a square based on the smaller dimension."""
    img_h, img_w = masked_img.shape[:2]
    crop_size = min(img_h, img_w)
    center_y, center_x = img_h // 2, img_w // 2
    
    y1 = center_y - crop_size // 2
    y2 = center_y + crop_size // 2
    x1 = center_x - crop_size // 2
    x2 = center_x + crop_size // 2
    
    return masked_img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def _crop_to_bbox(img, mask, padding=10, square=True, pad_value=0, depth_image=None):
    """Crop image, mask, and optionally depth to the bounding box of the mask."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return (img, mask, depth_image) if depth_image is not None else (img, mask)
    
    r_min, r_max = rows[0], rows[-1]
    c_min, c_max = cols[0], cols[-1]
    
    h, w = img.shape[:2]
    r_min = max(0, r_min - padding)
    r_max = min(h - 1, r_max + padding)
    c_min = max(0, c_min - padding)
    c_max = min(w - 1, c_max + padding)
    
    cropped_img = img[r_min:r_max + 1, c_min:c_max + 1]
    cropped_mask = mask[r_min:r_max + 1, c_min:c_max + 1]
    cropped_depth = depth_image[r_min:r_max + 1, c_min:c_max + 1] if depth_image is not None else None
    
    if square:
        ch, cw = cropped_img.shape[:2]
        size = max(ch, cw)
        
        square_img = np.full((size, size, 3), pad_value, dtype=img.dtype)
        square_mask = np.zeros((size, size), dtype=mask.dtype)
        
        y_off = (size - ch) // 2
        x_off = (size - cw) // 2
        square_img[y_off:y_off + ch, x_off:x_off + cw] = cropped_img
        square_mask[y_off:y_off + ch, x_off:x_off + cw] = cropped_mask
        
        if cropped_depth is not None:
            square_depth = np.zeros((size, size), dtype=depth_image.dtype)
            square_depth[y_off:y_off + ch, x_off:x_off + cw] = cropped_depth
            return square_img, square_mask, square_depth
        
        return square_img, square_mask
    
    if cropped_depth is not None:
        return cropped_img, cropped_mask, cropped_depth
    
    return cropped_img, cropped_mask

def _center_and_crop(img, mask, target_size=300, pad_value=0, depth_image=None):
    """Center the raspberry based on mask centroid and crop to fixed size."""
    h, w = img.shape[:2]
    
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        out_img = np.full((target_size, target_size, 3), pad_value, dtype=img.dtype)
        out_mask = np.zeros((target_size, target_size), dtype=mask.dtype)
        if depth_image is not None:
            out_depth = np.zeros((target_size, target_size), dtype=depth_image.dtype)
            return out_img, out_mask, out_depth
        return out_img, out_mask
    
    cy, cx = coords.mean(axis=0).astype(int)
    half = target_size // 2
    
    src_y1 = cy - half
    src_x1 = cx - half
    src_y2 = src_y1 + target_size
    src_x2 = src_x1 + target_size
    
    dst_y1 = max(0, -src_y1)
    dst_x1 = max(0, -src_x1)
    dst_y2 = target_size - max(0, src_y2 - h)
    dst_x2 = target_size - max(0, src_x2 - w)
    
    src_y1 = max(0, src_y1)
    src_x1 = max(0, src_x1)
    src_y2 = min(h, src_y2)
    src_x2 = min(w, src_x2)
    
    out_img = np.full((target_size, target_size, 3), pad_value, dtype=img.dtype)
    out_mask = np.zeros((target_size, target_size), dtype=mask.dtype)
    
    out_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    out_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    
    if depth_image is not None:
        out_depth = np.zeros((target_size, target_size), dtype=depth_image.dtype)
        out_depth[dst_y1:dst_y2, dst_x1:dst_x2] = depth_image[src_y1:src_y2, src_x1:src_x2]
        return out_img, out_mask, out_depth
    
    return out_img, out_mask

def create_dataset_imgs(masks, images, save_path=None, ids=None, 
                        all_gt_masks=None, all_gt_grades=None, img_size=256, spec_supression_bool = False, clean_protrusions_bool = False, filter_holes_bool = False, filter_holes_depth_thresh = 30, filter_holes_brightness_thresh = 80):
    """
    Given masks and images, create a dataset of single-object raspberry images.
    
    CHANGE: This function no longer accepts any filtering parameters.
    It creates the FULL unfiltered dataset. Filtering is done post-hoc
    by apply_filters().
    """

    if save_path:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

    # --- Grade matching (unchanged) ---
    grades_matched = []
    if all_gt_masks and all_gt_grades is not None:
        for pred_masks_img, gt_masks_img, gt_grades_img in zip(masks, all_gt_masks, all_gt_grades):
            curr_grades = np.full(len(pred_masks_img), -1, dtype=int)
            matched_pairs, _, _ = match_instances(pred_masks_img, gt_masks_img, iou_threshold=0.5, gt_grades=gt_grades_img)
            for pred_idx, _, _, grade in matched_pairs:
                curr_grades[pred_idx] = grade
            grades_matched.append(curr_grades)

    # --- Depth estimation (unchanged) ---
    pipe = pipeline(
        task="depth-estimation", 
        model="depth-anything/Depth-Anything-V2-Base-hf", 
        device='cuda:2' if torch.cuda.is_available() else 'cpu'
    )

    depth_masks = [[] for _ in range(len(images))]
    mean_depth_masks = [[] for _ in range(len(images))]

    for idx, (img, img_masks) in enumerate(zip(images, masks)):
        depth_pil = pipe(img)["depth"]
        depth = np.array(depth_pil).astype(np.float32)

        for mask in img_masks:
            depth_masked = np.zeros_like(depth)
            depth_masked[mask > 0] = depth[mask > 0]
            depth_masks[idx].append(depth_masked)
            mean_depth_masks[idx].append(depth[mask > 0].mean())
    
    mean_depth_masks = [np.array(depth_list) for depth_list in mean_depth_masks]

    # --- CHANGE: No filtering here anymore. Straight to dataset assembly. ---

    dataset = []
    for idx, (img, img_masks) in enumerate(zip(images, masks)):
        curr_img_raw = []
        curr_img_processed = []
        curr_masks_raw = []
        curr_masks_processed = []
        curr_masks_unfiltered = []
        curr_depth_raw = []
        curr_depth_processed = []
        curr_hole_booleans = []

        for j, mask in enumerate(img_masks):
            masked_img = img.copy()
            masked_img = np.array(masked_img)
            masked_img[~mask] = 0

            curr_img_raw.append(masked_img)
            curr_masks_raw.append(mask)
            curr_depth_raw.append(depth_masks[idx][j])

            masked_img_processed, mask_processed, depth_processed = _center_and_crop(
                masked_img, mask, target_size=img_size, pad_value=0,
                depth_image=depth_masks[idx][j]
            )

            mask_processed_unfiltered = mask_processed.copy()



            if filter_holes_bool:
                masked_img_processed, mask_processed, depth_processed = find_holes_fix(masked_img_processed, mask_processed, depth_processed, visualize_bool = False, depth_threshold_percentile=filter_holes_depth_thresh,
               brightness_threshold_percentile=filter_holes_brightness_thresh,
               small_hole_max_area=100,
               surrounding_threshold=0.80,
               min_hole_area=200,
               dilation_radius=15,
               border_exclusion_width = 30)


            # This only returns a boolean whether we detected any holes or not, using this for hole training augmentation purposes
            img_hole_boolean = find_holes_fix(masked_img_processed, mask_processed, depth_processed, visualize_bool = False, depth_threshold_percentile=filter_holes_depth_thresh,
            brightness_threshold_percentile=filter_holes_brightness_thresh,
            small_hole_max_area=100,
            surrounding_threshold=0.80,
            min_hole_area=200,
            dilation_radius=15,
            border_exclusion_width = 30,
            return_boolean = True)

            curr_hole_booleans.append(img_hole_boolean)

            if spec_supression_bool:
                masked_img_processed, mask_processed = apply_specular_suppression(masked_img_processed, mask_processed, visualize=False, inpainting = True)


            if clean_protrusions_bool:
                masked_img_processed, mask_processed, depth_processed = clean_protrusions(masked_img_processed, mask_processed, depth_processed)



            curr_img_processed.append(masked_img_processed)
            curr_masks_processed.append(mask_processed)
            curr_masks_unfiltered.append(mask_processed_unfiltered)
            curr_depth_processed.append(depth_processed)

        if ids is not None and grades_matched:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw,
                          curr_masks_processed, curr_masks_unfiltered, curr_depth_raw, curr_depth_processed,
                          curr_hole_booleans, ids[idx], grades_matched[idx]))
        elif ids is not None:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw,
                          curr_masks_processed, curr_masks_unfiltered, curr_depth_raw, curr_depth_processed,
                          curr_hole_booleans, ids[idx]))
        else:
            dataset.extend(list(zip(curr_img_raw, curr_img_processed, curr_masks_raw,
                                   curr_masks_processed, curr_masks_unfiltered, curr_depth_raw, curr_depth_processed,
                                   curr_hole_booleans)))

    # --- Saving logic (unchanged structure, but no filtering branches) ---
    if save_path is not None:
        raw_path = os.path.join(save_path, 'raw')
        processed_path = os.path.join(save_path, 'processed')
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)

        anomalous_processed_path = os.path.join(processed_path, 'anomalous')
        normal_processed_path = os.path.join(processed_path, 'normal')
        for p in [anomalous_processed_path, normal_processed_path]:
            os.makedirs(p, exist_ok=True)

        na_grades = [1, 2, 3]
        a_grades = [4, 5]

        records = {
            'anomalous_processed': [],
            'normal_processed': [],
        }

        for i, item in enumerate(dataset):
            if ids is not None and grades_matched:
                imgs_raw, imgs_processed, masks_raw, masks_processed, masks_unfiltered, depths_raw, depths_processed, hole_booleans, img_id, img_grades = item
                
                raw_img_folder = os.path.join(raw_path, img_id)
                processed_img_folder = os.path.join(processed_path, img_id)
                os.makedirs(raw_img_folder, exist_ok=True)
                os.makedirs(processed_img_folder, exist_ok=True)

                for j, (curr_img_raw, curr_img_processed, curr_depth_raw, curr_depth_processed) in enumerate(
                    zip(imgs_raw, imgs_processed, depths_raw, depths_processed)
                ):
                    grade = int(img_grades[j])
                    # CHANGE: filename uses original img_id and obj index j — always stable
                    img_filename = f"{img_id}_obj{j}_grade{grade}.png"

                    
                    raw_img_path = os.path.join(raw_img_folder, img_filename)
                    img_pil_raw = Image.fromarray(curr_img_raw.astype(np.uint8))
                    img_pil_raw.save(raw_img_path)
                    
                    processed_img_path = os.path.join(processed_img_folder, img_filename)
                    img_pil_processed = Image.fromarray(curr_img_processed.astype(np.uint8))
                    img_pil_processed.save(processed_img_path)

                    if grade in a_grades:
                        proc_dest = anomalous_processed_path
                        proc_key = 'anomalous_processed'
                    elif grade in na_grades:
                        proc_dest = normal_processed_path
                        proc_key = 'normal_processed'
                    else:
                        continue

                    shutil.copy2(processed_img_path, os.path.join(proc_dest, img_filename))

                    records[proc_key].append({
                        'img_path': os.path.join(proc_dest, img_filename),
                        'grade': grade,
                        'mask': masks_processed[j],
                        'mask_unfiltered': masks_unfiltered[j],
                        'image': curr_img_processed,
                        'depth': curr_depth_processed,
                        'has_hole': hole_booleans[j],
                    })

                raw_data_dict = {
                    'images': np.array(imgs_raw, dtype=object),  
                    'masks': np.array(masks_raw, dtype=object),  
                    'grades': np.array(img_grades, dtype=np.int32),  
                    'img_id': img_id
                }
                processed_data_dict = {
                    'images': np.array(imgs_processed, dtype=object), 
                    'masks': np.array(masks_processed, dtype=object),  
                    'grades': np.array(img_grades, dtype=np.int32), 
                    'img_id': img_id
                }
                np.savez_compressed(os.path.join(raw_img_folder, f'raw_{img_id}_data.npz'), **raw_data_dict)
                np.savez_compressed(os.path.join(processed_img_folder, f'processed_{img_id}_data.npz'), **processed_data_dict)

            elif ids is not None:
                imgs_raw, imgs_processed, masks_raw, masks_processed, masks_unfiltered, depths_raw, depths_processed, hole_booleans, img_id = item
                
                raw_img_folder = os.path.join(raw_path, img_id)
                processed_img_folder = os.path.join(processed_path, img_id)
                os.makedirs(raw_img_folder, exist_ok=True)
                os.makedirs(processed_img_folder, exist_ok=True)
                
                for j, (img_raw, img_processed, depth_raw, depth_processed) in enumerate(
                    zip(imgs_raw, imgs_processed, depths_raw, depths_processed)
                ):
                    img_filename = f"{img_id}_obj{j}.png"
                    
                    raw_img_path = os.path.join(raw_img_folder, img_filename)
                    img_pil_raw = Image.fromarray(img_raw.astype(np.uint8))
                    img_pil_raw.save(raw_img_path)

                    processed_img_path = os.path.join(processed_img_folder, img_filename)
                    img_pil_processed = Image.fromarray(img_processed.astype(np.uint8))
                    img_pil_processed.save(processed_img_path)

                    shutil.copy2(processed_img_path, os.path.join(normal_processed_path, img_filename))

                    records['normal_processed'].append({
                        'img_path': os.path.join(normal_processed_path, img_filename),
                        'grade': -1,
                        'mask': masks_processed[j],
                        'mask_unfiltered': masks_unfiltered[j],
                        'image': img_processed,
                        'depth': depth_processed,  
                        'has_hole': hole_booleans[j],
                    })

                raw_data_dict = {
                    'images': np.array(imgs_raw, dtype=object),  
                    'masks': np.array(masks_raw, dtype=object),  
                    'img_id': img_id
                }
                processed_data_dict = {
                    'images': np.array(imgs_processed, dtype=object), 
                    'masks': np.array(masks_processed, dtype=object),  
                    'img_id': img_id
                }
                np.savez_compressed(os.path.join(raw_img_folder, f'raw_{img_id}_data.npz'), **raw_data_dict)
                np.savez_compressed(os.path.join(processed_img_folder, f'processed_{img_id}_data.npz'), **processed_data_dict)

            else:
                imgs_raw, imgs_processed, masks_raw, masks_processed, masks_unfiltered, depths_raw, depths_processed, hole_booleans = item
                
                raw_img_folder = os.path.join(raw_path, f"img_{i}")
                processed_img_folder = os.path.join(processed_path, f"img_{i}")
                os.makedirs(raw_img_folder, exist_ok=True)
                os.makedirs(processed_img_folder, exist_ok=True)

                for j, (img_raw, img_processed, depth_raw, depth_processed) in enumerate(
                    zip(imgs_raw, imgs_processed, depths_raw, depths_processed)
                ):
                    img_filename = f"img_{i}_obj{j}.png"
                    
                    raw_img_path = os.path.join(raw_img_folder, img_filename)
                    img_pil_raw = Image.fromarray(img_raw.astype(np.uint8))
                    img_pil_raw.save(raw_img_path)

                    processed_img_path = os.path.join(processed_img_folder, img_filename)
                    img_pil_processed = Image.fromarray(img_processed.astype(np.uint8))
                    img_pil_processed.save(processed_img_path)

                    shutil.copy2(processed_img_path, os.path.join(normal_processed_path, img_filename))

                    records['normal_processed'].append({
                        'img_path': os.path.join(normal_processed_path, img_filename),
                        'grade': -1,
                        'mask': masks_processed[j],
                        'mask_unfiltered': masks_unfiltered[j],
                        'image': img_processed,
                        'depth': depth_processed,
                        'has_hole': hole_booleans,
                    })

                raw_data_dict = {
                    'images': np.array(imgs_raw, dtype=object),  
                    'masks': np.array(masks_raw, dtype=object),  
                }
                processed_data_dict = {
                    'images': np.array(imgs_processed, dtype=object), 
                    'masks': np.array(masks_processed, dtype=object),  
                }
                np.savez_compressed(os.path.join(raw_img_folder, f'raw_img_{i}_data.npz'), **raw_data_dict)
                np.savez_compressed(os.path.join(processed_img_folder, f'processed_img_{i}_data.npz'), **processed_data_dict)

        # Save pkl files
        with open(os.path.join(anomalous_processed_path, 'anomalous_samples.pkl'), 'wb') as f:
            pickle.dump(records['anomalous_processed'], f)
        with open(os.path.join(normal_processed_path, 'normal_samples.pkl'), 'wb') as f:
            pickle.dump(records['normal_processed'], f)

        print(f"[processed] Anomalous: {len(records['anomalous_processed'])}, Normal: {len(records['normal_processed'])}")

        # Overlay visualization (unchanged)
        raw_folder = os.path.join(save_path, 'raw')
        for subfolder_name in os.listdir(raw_folder):
            subfolder_path = os.path.join(raw_folder, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue
            output = os.path.join(subfolder_path, f'{subfolder_name}_overlayed_raspberries.png')
            overlay_raspberries(subfolder_path, output)

    return dataset

def data_split_non_anomalous(data_path_normal, data_path_anomalous, save_path, seed = 42):
    """
    Split non-anomalous samples into train/test, balanced against anomalous count.
    Deterministic by seed.
    """
    random.seed(seed) # For reproducibility

    splits_path = Path(save_path)
    splits_path.mkdir(parents=True, exist_ok=True)

    with open(data_path_normal, 'rb') as f:
        normal_data = pickle.load(f)
    with open(data_path_anomalous, 'rb') as f:
        anomalous_data = pickle.load(f)

    n_normal = len(normal_data)
    n_anomalous = len(anomalous_data)

    if n_anomalous > n_normal:
        raise ValueError(
            f"Number of anomalous samples ({n_anomalous}) "
            f"is larger than number of normal samples ({n_normal}). "
            "Balanced split is not possible."
        )

    all_normal_indices = list(range(n_normal))
    random.shuffle(all_normal_indices)

    test_normal_indices = all_normal_indices[:n_anomalous]
    train_normal_indices = all_normal_indices[n_anomalous:]

    train_normal_paths = [normal_data[i]['img_path'] for i in train_normal_indices]
    test_normal_paths = [normal_data[i]['img_path'] for i in test_normal_indices]

    save_path = Path(save_path)
    with open(save_path / 'train_normal_indices.pkl', 'wb') as f:
        pickle.dump(train_normal_indices, f)
    with open(save_path / 'test_normal_indices.pkl', 'wb') as f:
        pickle.dump(test_normal_indices, f)
    with open(save_path / 'train_normal_paths.pkl', 'wb') as f:
        pickle.dump(train_normal_paths, f)
    with open(save_path / 'test_normal_paths.pkl', 'wb') as f:
        pickle.dump(test_normal_paths, f)

def apply_filters(dataset_path,
                  size_filtering=False, size_filtering_factor=1.5,
                  darkness_filtering=False, darkness_threshold=80, max_dark_ratio=0.3):
    """
    Post-hoc filtering on a fully created and split dataset. Modifies the dataset IN-PLACE.

    After this function runs:
      - filtered-out images are saved to {parent}/filtered/{size,darkness}/ for inspection
      - filtered-out images are REMOVED from processed/normal/ and processed/anomalous/
      - normal_samples.pkl and anomalous_samples.pkl are OVERWRITTEN with filtered versions
      - split index and path files are OVERWRITTEN with filtered versions
      - downstream code (SingleRaspberryDataset) works identically without any filter_path

    Args:
        dataset_path: path to the 'processed' folder (contains normal/, anomalous/, splits/)
        size_filtering: whether to remove too-small raspberries
        size_filtering_factor: MAD multiplier for size lower bound
        darkness_filtering: whether to remove too-dark raspberries
        darkness_threshold: HSV V-channel threshold for "dark" pixels
        max_dark_ratio: max fraction of dark pixels before removal
    """
    dataset_path = Path(dataset_path)

    if not size_filtering and not darkness_filtering:
        print("No filters requested, nothing to do.")
        return

    # --- Create filtered/ folder structure ---
    # Lives BESIDE processed/ and raw/, i.e. one level up from dataset_path
    filtered_base = dataset_path.parent / 'filtered'
    if filtered_base.exists():
        shutil.rmtree(filtered_base)

    if size_filtering:
        filtered_size_path = filtered_base / 'size'
        filtered_size_path.mkdir(parents=True, exist_ok=True)
    if darkness_filtering:
        filtered_darkness_path = filtered_base / 'darkness'
        filtered_darkness_path.mkdir(parents=True, exist_ok=True)

    # --- Load the full unfiltered data ---
    normal_pkl_path = dataset_path / 'normal' / 'normal_samples.pkl'
    anomalous_pkl_path = dataset_path / 'anomalous' / 'anomalous_samples.pkl'
    splits_path = dataset_path / 'splits'

    with open(normal_pkl_path, 'rb') as f:
        normal_data = pickle.load(f)
    with open(anomalous_pkl_path, 'rb') as f:
        anomalous_data = pickle.load(f)
    with open(splits_path / 'train_normal_indices.pkl', 'rb') as f:
        train_normal_indices = pickle.load(f)
    with open(splits_path / 'test_normal_indices.pkl', 'rb') as f:
        test_normal_indices = pickle.load(f)

    # Tag every record with its source for bookkeeping
    all_samples = []
    for i, rec in enumerate(normal_data):
        all_samples.append({**rec, '_source': 'normal', '_orig_idx': i})
    for i, rec in enumerate(anomalous_data):
        all_samples.append({**rec, '_source': 'anomalous', '_orig_idx': i})

    # --- Build removal set, keyed by basename ---
    from collections import defaultdict
    remove_info = {}  # basename -> {'reason': str, ...}

    if size_filtering:
        # Group by source image to compute per-image statistics
        img_groups = defaultdict(list)
        for sample in all_samples:
            basename = Path(sample['img_path']).name
            img_id = basename.split('_obj')[0]
            mask_area = sample['mask'].sum()
            img_groups[img_id].append((basename, mask_area, sample))

        for img_id, group in img_groups.items():
            sizes = np.array([area for _, area, _ in group])
            median_size = np.median(sizes)
            mad_size = np.median(np.abs(sizes - median_size))
            size_lower_bound = median_size - size_filtering_factor * mad_size

            for basename, area, sample in group:
                if area < size_lower_bound:
                    remove_info[basename] = {
                        'reason': 'size',
                        'area': int(area),
                        'threshold': float(size_lower_bound),
                    }

    if darkness_filtering:
        for sample in all_samples:
            basename = Path(sample['img_path']).name
            if basename in remove_info:
                continue  # Already marked by size filter

            img = sample['image']
            mask = sample['mask']

            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            brightness = hsv[:, :, 2].astype(np.float32)
            masked_brightness = brightness[mask > 0]

            if masked_brightness.size == 0:
                continue

            dark_ratio = (masked_brightness < darkness_threshold).sum() / masked_brightness.size

            if dark_ratio > max_dark_ratio:
                remove_info[basename] = {
                    'reason': 'darkness',
                    'dark_ratio': float(dark_ratio),
                    'threshold': float(max_dark_ratio),
                }

    # --- Step 1: Save filtered-out images to filtered/{reason}/ for inspection ---
    for sample in all_samples:
        basename = Path(sample['img_path']).name
        if basename not in remove_info:
            continue

        reason = remove_info[basename]['reason']
        if reason == 'size':
            dest_dir = filtered_size_path
        elif reason == 'darkness':
            dest_dir = filtered_darkness_path
        else:
            continue

        # Copy the processed image to the filtered folder
        src_path = Path(sample['img_path'])
        if src_path.exists():
            shutil.copy2(src_path, dest_dir / basename)
        else:
            # Fallback: save from the in-memory image array
            img_pil = Image.fromarray(sample['image'].astype(np.uint8))
            img_pil.save(dest_dir / basename)

    # --- Step 2: Delete filtered-out images from processed/normal/ and processed/anomalous/ ---
    for basename in remove_info:
        for folder in [dataset_path / 'normal', dataset_path / 'anomalous']:
            img_file = folder / basename
            if img_file.exists():
                img_file.unlink()

    # --- Step 3: Build filtered data lists and OVERWRITE the original pkl files ---
    filtered_normal = []
    filtered_anomalous = []

    # We need a mapping from old normal index -> new normal index for the split files
    old_to_new_normal_idx = {}
    new_normal_idx = 0

    for sample in all_samples:
        basename = Path(sample['img_path']).name
        if basename in remove_info:
            continue

        # Strip internal tags
        clean_record = {k: v for k, v in sample.items() if not k.startswith('_')}

        if sample['_source'] == 'normal':
            old_to_new_normal_idx[sample['_orig_idx']] = new_normal_idx
            new_normal_idx += 1
            filtered_normal.append(clean_record)
        else:
            filtered_anomalous.append(clean_record)

    # Overwrite the pkl files
    with open(normal_pkl_path, 'wb') as f:
        pickle.dump(filtered_normal, f)
    with open(anomalous_pkl_path, 'wb') as f:
        pickle.dump(filtered_anomalous, f)

    # --- Step 4: Recompute and OVERWRITE the split index and path files ---
    # The original indices pointed into the old normal_data list.
    # Remap them to point into the new (shorter) filtered_normal list.
    # Any index whose sample was filtered out is simply dropped.

    new_train_indices = []
    new_test_indices = []

    for old_idx in train_normal_indices:
        if old_idx in old_to_new_normal_idx:
            new_train_indices.append(old_to_new_normal_idx[old_idx])

    for old_idx in test_normal_indices:
        if old_idx in old_to_new_normal_idx:
            new_test_indices.append(old_to_new_normal_idx[old_idx])

    new_train_paths = [filtered_normal[i]['img_path'] for i in new_train_indices]
    new_test_paths = [filtered_normal[i]['img_path'] for i in new_test_indices]

    with open(splits_path / 'train_normal_indices.pkl', 'wb') as f:
        pickle.dump(new_train_indices, f)
    with open(splits_path / 'test_normal_indices.pkl', 'wb') as f:
        pickle.dump(new_test_indices, f)
    with open(splits_path / 'train_normal_paths.pkl', 'wb') as f:
        pickle.dump(new_train_paths, f)
    with open(splits_path / 'test_normal_paths.pkl', 'wb') as f:
        pickle.dump(new_test_paths, f)

    # --- Save removal log for debugging ---
    with open(filtered_base / 'removed_samples.pkl', 'wb') as f:
        pickle.dump(remove_info, f)

    # --- Summary ---
    n_size = sum(1 for v in remove_info.values() if v['reason'] == 'size')
    n_dark = sum(1 for v in remove_info.values() if v['reason'] == 'darkness')
    print(f"[apply_filters] Removed {len(remove_info)} samples total")
    print(f"  Size filter: {n_size}")
    print(f"  Darkness filter: {n_dark}")
    print(f"  Remaining normal: {len(filtered_normal)} (train: {len(new_train_indices)}, test: {len(new_test_indices)})")
    print(f"  Remaining anomalous: {len(filtered_anomalous)}")
    print(f"  Filtered images saved to: {filtered_base}")




def main():

    CREATE_BASED_ON = 'yolo_640' # Options: 'gt', 'sam3' 'yolo', 'yolo_fullsize', 'yolo_640'

    # --- Config ---
    IMG_SIZE = 256
    UNBLURRED = False
    SPECULAR_SUPPRESSION = False
    CLEAN_PROTRUSIONS = False
    FILTER_HOLES = False
    HOLES_DEPTH_THRESH = 40
    HOLES_BRIGHTNESS_THRESH = 40
    SEED = 0
 

    SIZE_FILTERING = False
    SIZE_FILTERING_FACTOR = 1.5
    DARKNESS_FILTERING = False # Fully filters out too dark samples
    DARKNESS_THRESHOLD = 80
    MAX_DARK_RATIO = 0.3
 
    # --- Build SAVE_PATH name (restored from original main) ---
    # The folder name describes the final state of the dataset, including filters.
    # Even though create_dataset_imgs creates the full version first, apply_filters
    # modifies it in-place, so the name should reflect the end result.
    filter_parts = []
    if SIZE_FILTERING:
        filter_parts.append(f"size_{SIZE_FILTERING_FACTOR}")
    if DARKNESS_FILTERING:
        filter_parts.append(f"darkness_{DARKNESS_THRESHOLD}_{MAX_DARK_RATIO}")
    if UNBLURRED:
        filter_parts.append("unblurred")
    if SPECULAR_SUPPRESSION:
        filter_parts.append("specular_suppression")
    if CLEAN_PROTRUSIONS:
        filter_parts.append("clean_protrusions")
    if FILTER_HOLES:
        filter_parts.append("filter_holes")
        filter_parts.append(f"holes_depth_{HOLES_DEPTH_THRESH}")
        filter_parts.append(f"holes_brightness_{HOLES_BRIGHTNESS_THRESH}")
 
    if filter_parts:
        filter_str = "filtered_" + "_and_".join(filter_parts)
        filter_str += "_seed_" + str(SEED)
    else:
        filter_str = "full_no_filters" + "_seed_" + str(SEED)
    
    filter_str += f"_{CREATE_BASED_ON}"
 
    SAVE_PATH = f'../../nvme1/dataset_single_objects/GT/{filter_str}_{IMG_SIZE}' # NOTE : disk normally
    if CREATE_BASED_ON != 'gt':
        PRED_MASKS_FILE = f'../../disk/saved_masks/{CREATE_BASED_ON}/masks.pkl'
    else:
        PRED_MASKS_FILE = f'../../disk/saved_masks/sam3/masks.pkl' # irrelevant, just for loading purposes


    
    ds = load_dataset("FBK-TeV/RaspGrade")
    train_data = list(ds['train'])
    valid_data = list(ds['valid'])

    if UNBLURRED:
        unblurred_dir = '../../disk/dataset_bonnets/raspberries_unblurred'
        for i, sample in enumerate(train_data):
            sample['image'] = Image.open(f'{unblurred_dir}/{i}.jpg').convert('RGB')
        for j, sample in enumerate(valid_data):
            sample['image'] = Image.open(f'{unblurred_dir}/{len(train_data) + j}.jpg').convert('RGB')

    full_data = train_data + valid_data
    all_imgs_ids = [(sample['image'], sample['image_id']) for sample in full_data]
    all_gt_masks_ids, all_gt_grades_ids = _extract_masks(full_data, extract_grades_bool=True)

    with open(PRED_MASKS_FILE, 'rb') as f:
        pred_data = pickle.load(f)

    all_pred_masks_ids = [(pred_data[key], key) for key in pred_data.keys()]

    # Sort by ID (unchanged)
    all_imgs_ids.sort(key=lambda x: x[1])
    all_pred_masks_ids.sort(key=lambda x: x[1])
    all_gt_masks_ids.sort(key=lambda x: x[1])
    all_gt_grades_ids.sort(key=lambda x: x[1])

    if isinstance(all_pred_masks_ids[0][0], list) and len(all_pred_masks_ids[0][0]) == 4:
        all_pred_masks = [masks_and_xyn_and_conf_scores_and_imgs[0] for masks_and_xyn_and_conf_scores_and_imgs, img_id in all_pred_masks_ids]
        all_conf_scores = [masks_and_xyn_and_conf_scores_and_imgs[2] for masks_and_xyn_and_conf_scores_and_imgs, img_id in all_pred_masks_ids]
    else:
        all_pred_masks = [masks_and_conf_scores[0] for masks_and_conf_scores, img_id in all_pred_masks_ids]
        all_conf_scores = [masks_and_conf_scores[1] for masks_and_conf_scores, img_id in all_pred_masks_ids]

 

    all_imgs = [img for img, _ in all_imgs_ids]
    all_gt_masks = [masks for masks, _ in all_gt_masks_ids]
    all_gt_grades = [grades for grades, _ in all_gt_grades_ids]
    all_ids = [img_id for _, img_id in all_pred_masks_ids]


    # --- Step 1: Create full unfiltered dataset ---

    masks = all_gt_masks if CREATE_BASED_ON == 'gt' else all_pred_masks

    dataset_single_objects = create_dataset_imgs(
        masks, all_imgs, save_path=SAVE_PATH, ids=all_ids,
        all_gt_masks=all_gt_masks, all_gt_grades=all_gt_grades,
        img_size=IMG_SIZE, spec_supression_bool = SPECULAR_SUPPRESSION,
        clean_protrusions_bool = CLEAN_PROTRUSIONS, filter_holes_bool = FILTER_HOLES,
        filter_holes_depth_thresh=HOLES_DEPTH_THRESH, filter_holes_brightness_thresh=HOLES_BRIGHTNESS_THRESH
    )

    # --- Step 2: Create deterministic split (unchanged logic) ---
    data_split_non_anomalous(
        data_path_normal=f'{SAVE_PATH}/processed/normal/normal_samples.pkl',
        data_path_anomalous=f'{SAVE_PATH}/processed/anomalous/anomalous_samples.pkl',
        save_path=f'{SAVE_PATH}/processed/splits',
        seed = SEED
    )

    # --- Step 3 (NEW): Apply filters post-hoc if requested ---
    if SIZE_FILTERING or DARKNESS_FILTERING:
        apply_filters(
            dataset_path=f'{SAVE_PATH}/processed',
            size_filtering=SIZE_FILTERING,
            size_filtering_factor=SIZE_FILTERING_FACTOR,
            darkness_filtering=DARKNESS_FILTERING,
            darkness_threshold=DARKNESS_THRESHOLD,
            max_dark_ratio=MAX_DARK_RATIO,
        )


if __name__ == "__main__":
    main()