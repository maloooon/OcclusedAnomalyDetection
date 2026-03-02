## Create the dataset of single object masks from occluded objects

import numpy as np
from datasets import load_dataset
import os
from PIL import Image
from evaluation_segmentation import _extract_masks, match_instances, _hungarian_matching
from helper import overlay_raspberries
import matplotlib.pyplot as plt
import pickle
from image_manipulation import edge_smoothing, normalize_distribution
import pickle
import random
import shutil
from pathlib import Path
from transformers import pipeline
import torch
import matplotlib.cm as cm


def _center_object(masked_img, mask):
    """
    Center the non-zero object in the image without changing image size.
    Also update the mask accordingly.
    
    Returns:
        centered_img: Centered image
        centered_mask: Updated boolean mask
    """
    # Find bounding box of non-zero pixels in the MASK (not the image)
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return masked_img, mask  # Empty mask, return as is
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extract the object and its mask
    object_crop = masked_img[y_min:y_max+1, x_min:x_max+1].copy()
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1].copy() # extracts rectanglar area around the mask, but keeps the mask shape intact
    
    # Calculate center of original image
    img_h, img_w = masked_img.shape[:2]
    img_center_y, img_center_x = img_h // 2, img_w // 2
    
    # Calculate center of object
    obj_h, obj_w = object_crop.shape[:2]
    obj_center_y, obj_center_x = obj_h // 2, obj_w // 2
    
    # Calculate top-left position to center the object
    paste_y = img_center_y - obj_center_y
    paste_x = img_center_x - obj_center_x
    
    # Create new blank image and mask
    centered_img = np.zeros_like(masked_img)
    centered_mask = np.zeros_like(mask)
    
    # Calculate valid paste region
    src_y_start = max(0, -paste_y)
    src_x_start = max(0, -paste_x)
    src_y_end = min(obj_h, img_h - paste_y)
    src_x_end = min(obj_w, img_w - paste_x)
    
    dst_y_start = max(0, paste_y)
    dst_x_start = max(0, paste_x)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    
    # Paste the object and mask
    centered_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        object_crop[src_y_start:src_y_end, src_x_start:src_x_end]
    centered_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        mask_crop[src_y_start:src_y_end, src_x_start:src_x_end]
    

    # Visualize mask overlay (for debugging)
    #plt.imshow(centered_img)
    #plt.imshow(centered_mask, alpha=0.5)
    #plt.show()

    return centered_img, centered_mask

def _crop_image(masked_img, mask):
    """
    Crop the image to a square based on the smaller dimension.
    Removes equal amounts from both sides of the larger dimension.
    In MVtec, images are between 700x700 and 1024x1024 dimensions,
    so for the raspberry dataset, we will have 800x800 images.

    Args : 
        masked_img: The input masked image (numpy array).
        mask: The corresponding boolean mask (numpy array).
    
    Notes :
        1. Assumes the object is already centered in the image.
    
    
    """
    img_h, img_w = masked_img.shape[:2]
    
    # Determine the smaller dimension (this will be our square size)
    crop_size = min(img_h, img_w)
    
    # Calculate center
    center_y, center_x = img_h // 2, img_w // 2
    
    # Calculate crop coordinates (centered)
    y1 = center_y - crop_size // 2
    y2 = center_y + crop_size // 2
    x1 = center_x - crop_size // 2
    x2 = center_x + crop_size // 2
    
    return masked_img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def _crop_to_bbox(img, mask, padding=10, square=True, pad_value=0):

    """
    Crop image and mask to the bounding box of the mask.
    
    Args:
        img: numpy array (H, W, 3)
        mask: boolean numpy array (H, W)
        padding: pixels to add around the bbox
        square: if True, pad to square before returning
        pad_value: value to fill padded regions (0 = black)
    
    Returns:
        cropped_img, cropped_mask
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    
    if len(rows) == 0 or len(cols) == 0:
        return img, mask
    
    r_min, r_max = rows[0], rows[-1]
    c_min, c_max = cols[0], cols[-1]
    
    # Add padding, clamp to image bounds
    h, w = img.shape[:2]
    r_min = max(0, r_min - padding)
    r_max = min(h - 1, r_max + padding)
    c_min = max(0, c_min - padding)
    c_max = min(w - 1, c_max + padding)
    
    cropped_img = img[r_min:r_max + 1, c_min:c_max + 1]
    cropped_mask = mask[r_min:r_max + 1, c_min:c_max + 1]
    
    if square:
        ch, cw = cropped_img.shape[:2]
        size = max(ch, cw)
        
        # Create square canvas
        square_img = np.full((size, size, 3), pad_value, dtype=img.dtype)
        square_mask = np.zeros((size, size), dtype=mask.dtype)
        
        # Center the crop in the square
        y_off = (size - ch) // 2
        x_off = (size - cw) // 2
        square_img[y_off:y_off + ch, x_off:x_off + cw] = cropped_img
        square_mask[y_off:y_off + ch, x_off:x_off + cw] = cropped_mask
        
        return square_img, square_mask
    
    return cropped_img, cropped_mask


def create_dataset_imgs(masks, images, save_path=None, ids=None, all_gt_masks = None, all_gt_grades = None):
    """
    Given a list of lists of masks (True/False values) for each image, 
    create a dataset of single object images (RGB format).

    Args:
        masks: List of lists of boolean masks for each image (predicted by some model) ; these are the masks that are used to create the dataset.
        images: List of original images (PIL format) ; the original images from the original dataset.
        save_path: Optional path to save the single object images
        ids: Optional list of IDs corresponding to each image
        all_gt_masks: Optional list of lists of ground truth masks for each image
        all_gt_grades: Optional list of lists of ground truth grades for each image

    Notes :
        1. all_gt_masks, if provided, are used to match predicted masks to ground truth masks
              using Hungarian matching, and then assigning all_gt_grades, if provided, accordingly. This
              is important for our anomaly detection algorithm later, i.e. we need to know
              what objects have what grade (i.e. what level of anomaly).
    
    """

    # Empty the save path if it already exists (to avoid confusion with old data)
    if save_path:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        
        # Create save path
        os.makedirs(save_path, exist_ok=True)

    grades_matched = []
    if all_gt_masks and all_gt_grades is not None:
        # Match predicted masks to ground truth masks for each image
        for pred_masks_img, gt_masks_img, gt_grades_img in zip(masks, all_gt_masks, all_gt_grades):
            curr_grades = np.full(len(pred_masks_img), -1, dtype=int) # -1 indicates no match (i.e predicted mask has no corresponding GT mask)
            matched_pairs, _, _ = match_instances(pred_masks_img, gt_masks_img, iou_threshold= 0.5, gt_grades=gt_grades_img)
            for pred_idx, _, _, grade in matched_pairs:
                curr_grades[pred_idx] = grade
            grades_matched.append(curr_grades)

    
    # Filtering based on depth and mask sizes
    save_path_depths = "../../disk/depth_masks"
    # Delete if it already exists (to avoid confusion with old data), then create
    if os.path.exists(save_path_depths):
        shutil.rmtree(save_path_depths)
    os.makedirs(save_path_depths, exist_ok=True)
     # Load depth model ONCE outside the loop
    pipe = pipeline(
        task="depth-estimation", 
        model="depth-anything/Depth-Anything-V2-Base-hf", 
        device='cuda:2' if torch.cuda.is_available() else 'cpu'
    )

    mean_depth_masks = [[] for _ in range(len(images))]
    depth_masks = [[] for _ in range(len(images))]

    for idx, (img, img_masks) in enumerate(zip(images, masks)):
        depth_pil = pipe(img)["depth"]

        depth_norm = np.array(depth_pil).astype(np.float32)
        depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
        depth_colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)
        
        Image.fromarray(depth_colored).save(os.path.join(save_path_depths, f"img{idx:03d}_depth.png"))
        depth = np.array(depth_pil)  # Convert PIL -> numpy
        

        for mask in img_masks:
            depth_masked = np.zeros_like(depth)
            depth_masked[mask > 0] = depth[mask > 0]

            mean_depth_masks[idx].append(depth[mask > 0].mean())
            depth_masks[idx].append(depth_masked)

    mean_depth_masks = [np.array(depth_list) for depth_list in mean_depth_masks]

    removed_dir = "../../disk/removed_raspberries"
    # Delete if it already exists (to avoid confusion with old data), then create
    if os.path.exists(removed_dir):
        shutil.rmtree(removed_dir)
    os.makedirs(removed_dir, exist_ok=True)
    removed_count = 0

    # Filter individual raspberries that are both too small AND too deep (background noise)
    filtered_masks = []
    filtered_images = []
    filtered_mean_depth_masks = []
    filtered_depth_masks = []
    filtered_grades = []

    for img_idx in range(len(masks)):
        img_masks = masks[img_idx]
        img_depths = mean_depth_masks[img_idx]
        img = images[img_idx]


        sizes = np.array([mask.sum() for mask in img_masks])
        depths = np.array(img_depths)

        # TODO : maybe even absolute filtering better, since raspberries same size, same conditions ... just filter by absolute size ?? 
        median_size = np.median(sizes)
        mad_size = np.median(np.abs(sizes - median_size))
        size_lower_bound = median_size - 0.5 * mad_size

        median_depth = np.median(depths)
        mad_depth = np.median(np.abs(depths - median_depth))
        depth_upper_bound = median_depth + 1 * mad_depth

        keep_indices = []
        for i in range(len(img_masks)):
            too_small = sizes[i] < size_lower_bound
            too_deep = depths[i] > depth_upper_bound
            if too_small: #or too_deep:
                # Save removed raspberry
                mask = img_masks[i]
                img_array = np.array(img)  
                cropped = img_array.copy()
                cropped[mask == 0] = 0

                # Crop to bounding box of the mask
                ys, xs = np.where(mask > 0)
                cropped = cropped[ys.min():ys.max()+1, xs.min():xs.max()+1]

                Image.fromarray(cropped).save(
                    os.path.join(removed_dir, f"img{img_idx:03d}_mask{i:02d}.png")
                )
                removed_count += 1
            else:
                keep_indices.append(i)

        filtered_masks.append([img_masks[i] for i in keep_indices])
        filtered_images.append(img)
        filtered_mean_depth_masks.append([img_depths[i] for i in keep_indices])
        filtered_depth_masks.append([depth_masks[img_idx][i] for i in keep_indices])
        filtered_grades.append(grades_matched[img_idx][keep_indices] if grades_matched else None)
    print(f"Removed {removed_count} raspberries, saved to {removed_dir}")

    masks = filtered_masks
    images = filtered_images
    mean_depth_masks = filtered_mean_depth_masks
    depth_masks = filtered_depth_masks
    grades_matched = filtered_grades


    dataset = []
    for idx, (img, img_masks) in enumerate(zip(images, masks)):
        curr_img_raw = []
        curr_img_processed = []
        curr_masks_raw = []
        curr_masks_processed = []

        
        for mask in img_masks:
            # Create a new image for each mask
            masked_img = img.copy()
            masked_img = np.array(masked_img)
            masked_img[~mask] = 0  # Apply mask

            # Store raw version
            curr_img_raw.append(masked_img)
            curr_masks_raw.append(mask)

            # Center the object in the image
          #  masked_img_processed, mask_processed = _center_object(masked_img, mask)
            # Crop to square
           # masked_img_processed, mask_processed = _crop_image(masked_img_processed, mask_processed)

            masked_img_processed, mask_processed = _crop_to_bbox(masked_img, mask, padding=10, square=True, pad_value=0)
          #  masked_img_processed, mask_processed = normalize_distribution(masked_img_processed, mask_processed, target_mean = 134.08394672413232, target_std = 31.834956813835767)
          #  masked_img_processed, mask_processed = edge_smoothing(masked_img_processed, mask_processed)

            # Store processed version
            curr_img_processed.append(masked_img_processed)
            curr_masks_processed.append(mask_processed)

        if ids is not None and grades_matched:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, ids[idx], grades_matched[idx]))
        elif ids is not None:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, ids[idx]))
        else:
            dataset.extend(list(zip(curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed)))

    if save_path is not None:
            # Create raw and processed subdirectories
            raw_path = os.path.join(save_path, 'raw')
            processed_path = os.path.join(save_path, 'processed')
            os.makedirs(raw_path, exist_ok=True)
            os.makedirs(processed_path, exist_ok=True)

            # Create anomalous/normal folders for processed only
            anomalous_processed_path = os.path.join(processed_path, 'anomalous')
            normal_processed_path = os.path.join(processed_path, 'normal')
            for p in [anomalous_processed_path, normal_processed_path]:
                os.makedirs(p, exist_ok=True)

            na_grades = [2] # TODO : change!
            a_grades = [4, 5]

            # Accumulate records for pkl (processed only)
            records = {
                'anomalous_processed': [],
                'normal_processed': [],
            }

            for i, item in enumerate(dataset):
                if ids is not None and grades_matched:
                    imgs_raw, imgs_processed, masks_raw, masks_processed, img_id, img_grades = item
                    
                    raw_img_folder = os.path.join(raw_path, img_id)
                    processed_img_folder = os.path.join(processed_path, img_id)
                    os.makedirs(raw_img_folder, exist_ok=True)
                    os.makedirs(processed_img_folder, exist_ok=True)

                    for j, (curr_img_raw, curr_img_processed) in enumerate(zip(imgs_raw, imgs_processed)):
                        grade = int(img_grades[j])
                        img_filename = f"{img_id}_obj{j}_grade{grade}.png"
                        
                        raw_img_path = os.path.join(raw_img_folder, img_filename)
                        img_pil_raw = Image.fromarray(curr_img_raw.astype(np.uint8))
                        img_pil_raw.save(raw_img_path)
                        
                        processed_img_path = os.path.join(processed_img_folder, img_filename)
                        img_pil_processed = Image.fromarray(curr_img_processed.astype(np.uint8))
                        img_pil_processed.save(processed_img_path)

                        # Copy to anomalous/normal (processed only) and collect records
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
                            'image': curr_img_processed,
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
                    imgs_raw, imgs_processed, masks_raw, masks_processed, img_id = item
                    
                    raw_img_folder = os.path.join(raw_path, img_id)
                    processed_img_folder = os.path.join(processed_path, img_id)
                    os.makedirs(raw_img_folder, exist_ok=True)
                    os.makedirs(processed_img_folder, exist_ok=True)
                    
                    for j, (img_raw, img_processed) in enumerate(zip(imgs_raw, imgs_processed)):
                        img_filename = f"{img_id}_obj{j}.png"
                        
                        raw_img_path = os.path.join(raw_img_folder, img_filename)
                        img_pil_raw = Image.fromarray(img_raw.astype(np.uint8))
                        img_pil_raw.save(raw_img_path)

                        processed_img_path = os.path.join(processed_img_folder, img_filename)
                        img_pil_processed = Image.fromarray(img_processed.astype(np.uint8))
                        img_pil_processed.save(processed_img_path)

                        # No grades — put in normal with grade -1 (processed only)
                        shutil.copy2(processed_img_path, os.path.join(normal_processed_path, img_filename))

                        records['normal_processed'].append({
                            'img_path': os.path.join(normal_processed_path, img_filename),
                            'grade': -1,
                            'mask': masks_processed[j],
                            'image': img_processed,
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
                    imgs_raw, imgs_processed, masks_raw, masks_processed = item
                    
                    raw_img_folder = os.path.join(raw_path, f"img_{i}")
                    processed_img_folder = os.path.join(processed_path, f"img_{i}")
                    os.makedirs(raw_img_folder, exist_ok=True)
                    os.makedirs(processed_img_folder, exist_ok=True)

                    for j, (img_raw, img_processed) in enumerate(zip(imgs_raw, imgs_processed)):
                        img_filename = f"img_{i}_obj{j}.png"
                        
                        raw_img_path = os.path.join(raw_img_folder, img_filename)
                        img_pil_raw = Image.fromarray(img_raw.astype(np.uint8))
                        img_pil_raw.save(raw_img_path)

                        processed_img_path = os.path.join(processed_img_folder, img_filename)
                        img_pil_processed = Image.fromarray(img_processed.astype(np.uint8))
                        img_pil_processed.save(processed_img_path)

                        # No grades, no ids — put in normal with grade -1 (processed only)
                        shutil.copy2(processed_img_path, os.path.join(normal_processed_path, img_filename))

                        records['normal_processed'].append({
                            'img_path': os.path.join(normal_processed_path, img_filename),
                            'grade': -1,
                            'mask': masks_processed[j],
                            'image': img_processed,
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

            # Save pkl files (processed only)
            with open(os.path.join(anomalous_processed_path, 'anomalous_samples.pkl'), 'wb') as f:
                pickle.dump(records['anomalous_processed'], f)
            with open(os.path.join(normal_processed_path, 'normal_samples.pkl'), 'wb') as f:
                pickle.dump(records['normal_processed'], f)

            print(f"[processed] Anomalous: {len(records['anomalous_processed'])}, Normal: {len(records['normal_processed'])}")

            # For the raw content, visualize the overlayed raspberries for each subfolder
            raw_folder = os.path.join(save_path, 'raw')
            for subfolder_name in os.listdir(raw_folder):
                subfolder_path = os.path.join(raw_folder, subfolder_name)
                if not os.path.isdir(subfolder_path):
                    continue
                output = os.path.join(subfolder_path, f'{subfolder_name}_overlayed_raspberries.png')
                overlay_raspberries(subfolder_path, output)

    return dataset



def data_split_non_anomalous(data_path_normal, data_path_anomalous, save_path):
    """
    Get the split of non anomalous samples for training and testing, such that we have a balanced split of normal and anomalous samples in the test set.

    Returns:
        train_normal_indices: list of indices for normal samples used in training
        test_normal_indices: list of indices for normal samples used in testing
    """
    random.seed(42)  # For reproducibility

    # Create splits folder if it does not exist
    splits_path = Path(save_path)
    splits_path.mkdir(parents=True, exist_ok=True)

    # Load data
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

    # Shuffle normal indices
    all_normal_indices = list(range(n_normal))
    random.shuffle(all_normal_indices)

    # Balanced test set
    test_normal_indices = all_normal_indices[:n_anomalous]

    # Remaining normals go to training
    train_normal_indices = all_normal_indices[n_anomalous:]


    # Save splits so we do not have randomness in each call
    save_path = Path(save_path)
    with open(save_path / 'train_normal_indices.pkl', 'wb') as f:
        pickle.dump(train_normal_indices, f)
    with open(save_path / 'test_normal_indices.pkl', 'wb') as f:
        pickle.dump(test_normal_indices, f)





'''
def create_dataset_imgs(masks, images, save_path=None, ids=None, all_gt_masks = None, all_gt_grades = None):
    """
    Given a list of lists of masks (True/False values) for each image, 
    create a dataset of single object images (RGB format).

    Args:
        masks: List of lists of boolean masks for each image (predicted by some model) ; these are the masks that are used to create the dataset.
        images: List of original images (PIL format) ; the original images from the original dataset.
        save_path: Optional path to save the single object images
        ids: Optional list of IDs corresponding to each image
        all_gt_masks: Optional list of lists of ground truth masks for each image
        all_gt_grades: Optional list of lists of ground truth grades for each image

    Notes :
        1. all_gt_masks, if provided, are used to match predicted masks to ground truth masks
              using Hungarian matching, and then assigning all_gt_grades, if provided, accordingly. This
              is important for our anomaly detection algorithm later, i.e. we need to know
              what objects have what grade (i.e. what level of anomaly).
    
    """

    grades_matched = []
    if all_gt_masks and all_gt_grades is not None:
        # Match predicted masks to ground truth masks for each image
        for pred_masks_img, gt_masks_img, gt_grades_img in zip(masks, all_gt_masks, all_gt_grades):
            curr_grades = np.full(len(pred_masks_img), -1, dtype=int) # -1 indicates no match (i.e predicted mask has no corresponding GT mask)
            matched_pairs, _, _ = match_instances(pred_masks_img, gt_masks_img, iou_threshold= 0.5, gt_grades=gt_grades_img)
            for pred_idx, _, _, grade in matched_pairs:
                curr_grades[pred_idx] = grade
            grades_matched.append(curr_grades)

        
        
        

    dataset = []
    for idx, (img, img_masks) in enumerate(zip(images, masks)):
        curr_img_raw = []
        curr_img_processed = []
        curr_masks_raw = []
        curr_masks_processed = []
        
        for mask in img_masks:
            # Create a new image for each mask
            masked_img = img.copy()
            masked_img = np.array(masked_img)
            masked_img[~mask] = 0  # Apply mask

            # Store raw version
            curr_img_raw.append(masked_img)
            curr_masks_raw.append(mask)

            # Center the object in the image
            masked_img_processed, mask_processed = _center_object(masked_img, mask)
            # Crop to square
            masked_img_processed, mask_processed = _crop_image(masked_img_processed, mask_processed)

            # Store processed version
            curr_img_processed.append(masked_img_processed)
            curr_masks_processed.append(mask_processed)

        if ids is not None and grades_matched:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, ids[idx], grades_matched[idx]))
        elif ids is not None:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, ids[idx]))
        else:
            dataset.extend(list(zip(curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed)))

    if save_path is not None:
        # Create raw and processed subdirectories
        raw_path = os.path.join(save_path, 'raw')
        processed_path = os.path.join(save_path, 'processed')
        os.makedirs(raw_path, exist_ok=True)
        os.makedirs(processed_path, exist_ok=True)
        
        for i, item in enumerate(dataset):
            if ids is not None and grades_matched:
                imgs_raw, imgs_processed, masks_raw, masks_processed, img_id, img_grades = item
                
                # Make folders for both raw and processed
                raw_img_folder = os.path.join(raw_path, img_id)
                processed_img_folder = os.path.join(processed_path, img_id)
                os.makedirs(raw_img_folder, exist_ok=True)
                os.makedirs(processed_img_folder, exist_ok=True)

                for j, (curr_img_raw, curr_img_processed) in enumerate(zip(imgs_raw, imgs_processed)):
                    img_filename = f"{img_id}_obj{j}_grade{img_grades[j]}.png"
                
                    
                    # Save raw version
                    img_pil_raw = Image.fromarray(curr_img_raw.astype(np.uint8))
                    img_pil_raw.save(os.path.join(raw_img_folder, img_filename))
           
                    
                    # Save processed version
                    img_pil_processed = Image.fromarray(curr_img_processed.astype(np.uint8))
                    img_pil_processed.save(os.path.join(processed_img_folder, img_filename))

                # Save raw and processed masks and images for all objects of an image 

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
                imgs_raw, imgs_processed, masks_raw, masks_processed, img_id = item
                
                # Make folders for both raw and processed
                raw_img_folder = os.path.join(raw_path, img_id)
                processed_img_folder = os.path.join(processed_path, img_id)
                os.makedirs(raw_img_folder, exist_ok=True)
                os.makedirs(processed_img_folder, exist_ok=True)
                
                for j, (img_raw, img_processed) in enumerate(zip(imgs_raw, imgs_processed)):
                    img_filename = f"{img_id}_obj{j}.png"
                    
                    # Save raw version
                    img_pil_raw = Image.fromarray(img_raw.astype(np.uint8))
                    img_pil_raw.save(os.path.join(raw_img_folder, img_filename))

                    # Save processed version
                    img_pil_processed = Image.fromarray(img_processed.astype(np.uint8))
                    img_pil_processed.save(os.path.join(processed_img_folder, img_filename))


                # Save raw and processed masks and images for all objects of an image 

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
                imgs_raw, imgs_processed, masks_raw, masks_processed = item
                
                # Make folders for both raw and processed
                raw_img_folder = os.path.join(raw_path, f"img_{i}")
                processed_img_folder = os.path.join(processed_path, f"img_{i}")
                os.makedirs(raw_img_folder, exist_ok=True)
                os.makedirs(processed_img_folder, exist_ok=True)

                for j, (img_raw, img_processed) in enumerate(zip(imgs_raw, imgs_processed)):
                    img_filename = f"img_{i}_obj{j}.png"
                    
                    # Save raw version
                    img_pil_raw = Image.fromarray(img_raw.astype(np.uint8))
                    img_pil_raw.save(os.path.join(raw_img_folder, img_filename))
                    # Save processed version
                    img_pil_processed = Image.fromarray(img_processed.astype(np.uint8))
                    img_pil_processed.save(os.path.join(processed_img_folder, img_filename))

                # Save raw and processed masks and images for all objects of an image 

                raw_data_dict = {
                    'images': np.array(imgs_raw, dtype=object),  
                    'masks': np.array(masks_raw, dtype=object),  
                }

                processed_data_dict = {
                    'images': np.array(imgs_processed, dtype=object), 
                    'masks': np.array(masks_processed, dtype=object),  
                }

                np.savez_compressed(os.path.join(raw_img_folder, f'raw_{img_id}_data.npz'), **raw_data_dict)
                np.savez_compressed(os.path.join(processed_img_folder, f'processed_{img_id}_data.npz'), **processed_data_dict)

    # For the raw content, visualize the overlayed raspberries for each subfolder
    raw_folder = os.path.join(save_path, 'raw')

    # Iterate over all subfolders in raw/
    for subfolder_name in os.listdir(raw_folder):
        subfolder_path = os.path.join(raw_folder, subfolder_name)
        
        # Skip if not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Create output filename based on subfolder name
        output = os.path.join(subfolder_path, f'{subfolder_name}_overlayed_raspberries.png')
        
        # Generate overlay for this subfolder
        overlay_raspberries(subfolder_path, output)


    return dataset
'''
def main():
    SAVE_PATH = '../../disk/dataset_single_objects/GT'
    PRED_MASKS_FILE = '../../disk/saved_masks/DINO_SAM_mobile/masks.pkl'

    
    ## Load original images and masks
    ds = load_dataset("FBK-TeV/RaspGrade")
    train_data = list(ds['train'])
    valid_data = list(ds['valid'])
    full_data = train_data + valid_data
    all_imgs_ids = [(sample['image'], sample['image_id']) for sample in full_data]

    # Extract masks from all training samples
    all_gt_masks_ids, all_gt_grades_ids = _extract_masks(full_data, extract_grades_bool=True)


    ## Load predicted masks
   # pred_data = np.load(PRED_MASKS_FILE)
    with open(PRED_MASKS_FILE, 'rb') as f:
        pred_data = pickle.load(f)


    all_pred_masks_ids = [(pred_data[key], key) for key in pred_data.keys()]

    # Sort both lists by image ID to ensure alignment
    all_imgs_ids.sort(key=lambda x: x[1])
    all_pred_masks_ids.sort(key=lambda x: x[1])
    all_gt_masks_ids.sort(key=lambda x: x[1])
    all_gt_grades_ids.sort(key=lambda x: x[1])

    # Since we have two different return formats... the xyn we needed for yolo model training
    if isinstance(all_pred_masks_ids[0][0], list) and len(all_pred_masks_ids[0][0]) == 4:
        all_pred_masks = [masks_and_xyn_and_conf_scores_and_imgs[0] for masks_and_xyn_and_conf_scores_and_imgs, img_id in all_pred_masks_ids]
        all_conf_scores = [masks_and_xyn_and_conf_scores_and_imgs[2] for masks_and_xyn_and_conf_scores_and_imgs, img_id in all_pred_masks_ids]
    else:
        all_pred_masks = [masks_and_conf_scores[0] for masks_and_conf_scores, img_id in all_pred_masks_ids]
        all_conf_scores = [masks_and_conf_scores[1] for masks_and_conf_scores, img_id in all_pred_masks_ids]

    # Remove sample idx from all
    all_imgs = [img for img, _ in all_imgs_ids]
   # all_pred_masks = [masks for masks, _ in all_pred_masks_ids]
    all_gt_masks = [masks for masks, _ in all_gt_masks_ids]
    all_gt_grades = [grades for grades, _ in all_gt_grades_ids]

    # If available, keep them for naming the files later
    all_ids = [img_id for _, img_id in all_pred_masks_ids]

    # NOTE : for testing ,only first 15 images
    #all_imgs = all_imgs[:15]
    #all_ids = all_ids[:15]
    #all_gt_masks = all_gt_masks[:15]
    #all_gt_grades = all_gt_grades[:15]

    # Create dataset of single object images
    dataset_single_objects = create_dataset_imgs(all_gt_masks, all_imgs, save_path=SAVE_PATH, ids=all_ids, all_gt_masks=all_gt_masks, all_gt_grades=all_gt_grades)
    data_split_non_anomalous(
        data_path_normal='../../disk/dataset_single_objects/GT/processed/normal/normal_samples.pkl',
        data_path_anomalous='../../disk/dataset_single_objects/GT/processed/anomalous/anomalous_samples.pkl',
        save_path='../../disk/dataset_single_objects/GT/processed/splits'
    )




if __name__ == "__main__":
    main()