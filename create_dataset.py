## Create the dataset of single object masks from occluded objects

import numpy as np
from datasets import load_dataset
import os
from PIL import Image
from evaluation_segmentation import _extract_masks, match_instances, _hungarian_matching
from helper import overlay_raspberries
import matplotlib.pyplot as plt
import pickle
from image_manipulation import edge_smoothing, normalize_distribution, find_holes_fix
import pickle
import random
import shutil
from pathlib import Path
from transformers import pipeline
import torch
import matplotlib.cm as cm
import cv2



'''
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

def _crop_to_bbox(img, mask, padding=10, square=True, pad_value=0, depth_image=None):
    """
    Crop image, mask, and optionally depth to the bounding box of the mask.
    
    Args:
        img: numpy array (H, W, 3)
        mask: boolean numpy array (H, W)
        padding: pixels to add around the bbox
        square: if True, pad to square before returning
        pad_value: value to fill padded regions (0 = black)
        depth_image: optional numpy array (H, W), depth map
    
    Returns:
        cropped_img, cropped_mask[, cropped_depth]
    """
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
    """
    Center the raspberry based on mask centroid and crop to fixed size.
    No resizing — just center and crop/pad to target_size x target_size.
    Need same sizes for Dataloader later with depth and filters ...
    
    Args:
        img: numpy array (H, W, 3)
        mask: boolean numpy array (H, W)
        target_size: output size (target_size x target_size)
        pad_value: value for padded regions
        depth_image: optional numpy array (H, W)
    
    Returns:
        cropped_img, cropped_mask[, cropped_depth]
    """
    h, w = img.shape[:2]
    
    # Find mask centroid
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
    
    # Source region in original image
    src_y1 = cy - half
    src_x1 = cx - half
    src_y2 = src_y1 + target_size
    src_x2 = src_x1 + target_size
    
    # Destination region in output (handles cases where source goes out of bounds)
    dst_y1 = max(0, -src_y1)
    dst_x1 = max(0, -src_x1)
    dst_y2 = target_size - max(0, src_y2 - h)
    dst_x2 = target_size - max(0, src_x2 - w)
    
    # Clamp source to image bounds
    src_y1 = max(0, src_y1)
    src_x1 = max(0, src_x1)
    src_y2 = min(h, src_y2)
    src_x2 = min(w, src_x2)
    
    # Create output arrays
    out_img = np.full((target_size, target_size, 3), pad_value, dtype=img.dtype)
    out_mask = np.zeros((target_size, target_size), dtype=mask.dtype)
    
    out_img[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    out_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    
    if depth_image is not None:
        out_depth = np.zeros((target_size, target_size), dtype=depth_image.dtype)
        out_depth[dst_y1:dst_y2, dst_x1:dst_x2] = depth_image[src_y1:src_y2, src_x1:src_x2]
        return out_img, out_mask, out_depth
    
    return out_img, out_mask


def create_dataset_imgs(masks, images, save_path=None, ids=None, all_gt_masks = None, all_gt_grades = None, filter_train_data_leak = False, size_filtering = False, darkness_filtering = False, darkness_threshold = 50, max_dark_ratio = 0.5, size_filtering_factor = 1.5, img_size = 256):
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
        filter_train_data_leak : If True, filter out samples from the created train set that are in the full (i.e. no filters, all data) test set. This is in order to evaluate
                    on one and the same, full, test set, if wanted.
        size_filtering : If True, filter out objects that are too small based on size_filtering_factor (e.g. 1.5 means remove objects smaller than median - 1.5 * MAD)
        size_filtering_factor : Factor for size filtering (e.g. 1.5 means remove objects smaller than median - 1.5 * MAD)

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


    # Load depth model 
    pipe = pipeline(
        task="depth-estimation", 
        model="depth-anything/Depth-Anything-V2-Base-hf", 
        device='cuda:2' if torch.cuda.is_available() else 'cpu'
    )


    # Get depths for all single raspberries 
    depth_masks = [[] for _ in range(len(images))]  # per-image, per-mask depth crops
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


    if size_filtering:
        removed_dir_size = "../../disk/removed_size_raspberries"
        if os.path.exists(removed_dir_size):
            shutil.rmtree(removed_dir_size)
        os.makedirs(removed_dir_size, exist_ok=True)

    if darkness_filtering:
        removed_dir_darkness = "../../disk/removed_dark_raspberries"
        if os.path.exists(removed_dir_darkness):
            shutil.rmtree(removed_dir_darkness)
        os.makedirs(removed_dir_darkness, exist_ok=True)


    if size_filtering or darkness_filtering:
        filtered_path = os.path.join(save_path, 'filtered')
        os.makedirs(filtered_path, exist_ok = True)
   #     full_no_filters_path = f'../../disk/dataset_single_objects/GT/full_no_filters_{img_size}'
     #   assert os.path.exists(full_no_filters_path), "Need the unfiltered version to cleanly\
    #    remove filtered objects from the original (i.e. full_no_filters) test set!"

        # Open data files
      #  anomalous_test_samples = Path(full_no_filters_path) / 'processed' / 'anomalous' / 'anomalous_samples.pkl'
      #  nanomalous_test_samples = Path(full_n)
    


    # TODO : need to fix the indexing ... because when we remove stuff here first in size filtering, then indexing will be different for darkness filtering and the img paths will not 
    # TODO : be correct anymore...
    # TODO : + already now the data leak works based on img paths and when we filter before, it will mess things up ! we need constant indexing 
    # TODO : it should be like this : Even when we filter, keep the original img indices and object indices and name the saved img paths with always the original img idx and object idx 
    # TODO : such that matching later on (i.e. a filtered version to a non-filtered one) stays trivial
    if size_filtering:



        removed_count = 0

        
        # Filter individual raspberries that are both too small AND too deep (background noise)
        filtered_masks = []
        filtered_images = []
        filtered_mean_depth_masks = []
        filtered_depth_masks = []
        filtered_grades = []
        filtered_img_paths = []

        for img_idx in range(len(masks)):
            img_masks = masks[img_idx]
            img_depths = mean_depth_masks[img_idx]
            img = images[img_idx]


            sizes = np.array([mask.sum() for mask in img_masks])
            depths = np.array(img_depths)

            median_size = np.median(sizes)
            mad_size = np.median(np.abs(sizes - median_size))
            size_lower_bound = median_size - size_filtering_factor * mad_size

            median_depth = np.median(depths)
            mad_depth = np.median(np.abs(depths - median_depth))
            depth_upper_bound = median_depth + 1.5 * mad_depth

            keep_indices = []
            for i in range(len(img_masks)):
                too_small = sizes[i] < size_lower_bound
                too_deep = depths[i] > depth_upper_bound
                if too_small:
                    # Save removed raspberry
                    mask = img_masks[i]
                    img_array = np.array(img)  
                    cropped = img_array.copy()
                    cropped[mask == 0] = 0

                    # Crop to bounding box of the mask
                    ys, xs = np.where(mask > 0)
                    cropped = cropped[ys.min():ys.max()+1, xs.min():xs.max()+1]

                    Image.fromarray(cropped).save(
                        os.path.join(removed_dir_size, f"img{img_idx:03d}_obj{i:02d}_grade{grades_matched[img_idx][i]}.png")
                    )
                    filtered_img_paths.append(f"img{img_idx:03d}_obj{i:02d}_grade{grades_matched[img_idx][i]}.png")
                    removed_count += 1
                else:
                    keep_indices.append(i)

            filtered_masks.append([img_masks[i] for i in keep_indices])
            filtered_images.append(img)
            filtered_mean_depth_masks.append([img_depths[i] for i in keep_indices])
            filtered_depth_masks.append([depth_masks[img_idx][i] for i in keep_indices])
            filtered_grades.append(grades_matched[img_idx][keep_indices] if grades_matched else None)
        print(f"Removed {removed_count} raspberries, saved to {removed_dir_size}")

        masks = filtered_masks
        images = filtered_images
        mean_depth_masks = filtered_mean_depth_masks
        depth_masks = filtered_depth_masks
        grades_matched = filtered_grades

        # Store filtered_img_paths

        with open(os.path.join(filtered_path, 'filtered_samples_path.pkl'), 'wb') as f:
            pickle.dump(filtered_img_paths)

        


    if darkness_filtering:

        removed_count = 0

        filtered_masks = []
        filtered_images = []
        filtered_mean_depth_masks = []
        filtered_depth_masks = []
        filtered_grades = []

        for img_idx in range(len(masks)):
            img_masks = masks[img_idx]
            img_depths = mean_depth_masks[img_idx]
            img = images[img_idx]
            img_array = np.array(img)

            # Compute darkness ratio for each raspberry
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            brightness = hsv[:, :, 2].astype(np.float32)

            keep_indices = []
            for i in range(len(img_masks)):
                mask = img_masks[i]
                masked_brightness = brightness[mask > 0]
                dark_ratio = (masked_brightness < darkness_threshold).sum() / masked_brightness.size

                if dark_ratio > max_dark_ratio:
                    # Save removed raspberry
                    cropped = img_array.copy()
                    cropped[mask == 0] = 0
                    ys, xs = np.where(mask > 0)
                    cropped = cropped[ys.min():ys.max()+1, xs.min():xs.max()+1]
                    Image.fromarray(cropped).save(
                        os.path.join(removed_dir_darkness, f"img{img_idx:03d}_obj{i:02d}_grade{grades_matched[img_idx][i]}.png")
                    )
                    removed_count += 1
                else:
                    keep_indices.append(i)

            filtered_masks.append([img_masks[i] for i in keep_indices])
            filtered_images.append(img)
            filtered_mean_depth_masks.append([img_depths[i] for i in keep_indices])
            filtered_depth_masks.append([depth_masks[img_idx][i] for i in keep_indices])
            filtered_grades.append(grades_matched[img_idx][keep_indices] if grades_matched else None)

        print(f"Removed {removed_count} dark raspberries, saved to {removed_dir_darkness}")

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
        curr_depth_raw = []
        curr_depth_processed = []

        
        for j,mask in enumerate(img_masks):
            # Create a new image for each mask
            masked_img = img.copy()
            masked_img = np.array(masked_img)
            masked_img[~mask] = 0  # Apply mask

            # Store raw version
            curr_img_raw.append(masked_img)
            curr_masks_raw.append(mask)
            curr_depth_raw.append(depth_masks[idx][j])

            # Center the object in the image
          #  masked_img_processed, mask_processed = _center_object(masked_img, mask)
            # Crop to square
           # masked_img_processed, mask_processed = _crop_image(masked_img_processed, mask_processed)

         #   masked_img_processed, mask_processed, depth_processed = _crop_to_bbox(masked_img, mask, padding=8, square=True, pad_value=0, depth_image = depth_masks[idx][j])
            masked_img_processed, mask_processed, depth_processed = _center_and_crop(masked_img, mask, target_size=img_size, pad_value=0, depth_image = depth_masks[idx][j])

          #  masked_img_processed, mask_processed, depth_processed = find_holes_fix(masked_img_processed, mask_processed, depth_processed, small_hole_max_area = 100, save_folder = None, filename = None, visualize_bool=False) 


          #  masked_img_processed, mask_processed = normalize_distribution(masked_img_processed, mask_processed, target_mean = 134.08394672413232, target_std = 31.834956813835767)
          #  masked_img_processed, mask_processed = edge_smoothing(masked_img_processed, mask_processed)

            # Store processed version
            curr_img_processed.append(masked_img_processed)
            curr_masks_processed.append(mask_processed)
            curr_depth_processed.append(depth_processed)

        if ids is not None and grades_matched:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, curr_depth_raw, curr_depth_processed, ids[idx], grades_matched[idx], ))
        elif ids is not None:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, curr_depth_raw, curr_depth_processed, ids[idx]))
        else:
            dataset.extend(list(zip(curr_img_raw, curr_img_processed, curr_masks_raw, curr_masks_processed, curr_depth_raw, curr_depth_processed)))

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

       
            na_grades = [1,2,3]

            a_grades = [4, 5]

            # Accumulate records for pkl (processed only)
            records = {
                'anomalous_processed': [],
                'normal_processed': [],
            }

            # Get the test set paths of non-anomalous samples 
            if filter_train_data_leak:
                test_set_path = f'../../disk/dataset_single_objects/GT/full_no_filters_{img_size}/processed/splits/test_normal_paths.pkl'
                assert os.path.exists(test_set_path), f"Full Test set paths file not found at {test_set_path}! Needed to have the same test set across all experiments and no data leakage."
                with open(test_set_path, 'rb') as f:
                    test_normal_paths = pickle.load(f)
            
                # Extract only the base name
                test_normal_paths = [os.path.basename(path) for path in test_normal_paths]

            for i, item in enumerate(dataset):
                if ids is not None and grades_matched:
                    imgs_raw, imgs_processed, masks_raw, masks_processed, depths_raw, depths_processed, img_id, img_grades = item
                    
                    raw_img_folder = os.path.join(raw_path, img_id)
                    processed_img_folder = os.path.join(processed_path, img_id)
                    os.makedirs(raw_img_folder, exist_ok=True)
                    os.makedirs(processed_img_folder, exist_ok=True)

                    for j, (curr_img_raw, curr_img_processed, curr_depth_raw, curr_depth_processed) in enumerate(zip(imgs_raw, imgs_processed, depths_raw, depths_processed)):
                        grade = int(img_grades[j])
                        img_filename = f"{img_id}_obj{j}_grade{grade}.png"

                        if filter_train_data_leak:
                            if img_filename in test_normal_paths:
                                print(f"Skipping {img_filename} as it is in the test set of non-anomalous samples to avoid data leakage.")
                                continue
                        
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
                            'depth': curr_depth_processed
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

                # TODO : these probably need some updating ...
                elif ids is not None: 
                    imgs_raw, imgs_processed, masks_raw, masks_processed, depths_raw, depths_processed, img_id = item
                    
                    raw_img_folder = os.path.join(raw_path, img_id)
                    processed_img_folder = os.path.join(processed_path, img_id)
                    os.makedirs(raw_img_folder, exist_ok=True)
                    os.makedirs(processed_img_folder, exist_ok=True)
                    
                    for j, (img_raw, img_processed, depth_raw, depth_processed) in enumerate(zip(imgs_raw, imgs_processed, depths_raw, depths_processed)):
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
                            'depth': curr_depth_processed
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
                    imgs_raw, imgs_processed, masks_raw, masks_processed, depths_raw, depths_processed = item
                    
                    raw_img_folder = os.path.join(raw_path, f"img_{i}")
                    processed_img_folder = os.path.join(processed_path, f"img_{i}")
                    os.makedirs(raw_img_folder, exist_ok=True)
                    os.makedirs(processed_img_folder, exist_ok=True)

                    for j, (img_raw, img_processed, depth_raw, depth_processed) in enumerate(zip(imgs_raw, imgs_processed, depths_raw, depths_processed)):
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
                            'depth': depths_processed
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

    # Save splits with img_paths
    train_normal_paths = [normal_data[i]['img_path'] for i in train_normal_indices]
    test_normal_paths = [normal_data[i]['img_path'] for i in test_normal_indices]

    # Save splits so we do not have randomness in each call
    save_path = Path(save_path)
    with open(save_path / 'train_normal_indices.pkl', 'wb') as f:
        pickle.dump(train_normal_indices, f)
    with open(save_path / 'test_normal_indices.pkl', 'wb') as f:
        pickle.dump(test_normal_indices, f)
    with open(save_path / 'train_normal_paths.pkl', 'wb') as f:
        pickle.dump(train_normal_paths, f)
    with open(save_path / 'test_normal_paths.pkl', 'wb') as f:
        pickle.dump(test_normal_paths, f)





def main():

    SIZE_FILTERING = False
    SIZE_FILTERING_FACTOR = 1.5
    DARKNESS_FILTERING = False
    DARKNESS_THRESHOLD = 80
    MAX_DARK_RATIO = 0.3
    IMG_SIZE = 256
    UNBLURRED = True

    filter_parts = []
    if SIZE_FILTERING:
        filter_parts.append(f"size_{SIZE_FILTERING_FACTOR}")
    if DARKNESS_FILTERING:
        filter_parts.append(f"darkness_{DARKNESS_THRESHOLD}_{MAX_DARK_RATIO}")
    if UNBLURRED:
        filter_parts.append("unblurred")

    if filter_parts:
        filter_str = "filtered_" + "_and_".join(filter_parts)
    else:
        filter_str = "full_no_filters"

    SAVE_PATH = f'../../disk/dataset_single_objects/GT/{filter_str}_{IMG_SIZE}'

    PRED_MASKS_FILE = '../../disk/saved_masks/DINO_SAM_mobile/masks.pkl'

    
    ## Load original images and masks
    ds = load_dataset("FBK-TeV/RaspGrade")
    train_data = list(ds['train'])
    valid_data = list(ds['valid'])

    if UNBLURRED:
        # Replace original images with unblurred variants (keep the same image IDs and masks, just different images)
        unblurred_dir = '../../disk/dataset_bonnets/raspberries_unblurred'

        for i, sample in enumerate(train_data):
            sample['image'] = Image.open(f'{unblurred_dir}/{i}.jpg').convert('RGB')

        for j, sample in enumerate(valid_data):
            sample['image'] = Image.open(f'{unblurred_dir}/{len(train_data) + j}.jpg').convert('RGB')

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


    # Create dataset of single object images
    dataset_single_objects = create_dataset_imgs(all_gt_masks, all_imgs, save_path=SAVE_PATH, ids=all_ids, all_gt_masks=all_gt_masks, all_gt_grades=all_gt_grades,filter_train_data_leak = False, size_filtering=SIZE_FILTERING, darkness_filtering=DARKNESS_FILTERING, darkness_threshold=DARKNESS_THRESHOLD, max_dark_ratio=MAX_DARK_RATIO, size_filtering_factor=SIZE_FILTERING_FACTOR, img_size = IMG_SIZE)
    data_split_non_anomalous(
        data_path_normal=f'{SAVE_PATH}/processed/normal/normal_samples.pkl',
        data_path_anomalous=f'{SAVE_PATH}/processed/anomalous/anomalous_samples.pkl',
        save_path=f'{SAVE_PATH}/processed/splits'
    )
'''



"""
Refactored dataset creation pipeline.

ARCHITECTURE CHANGE SUMMARY:
============================
OLD: create_dataset_imgs() did creation + size filtering + darkness filtering + data leak filtering
     all in one function, mutating lists in-place and breaking index correspondence.

NEW: 
  1. create_dataset_imgs() ONLY creates the full unfiltered dataset. No filtering flags.
     Every raspberry is saved with its ORIGINAL img_idx and obj_idx in the filename.
  2. data_split_non_anomalous() does the train/test split on the FULL set, deterministically.
     This is the single source of truth. Always the same split regardless of filters.
  3. apply_filters() is a NEW post-processing function that takes the already-created,
     already-split dataset and produces filtered versions. Original data is never modified.
     The SingleRaspberryDataset class loads filtered or unfiltered versions.

WHAT THIS FIXES:
- Size filtering no longer shifts indices, so darkness filtering sees correct img_idx values
- Filenames are stable across filtered/unfiltered experiments → matching is trivial
- filter_train_data_leak flag is gone — the split is always the same, filters only prune from it
- The buggy `records['anomalous_processed']` reference (used before `records` existed) is gone
"""



# ==============================================================================
# UNCHANGED helper functions — _center_object, _crop_image, _crop_to_bbox, _center_and_crop
# ==============================================================================

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


# ==============================================================================
# CHANGED: create_dataset_imgs — all filtering logic removed
# ==============================================================================
# 
# What was removed:
#   - size_filtering, darkness_filtering parameters and all their logic
#   - filter_train_data_leak parameter and its logic
#   - size_filtering_factor, darkness_threshold, max_dark_ratio parameters
#   - The buggy `pickle.dump(records['anomalous_processed'], f)` that referenced
#     `records` before it was defined
#   - In-place mutation of masks/images/grades lists during filtering
#
# What stays the same:
#   - Depth estimation with Depth-Anything-V2
#   - Hungarian matching of predicted masks to GT masks for grade assignment
#   - _center_and_crop processing
#   - Saving raw + processed images, npz files, pkl records
#   - Overlay visualization for raw subfolders
#
# Key guarantee: every raspberry is saved with its ORIGINAL image index and 
# object index in the filename, so these are stable identifiers across experiments.
# ==============================================================================

def create_dataset_imgs(masks, images, save_path=None, ids=None, 
                        all_gt_masks=None, all_gt_grades=None, img_size=256):
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
        curr_depth_raw = []
        curr_depth_processed = []

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

            curr_img_processed.append(masked_img_processed)
            curr_masks_processed.append(mask_processed)
            curr_depth_processed.append(depth_processed)

        if ids is not None and grades_matched:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw, 
                          curr_masks_processed, curr_depth_raw, curr_depth_processed,
                          ids[idx], grades_matched[idx]))
        elif ids is not None:
            dataset.append((curr_img_raw, curr_img_processed, curr_masks_raw,
                          curr_masks_processed, curr_depth_raw, curr_depth_processed,
                          ids[idx]))
        else:
            dataset.extend(list(zip(curr_img_raw, curr_img_processed, curr_masks_raw,
                                   curr_masks_processed, curr_depth_raw, curr_depth_processed)))

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
                imgs_raw, imgs_processed, masks_raw, masks_processed, depths_raw, depths_processed, img_id, img_grades = item
                
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

                    # CHANGE: No filter_train_data_leak check here anymore
                    
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
                        'image': curr_img_processed,
                        'depth': curr_depth_processed,
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
                imgs_raw, imgs_processed, masks_raw, masks_processed, depths_raw, depths_processed, img_id = item
                
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
                        'image': img_processed,
                        'depth': depth_processed,  # BUGFIX: was `curr_depth_processed` (undefined in this branch)
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
                imgs_raw, imgs_processed, masks_raw, masks_processed, depths_raw, depths_processed = item
                
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
                        'image': img_processed,
                        'depth': depth_processed,  # BUGFIX: was `depths_processed` (wrong variable name)
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


# ==============================================================================
# UNCHANGED: data_split_non_anomalous
# ==============================================================================
# This was already correct. It splits once, deterministically, on the FULL
# unfiltered dataset. This split is the single source of truth.
# ==============================================================================

def data_split_non_anomalous(data_path_normal, data_path_anomalous, save_path):
    """
    Split non-anomalous samples into train/test, balanced against anomalous count.
    Deterministic (seed=42). Saves indices AND paths.
    """
    random.seed(42)

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


# ==============================================================================
# NEW: apply_filters — post-hoc IN-PLACE filtering on an already-created, already-split dataset
# ==============================================================================
#
# Design decisions:
#   - Works on the pkl records (normal_samples.pkl, anomalous_samples.pkl) and 
#     the split index files. Does NOT re-run dataset creation.
#   - OVERWRITES the original pkl and split files in-place, so downstream code
#     (SingleRaspberryDataset) works identically without any changes.
#   - Saves filtered-out images to {parent}/filtered/{size,darkness}/ for inspection.
#   - DELETES filtered-out .png files from processed/normal/ and processed/anomalous/.
#   - Recomputes integer indices into the now-shorter normal pkl.
#   - Consequence: after this runs, the unfiltered state is gone. Re-run 
#     create_dataset_imgs + data_split_non_anomalous to restore it.
# ==============================================================================

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


# ==============================================================================
# CHANGED: main() — simplified, no filtering flags passed to create_dataset_imgs
# ==============================================================================

def main():
    # --- Config ---
    IMG_SIZE = 256
    UNBLURRED = False
 
    # CHANGE: These are now only used by apply_filters(), not by create_dataset_imgs()
    SIZE_FILTERING = True
    SIZE_FILTERING_FACTOR = 1.5
    DARKNESS_FILTERING = False
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
 
    if filter_parts:
        filter_str = "filtered_" + "_and_".join(filter_parts)
    else:
        filter_str = "full_no_filters"
 
    SAVE_PATH = f'../../disk/dataset_single_objects/GT/{filter_str}_{IMG_SIZE}'
    PRED_MASKS_FILE = '../../disk/saved_masks/DINO_SAM_mobile/masks.pkl'

    # --- Load data (unchanged) ---
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
    # CHANGE: No filtering params passed. This is always the complete dataset.
    dataset_single_objects = create_dataset_imgs(
        all_gt_masks, all_imgs, save_path=SAVE_PATH, ids=all_ids,
        all_gt_masks=all_gt_masks, all_gt_grades=all_gt_grades,
        img_size=IMG_SIZE
    )

    # --- Step 2: Create deterministic split (unchanged logic) ---
    data_split_non_anomalous(
        data_path_normal=f'{SAVE_PATH}/processed/normal/normal_samples.pkl',
        data_path_anomalous=f'{SAVE_PATH}/processed/anomalous/anomalous_samples.pkl',
        save_path=f'{SAVE_PATH}/processed/splits'
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