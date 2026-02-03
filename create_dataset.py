## Create the dataset of single object masks from occluded objects

import numpy as np
from datasets import load_dataset
import os
from PIL import Image
from evaluation_segmentation import _extract_masks, match_instances, _hungarian_matching
from helper import overlay_raspberries
import matplotlib.pyplot as plt


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

def main():
    SAVE_PATH = 'dataset_single_objects/SAM_mobile'
    PRED_MASKS_FILE = 'saved_masks/SAM_mobile/masks.npz'

    ## Load original images and masks
    ds = load_dataset("FBK-TeV/RaspGrade")
    train_data = list(ds['train'])
    all_imgs_ids = [(sample['image'], sample['image_id']) for sample in train_data]

    # Extract masks from all training samples
    all_gt_masks_ids, all_gt_grades_ids = _extract_masks(train_data, extract_grades_bool=True)


    ## Load predicted masks
    pred_data = np.load(PRED_MASKS_FILE)
    all_pred_masks_ids = [(pred_data[key], key) for key in pred_data.keys()]

    # Sort both lists by image ID to ensure alignment
    all_imgs_ids.sort(key=lambda x: x[1])
    all_pred_masks_ids.sort(key=lambda x: x[1])
    all_gt_masks_ids.sort(key=lambda x: x[1])
    all_gt_grades_ids.sort(key=lambda x: x[1])

    # Remove sample idx from all
    all_imgs = [img for img, _ in all_imgs_ids]
    all_pred_masks = [masks for masks, _ in all_pred_masks_ids]
    all_gt_masks = [masks for masks, _ in all_gt_masks_ids]
    all_gt_grades = [grades for grades, _ in all_gt_grades_ids]

    # If available, keep them for naming the files later
    all_ids = [img_id for _, img_id in all_pred_masks_ids]

    # NOTE : for testing ,only first 15 images
    all_imgs = all_imgs[:15]
    all_ids = all_ids[:15]
    all_gt_masks = all_gt_masks[:15]
    all_gt_grades = all_gt_grades[:15]

    # Create dataset of single object images
    dataset_single_objects = create_dataset_imgs(all_pred_masks, all_imgs, save_path=SAVE_PATH, ids=all_ids, all_gt_masks=all_gt_masks, all_gt_grades=all_gt_grades)




if __name__ == "__main__":
    main()