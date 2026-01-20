from datasets import load_dataset
from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM3SemanticPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import os



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

def model_SAM3(samples, save_imgs_bool=False, store_masks_bool=False, testing_samples=1):
    """
    SAM3 segmentation model inference and mask saving.   
    
  
    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process (i.e. processing takes a long time, so for testing we can limit this).
    
    
    """


    # TODO : lower conf score detects more, but sometimes we then have overlaps of masks on the SAME raspberry. 
    # TODO : possibly implement a filter that chooses the mask with higher confidence in case of overlap?
    # TODO : problem is that we also have overlaps of correct masks (e.g. raspberry in front of another raspberry)
    # TODO : do not know how to distinguish those cases yet... 


    if type(samples) is not list:
        samples = [samples]


    overrides = dict(
        conf=0.70,
        task="segment",
        mode="predict",
        model="pretrained_models/sam3.pt",
        half=True, 
        save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides) 

    masks_list = []
    img_ids_list = []

    # Let us only look at the first k samples for testing
    samples = samples[:testing_samples]

    for sample in samples:

        sample_img = sample['image']
        sample_idx = sample['image_id']
        predictor.set_image(sample_img)
        results = predictor(text=["raspberry"])
        
        # Calculate median area and filter outliers
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights

            # Save a plot of the distribution of areas
            if save_imgs_bool:
                plt.figure()
                plt.hist(areas, bins=30, color='blue', alpha=0.7)
                plt.axvline(np.median(areas), color='red', linestyle='dashed', linewidth=1)
                plt.title('Distribution of Detected Object Areas')
                plt.xlabel('Area (pixels)')
                plt.ylabel('Frequency')
                plt.savefig('area_distribution.png')
                plt.close()
            
            # Calculate median and filter outliers
            median_area = np.median(areas)
            
            # Keep objects within a range of the median
            # Adjust multiplier as needed (2.0 = keep areas up to 2x median)
            # NOTE : no lower multiplier, as we else might remove small raspberries (i.e. strongly occluded ones)
            UPPER_MULTIPLIER = 4.0

            valid_idx = (
                (areas <= median_area * UPPER_MULTIPLIER)
            )
            
            print(f"Median area: {median_area:.0f} pixels")
            #   print(f"Filtering range: {median_area * LOWER_MULTIPLIER:.0f} - {median_area * UPPER_MULTIPLIER:.0f} pixels")
            print(f"Filtered: {len(boxes)} -> {valid_idx.sum()} detections")
            
            results[0].boxes = results[0].boxes[valid_idx]
            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]
        
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
            cv2.imwrite('output.jpg', plotted_img)
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        # Save masks
        masks_list.append(results[0].masks.data.cpu().numpy())
        # Save image IDs
        img_ids_list.append(sample_idx)
    
    if store_masks_bool:
        store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM3')

def model_SAM(samples, save_imgs_bool = False, store_masks_bool = False, testing_samples = 1):
    """
    
    SAM segmentation model inference and mask saving.

    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process (i.e. processing takes a long time, so for testing we can limit this).
    
    """

    if type(samples) is not list:
        samples = [samples]

    # Load SAM model
    # We can set a higher conf score, possibly try finding some optimal one across multiple images...
    overrides = dict(conf=0.70, 
                     task="segment", 
                     mode="predict", 
                     imgsz=1024, 
                     model="pretrained_models/sam_b.pt",
                     save = False) 
    predictor = SAMPredictor(overrides=overrides)

    masks_list = []
    img_ids_list = []

    # Let us only look at the first k samples for testing
    samples = samples[:testing_samples]

    for sample in samples:

        sample_img = sample['image']
        sample_idx = sample['image_id']
        predictor.set_image(sample_img)
        results = predictor()
        
        # Calculate median area and filter outliers
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights

            # Save a plot of the distribution of areas
            if save_imgs_bool:
                plt.figure()
                plt.hist(areas, bins=30, color='blue', alpha=0.7)
                plt.axvline(np.median(areas), color='red', linestyle='dashed', linewidth=1)
                plt.title('Distribution of Detected Object Areas')
                plt.xlabel('Area (pixels)')
                plt.ylabel('Frequency')
                plt.savefig('area_distribution.png')
                plt.close()
            
            # Calculate median and filter outliers
            median_area = np.median(areas)
            
            # Keep objects within a range of the median
            # Adjust multiplier as needed (2.0 = keep areas up to 2x median)
            UPPER_MULTIPLIER = 4.0

            valid_idx = (
                (areas <= median_area * UPPER_MULTIPLIER)
            )
            
            print(f"Median area: {median_area:.0f} pixels")
            #   print(f"Filtering range: {median_area * LOWER_MULTIPLIER:.0f} - {median_area * UPPER_MULTIPLIER:.0f} pixels")
            print(f"Filtered: {len(boxes)} -> {valid_idx.sum()} detections")
            
            results[0].boxes = results[0].boxes[valid_idx]
            if results[0].masks is not None:
                results[0].masks = results[0].masks[valid_idx]
        
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
        store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM') 

def model_SAM_manipulate(samples, save_imgs_bool=False, store_masks_bool=False, testing_samples=1):
    """
    SAM segmentation model using Meta's implementation (not Ultralytics).
    Provides more flexibility in hyperparameters.
    
    Args:
        samples (list or dict): List of samples or a single sample dictionary containing 'image' and 'image_id'.
        save_imgs_bool (bool): Whether to save output images with masks overlaid.
        store_masks_bool (bool): Whether to store the generated masks to disk.
        testing_samples (int): Number of samples to process.
    """
    
    if type(samples) is not list:
        samples = [samples]
    
    # Load SAM model (Meta implementation)
    sam = sam_model_registry["vit_b"](checkpoint="pretrained_models/sam_b.pt")
    
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.25,
        stability_score_thresh=0.92,
        box_nms_thresh=0.5,
        crop_nms_thresh=0.5,
        min_mask_region_area=100,
    )
    
    masks_list = []
    img_ids_list = []
    
    # Let us only look at the first k samples for testing
    samples = samples[:testing_samples]
    
    for sample in samples:
        sample_img = sample['image']
        sample_idx = sample['image_id']
        
        # Generate masks (Meta SAM returns list of dicts)
        mask_dicts = mask_generator.generate(np.array(sample_img))
        
        # Extract masks and bounding boxes for filtering
        if len(mask_dicts) > 0:
            # Extract bounding boxes and calculate areas
            bboxes = np.array([m['bbox'] for m in mask_dicts])  # [x, y, w, h]
            areas = bboxes[:, 2] * bboxes[:, 3]  # width * height
            
            # Save area distribution plot
            if save_imgs_bool:
                plt.figure()
                plt.hist(areas, bins=30, color='blue', alpha=0.7)
                plt.axvline(np.median(areas), color='red', linestyle='dashed', linewidth=1)
                plt.title('Distribution of Detected Object Areas')
                plt.xlabel('Area (pixels)')
                plt.ylabel('Frequency')
                plt.savefig(f'area_distribution.png')
                plt.close()
            
            # Calculate median and filter outliers
            median_area = np.median(areas)
            UPPER_MULTIPLIER = 4.0
            
            valid_idx = (areas <= median_area * UPPER_MULTIPLIER)
            
            print(f"Image {sample_idx} - Median area: {median_area:.0f} pixels")
            print(f"Filtered: {len(mask_dicts)} -> {valid_idx.sum()} detections")
            
            # Filter mask_dicts
            mask_dicts = [m for i, m in enumerate(mask_dicts) if valid_idx[i]]
        
        # Extract binary masks from the filtered dictionaries
        binary_masks = np.array([m['segmentation'] for m in mask_dicts])
        
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
    





def store_masks(masks, image_ids, filepath):
 
    os.makedirs(filepath, exist_ok=True)

    output_file = os.path.join(filepath, 'masks.npz')
    
    save_data = {i: masks for i, masks in zip(image_ids, masks)}
    np.savez_compressed(output_file, **save_data)
    print(f"Saved {len(masks)} mask sets to {output_file}")


def main():

    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

   # example_sample = ds['train'][0]
   # bonnet_bbox = extract_bonnet_bounding_box(example_sample, visualize_bool=False)
   # example_img = ds['train'][1]['image']
    # Crop to bonnet area
  #  example_img = example_img.crop((bonnet_bbox[0], bonnet_bbox[1], bonnet_bbox[2], bonnet_bbox[3])) # left, upper, right, lower

    MODE = "SAM_manipulate"

    if MODE == "SAM_manipulate":
        model_SAM_manipulate(list(ds['train']), save_imgs_bool = True, store_masks_bool = True, testing_samples = 1)
        
    elif MODE == "SAM":
        model_SAM(list(ds['train']), save_imgs_bool = True, store_masks_bool = True, testing_samples = 1)

    elif MODE == "SAM3":
        model_SAM3(list(ds['train']), save_imgs_bool = False, store_masks_bool = True)




if __name__ == "__main__":
    main()