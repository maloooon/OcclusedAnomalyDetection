from datasets import load_dataset
from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM3SemanticPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2



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

def model_SAM3(samples, save_imgs_bool=False, store_masks_bool=False):
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

    # Let us only look at the first 15 samples for testing
    samples = samples[:15]

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
            UPPER_MULTIPLIER = 4.0

            valid_idx = (
                (areas <= median_area * UPPER_MULTIPLIER)
            )
            
            print(f"Median area: {median_area:.0f} pixels")
            #   print(f"Filtering range: {median_area * LOWER_MULTIPLIER:.0f} - {median_area * UPPER_MULTIPLIER:.0f} pixels")
            #   print(f"Filtered: {len(boxes)} -> {valid_idx.sum()} detections")
            
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


def store_masks(masks, image_ids, filepath):
    save_data = {i: masks for i, masks in zip(image_ids, masks)}
    np.savez_compressed(f'{filepath}.npz', **save_data)
    print(f"Saved {len(masks)} mask sets to {filepath}.npz")


def main():

    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

    example_sample = ds['train'][0]
    bonnet_bbox = extract_bonnet_bounding_box(example_sample, visualize_bool=False)

    example_img = ds['train'][1]['image']


    ds_train_lst = list(ds['train'])

    # Crop to bonnet area
  #  example_img = example_img.crop((bonnet_bbox[0], bonnet_bbox[1], bonnet_bbox[2], bonnet_bbox[3])) # left, upper, right, lower

    MODE = "SAM3"

    if MODE == "SAM_manipulate":
        
        sam = sam_model_registry["vit_b"](checkpoint="pretrained_models/sam_b.pt")
        
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=8, # This one is important to reduce number of unneccessary masks ; smaller number less masks
            pred_iou_thresh=0.25,
            stability_score_thresh=0.92,
            box_nms_thresh=0.5,      
            crop_nms_thresh=0.5,     
            min_mask_region_area=100,
        )
        
        masks = mask_generator.generate(np.array(example_img))

        # Visualize and save
        plt.figure(figsize=(12, 8))
        plt.imshow(example_img)
        
        # Draw all masks and bounding boxes
        for mask_dict in masks:
            mask = mask_dict['segmentation']
            bbox = mask_dict['bbox']  # Format: [x, y, width, height]


            # Draw mask
            color = np.random.random(3)
            colored_mask = np.zeros((*mask.shape, 4))
            colored_mask[mask] = [*color, 0.6]  # RGBA with alpha=0.6
            plt.imshow(colored_mask)
            
            # Draw bounding box
            x, y, w, h = bbox
            rect = plt.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            plt.gca().add_patch(rect)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('output.jpg', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Saved output.jpg with {len(masks)} segments")
        
        return masks

    elif MODE == "SAM":

        # Load SAM model
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="pretrained_models/sam_b.pt") # We can set a higher conf score, but this would then be different from img to img... not that useful
        predictor = SAMPredictor(overrides=overrides)

        # Run inference
        results = predictor(source= example_img) 
    
    elif MODE == "SAM3":
        model_SAM3(list(ds['train']), save_imgs_bool = False, store_masks_bool = True)




    # Save the output
  #  results[0].save('output.jpg')





if __name__ == "__main__":
    main()