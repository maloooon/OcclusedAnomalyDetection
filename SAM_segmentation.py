from datasets import load_dataset
from ultralytics import SAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.models.sam import Predictor as SAMPredictor
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics.models.fastsam import FastSAMPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics.models.sam.amg import build_point_grid
import cv2
import os
from time import time 


import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import requests
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


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

def model_SAM3(samples, save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes = (True, 0.2, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None)):
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
        conf=0.75, # NOTE : try with different values and evaluate results on e.g. 15 samples only 
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
        start_time_0 = time()
        predictor.set_image(sample_img)
        end_time_0 = time()
        print(f"Image encoding time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        start_time = time()
        results = predictor(text=["raspberry"])
        end_time = time()
        print(f"Inference (prompt encoding and mask decoding) time for sample {sample_idx}: {end_time - start_time:.2f} seconds")

        # Calculate median area and filter outliers
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0 and filter_bboxes[0]:
            valid_idx, median_area = _filter_bbox_sizes(
                boxes, 
                upper_multiplier= filter_bboxes[2],
                lower_multiplier= filter_bboxes[1],
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
         #   cv2.imwrite('output.jpg', plotted_img)
            cv2.imwrite('output_w_boxes.jpg', plotted_img_w_boxes)
            cv2.imwrite('sample_img.jpg', cv2.cvtColor(np.array(sample_img), cv2.COLOR_RGB2BGR))

        # Save masks
        masks_list.append(results[0].masks.data.cpu().numpy())
        # Save image IDs
        img_ids_list.append(sample_idx)
    
    if store_masks_bool:
        store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM3')

def model_SAM(samples, mode = 'mobile', save_imgs_bool = False, store_masks_bool = False, testing_samples = 1, filter_bboxes = (True, 0.2, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None)):
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

    """

    if type(samples) is not list:
        samples = [samples]

    # Load SAM model
    # We can set a higher conf score, possibly try finding some optimal one across multiple images...
    model = "pretrained_models/mobile_sam.pt" if mode == 'mobile' else "pretrained_models/sam_b.pt"
    overrides = dict(conf=0.70, 
                     task="segment", 
                     mode="predict", 
                     imgsz = 1024, # internal resize to (1024,1024) seems to give the best results, since SAM was trained on this size
                     model=model,
                     save = False) 
    predictor = SAMPredictor(overrides=overrides)
    

    masks_list = []
    img_ids_list = []

    # Let us only look at the first k samples for testing
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
        results = predictor(points = filtered_points) # PROMPT ENCODING (automatic 32x32 grid of points I think ?) AND MASK DECODING (for each of these points, predict masks)
       # results = predictor()
        end_time = time()
        print(f"Inference (prompt encoding and mask decoding) time for sample {sample_idx}: {end_time - start_time:.2f} seconds")

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
        if mode == 'mobile':
            store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM_mobile')
        else:
            store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM') 

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
    

def model_grounding_SAM(samples, mode = 'base', save_imgs_bool=False, store_masks_bool=False, testing_samples=1, filter_bboxes=(True, 0.2, 3.0), filter_masks_shapes=(True, 0.85), filter_masks_sizes=(True, 0.2, None)):
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
    """

    if type(samples) is not list:
        samples = [samples]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # BBOX DETECTOR : Grounding DINO
    detector_id = "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    labels = ["raspberry."]
    labels = [label if label.endswith(".") else label+"." for label in labels]

    # MASK GENERATOR : SAM
    segmenter_id = "pretrained_models/sam_b.pt"
    overrides = dict(conf=0.70, 
                     task="segment", 
                     mode="predict", 
                     imgsz = 1024, # internal resize to (1024,1024) seems to give the best results, since SAM was trained on this size
                     model=segmenter_id,
                     save = False) 
    mask_predictor = SAMPredictor(overrides=overrides)


    masks_list = []
    img_ids_list = []

    # Let us only look at the first k samples for testing
    samples = samples[:testing_samples]

    samples = [samples[0]]
  

    for sample in samples:

        sample_img = sample['image']
        sample_idx = sample['image_id']


        # TODO : understand why detector takes like 2 seconds longer here than in the grounding_SAM.py ...
        # DINO
        start_time_0 = time()
        detections = object_detector(sample_img, candidate_labels = labels, threshold = 0.2)
        end_time_0 = time()
        print(f"Detection time for sample {sample_idx}: {end_time_0 - start_time_0:.2f} seconds")
        detections = [DetectionResult.from_dict(det) for det in detections]

        bboxes_detections = get_boxes(detections)

        # Remove one layer of the inner list 
        bboxes_detections = np.array(bboxes_detections[0])

        # SAM
        start_time_1 = time()
        mask_predictor.set_image(sample_img) # IMAGE ENCODING
        end_time_1 = time()
        print(f"Image encoding time for sample {sample_idx}: {end_time_1 - start_time_1:.2f} seconds")
        start_time_2 = time()
        results = mask_predictor(bboxes = bboxes_detections) # PROMPT ENCODING AND MASK DECODING
        end_time_2 = time()
        print(f"Inference (prompt encoding and mask decoding) time for sample {sample_idx}: {end_time_2 - start_time_2:.2f} seconds")

        print(f"Full inference time DINO + SAM: {end_time_2 - start_time_2 + end_time_1 - start_time_1 + end_time_0 - start_time_0} seconds")

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
        if mode == 'mobile':
            store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM_mobile')
        else:
            store_masks(masks_list, img_ids_list, filepath= 'saved_masks/SAM') 




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

    MODE = "grounding_SAM"

    if MODE == "SAM_manipulate":
        model_SAM_manipulate(list(ds['train']), save_imgs_bool = True, store_masks_bool = False, testing_samples = 1, filter_bboxes = (False, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (False, 0.2, None))
        
    elif MODE == "SAM":
        model_SAM(list(ds['train']), mode = 'mobile', save_imgs_bool = False, store_masks_bool = True, testing_samples = 15, filter_bboxes = (True, None, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None))
    
    elif MODE == "grounding_SAM":
        model_grounding_SAM(list(ds['train']), mode = 'base', save_imgs_bool = True, store_masks_bool = False, testing_samples = 15, filter_bboxes = (True, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (False, 0.2, None))

    elif MODE == "FastSAM":
        model_FastSAM(list(ds['train']), save_imgs_bool = False, store_masks_bool = False, testing_samples = 15, filter_bboxes = (True, None, 3.0), filter_masks_shapes = (True, 0.85), filter_masks_sizes = (True, 0.2, None))

    elif MODE == "SAM3":
        model_SAM3(list(ds['train']), save_imgs_bool = False, store_masks_bool = False, testing_samples = 1, filter_masks_shapes = (False, 0.95), filter_bboxes = (False, 0.2, 3.0), filter_masks_sizes = (False, 0.2, None))




if __name__ == "__main__":
    main()