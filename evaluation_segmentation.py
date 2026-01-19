import numpy as np
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
import cv2 

def calculate_instance_segmentation_metrics(pred_masks, gt_masks, iou_threshold=0.5):
    """
    Calculate instance segmentation metrics for multiple objects.
    
    Args:
        pred_masks: List of N predicted binary masks [H, W]
        gt_masks: List of M ground truth binary masks [H, W]
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        Dictionary with all metrics
    """
    
    # Step 1: Match predictions to ground truth
    matched_pairs, false_positives, false_negatives = match_instances(
        pred_masks, gt_masks, iou_threshold
    )
    
    # Step 2: Calculate metrics
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    n_matched = len(matched_pairs)
    
    # True Positives, False Positives, False Negatives
    TP = n_matched
    FP = len(false_positives)
    FN = len(false_negatives)
    
    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Mean IoU (only for matched instances)
    if n_matched > 0:
        ious = [iou for _, _, iou in matched_pairs]
        mean_iou = np.mean(ious)
        std_iou = np.std(ious)
    else:
        mean_iou = 0.0
        std_iou = 0.0
    
    
    return {
        'mean_iou': mean_iou,
        'std_iou': std_iou,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': TP,
        'false_positives': FP,
        'false_negatives': FN,
        'matched_pairs': n_matched,
        'n_predictions': n_pred,
        'n_ground_truth': n_gt
    }


def match_instances(pred_masks, gt_masks, iou_threshold=0.5):
    """
    Match predicted instances to ground truth using greedy or Hungarian matching.
    
    Args:
        pred_masks: List of N predicted binary masks [H, W]
        gt_masks: List of M ground truth binary masks [H, W]
        iou_threshold: Minimum IoU to consider a match valid
    
    Returns:
        matched_pairs: List of (pred_idx, gt_idx, iou) tuples
        unmatched_preds: List of prediction indices with no match (False Positives)
        unmatched_gts: List of ground truth indices with no match (False Negatives)
    
    References:
        - greedy: Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014
        - hungarian: Kirillov et al., "Panoptic Segmentation", CVPR 2019
    """
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    
    # Handle empty cases
    if n_pred == 0 or n_gt == 0:
        return [], list(range(n_pred)), list(range(n_gt))
    
    # Build IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = calculate_mask_iou(pred_masks[i], gt_masks[j])
    

    matched_pairs, unmatched_preds, unmatched_gts = _hungarian_matching(
            iou_matrix, iou_threshold
        )

    
    return matched_pairs, unmatched_preds, unmatched_gts


def _hungarian_matching(iou_matrix, iou_threshold):
    """
    Optimal bipartite matching using Hungarian algorithm.
    
    Args:
        iou_matrix: [N_pred, N_gt] matrix of IoU values
        iou_threshold: Minimum IoU for valid match
    
    Returns:
        matched_pairs, unmatched_preds, unmatched_gts
    """
    
    
    n_pred, n_gt = iou_matrix.shape

    # Find optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(iou_matrix, maximize= True)
    
    # Filter matches by IoU threshold
    matched_pairs = []
    unmatched_preds = set(range(n_pred))
    unmatched_gts = set(range(n_gt))
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou = iou_matrix[pred_idx, gt_idx]
        
        if iou >= iou_threshold:
            # Valid match
            matched_pairs.append((pred_idx, gt_idx, iou))
            unmatched_preds.discard(pred_idx)
            unmatched_gts.discard(gt_idx)
    
    return matched_pairs, list(unmatched_preds), list(unmatched_gts)


def calculate_mask_iou(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def metrics():
    pass 

def evaluation():
    pass 


def _points_to_mask(coords, width, height):

    # Scale to image size
    polygon_points = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            x = coords[i] * width
            y = coords[i + 1] * height
            polygon_points.append((x, y))

    polygon_points = np.array(polygon_points, dtype=np.int32)

    def _polygon_to_mask(polygon_points, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points], 1)
        return mask.astype(bool)
    
    return _polygon_to_mask(polygon_points, width, height)





def main():
    ## Get the GT masks

    # Load dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

    # Extract masks from all training samples
    train_data = list(ds['train'])
    all_gt_masks = []
    
    for sample in train_data:
        masks_sample = []
        labels = sample['labels']
        img = sample['image']
        img_id = sample['image_id']
        width, height = img.size
        for label_data in labels:
            class_id = int(label_data[0])
            if class_id > 0:  # Exclude class 0 (bonnet)
                mask_points = label_data[1:]
                # Convert points to binary mask
                mask = _points_to_mask(mask_points, width, height)
                masks_sample.append(mask)

        all_gt_masks.append((np.array(masks_sample), img_id))


    ## Load predicted masks
    PRED_MASKS_FILE = 'saved_masks/SAM3.npz'
    pred_data = np.load(PRED_MASKS_FILE)
    all_pred_masks = [(pred_data[key], key) for key in pred_data.keys()]

    # For testing only first 15 elements of gt mask
    all_gt_masks = all_gt_masks[:15]

    # Sort both lists by image ID to ensure alignment
    all_gt_masks.sort(key=lambda x: x[1])
    all_pred_masks.sort(key=lambda x: x[1])

    # Remove sample idx from both lists
    all_gt_masks = [masks for masks, img_id in all_gt_masks]
    all_pred_masks = [masks for masks, img_id in all_pred_masks]


    # Calculate metrics for each sample (since each sample is image of multiple objects)
    for sample_idx in range(len(all_gt_masks)):
        
        metrics = calculate_instance_segmentation_metrics(
            all_gt_masks[sample_idx],
            all_pred_masks[sample_idx],
            iou_threshold=0.5
        )

        print(f"Sample {sample_idx} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("\n")



    
        

    


if __name__ == "__main__":
    main()