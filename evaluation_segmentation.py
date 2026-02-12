import numpy as np
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment
import cv2 
import pickle

def calculate_segmentation_metrics(pred_masks, gt_masks, iou_threshold=0.5, mode = 'pixels'):
    """
    Calculate instance segmentation metrics for multiple objects.
    
    Args:
        pred_masks: List of N predicted binary masks [H, W]
        gt_masks: List of M ground truth binary masks [H, W]
        iou_threshold: Minimum IoU to consider a match
        mode : 'pixels' or 'instances' - evaluation mode
    
    Returns:
        Dictionary with all metrics
    """
    
    # Match predictions to ground truth
    matched_pairs, unmatched_preds, unmatched_gts = match_instances(
        pred_masks, gt_masks, iou_threshold
    )
    
    # Calculate metrics
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    n_matched = len(matched_pairs)
    
    if mode == 'instances':
        # True Positives : matched instances
        TP = n_matched
        # False Positives: unmatched predictions
        FP = len(unmatched_preds)
        # False Negatives: unmatched ground truths
        FN = len(unmatched_gts)
    
    elif mode == 'pixels':
        ## For pixel-wise evaluation, calculate total TP, FP, FN based on masks

        # True Positives : sum of intersection areas of matched instances
        TP = 0
        # False Positives: sum of areas of unmatched predictions + unmatched areas in matched predictions (i.e. area was predicted but not in GT)
        FP = 0
        # False Negatives: sum of areas of unmatched ground truths + unmatched areas in matched GTs (i.e. area in GT but not predicted)
        FN = 0
        
        matched_pred_indices = set()
        matched_gt_indices = set()
        

        ## For the matched pairs
    
        for pred_idx, gt_idx, _ in matched_pairs:
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)
            intersection = np.logical_and(pred_masks[pred_idx], gt_masks[gt_idx]).sum()
            TP += intersection
            
            FP += pred_masks[pred_idx].sum() - intersection
            FN += gt_masks[gt_idx].sum() - intersection
        

        ## For the unmatched predictions and ground truths

        for pred_idx in range(n_pred):
            if pred_idx not in matched_pred_indices:
                FP += pred_masks[pred_idx].sum()
        
        for gt_idx in range(n_gt):
            if gt_idx not in matched_gt_indices:
                FN += gt_masks[gt_idx].sum()

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

def match_instances(pred_masks, gt_masks, iou_threshold=0.5, gt_grades=None):
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
            iou_matrix, iou_threshold, gt_grades
        )

    
    return matched_pairs, unmatched_preds, unmatched_gts

def _hungarian_matching(iou_matrix, iou_threshold, gt_grades=None):
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
            if gt_grades is None:
                matched_pairs.append((pred_idx, gt_idx, iou))
            else:
                matched_pairs.append((pred_idx, gt_idx, iou, gt_grades[gt_idx]))
            unmatched_preds.discard(pred_idx)
            unmatched_gts.discard(gt_idx)
    
    return matched_pairs, list(unmatched_preds), list(unmatched_gts)

def calculate_mask_iou(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

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

def _extract_masks(data, extract_grades_bool = False):
    """
    Extract ground truth masks from dataset samples.
    Args:
        data: List of dataset samples
        extract_grades_bool: Whether to also extract grades
    
    """
    all_gt_masks_ids = []
    all_gt_grades_ids = []
    for sample in data:
        masks_sample = []
        grades_sample = []
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
                # Also store grades, since we need to match them to the predicted masks for usage in anomaly detection later on
                grades_sample.append(int(label_data[0]))  
            
    
                

        all_gt_masks_ids.append((np.array(masks_sample), img_id))
        all_gt_grades_ids.append((np.array(grades_sample), img_id))

    if extract_grades_bool:
        return all_gt_masks_ids, all_gt_grades_ids
    return all_gt_masks_ids


# =============================================================================
# AP / mAP computation (COCO-style, mask IoU, single class)
# =============================================================================

def _compute_iou_matrix(pred_masks, gt_masks):
    """Compute full IoU matrix between predicted and GT masks for one image."""
    n_pred = len(pred_masks)
    n_gt = len(gt_masks)
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt))
    iou_matrix = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            iou_matrix[i, j] = calculate_mask_iou(pred_masks[i], gt_masks[j])
    return iou_matrix

def _greedy_match_at_threshold(iou_matrix, confidence_order, iou_threshold):
    """
    COCO-style greedy matching for a single image at a single IoU threshold.

    Predictions are processed in descending confidence order (given by
    `confidence_order`, an array of prediction indices already sorted).
    Each prediction is greedily matched to the GT with highest IoU that
    (a) exceeds `iou_threshold` and (b) has not already been matched.

    Returns:
        tp_flags: np.array of shape (n_pred,) with 1 for TP and 0 for FP,
                  ordered according to `confidence_order`.
    """
    n_pred, n_gt = iou_matrix.shape
    tp_flags = np.zeros(len(confidence_order), dtype=np.int32)
    matched_gt = set()

    for rank, pred_idx in enumerate(confidence_order):
        if n_gt == 0:
            break
        # Find best unmatched GT for this prediction
        best_iou = -1.0
        best_gt = -1
        for gt_idx in range(n_gt):
            if gt_idx in matched_gt:
                continue
            if iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt = gt_idx

        # Mark predicted mask as TP if best IoU exceeds threshold and GT is unmatched
        if best_iou >= iou_threshold and best_gt >= 0:
            tp_flags[rank] = 1
            matched_gt.add(best_gt)

    return tp_flags

def compute_ap_at_threshold(all_pred_masks, all_conf_scores, all_gt_masks, iou_threshold=0.5):
    """
    Compute AP at a single IoU threshold across all images (COCO-style).

    The key idea: pool *all* predictions from *all* images, sort globally by
    confidence, and sweep once to build a single precision-recall curve.

    Args:
        all_pred_masks:  List of lists/arrays of predicted binary masks, one per image.
        all_conf_scores: List of lists/arrays of confidence scores, one per image
                         (same structure / ordering as all_pred_masks).
        all_gt_masks:    List of lists/arrays of GT binary masks, one per image.
        iou_threshold:   IoU threshold for a match to count as TP.

    Returns:
        ap:    Average Precision (scalar).
        prec:  Precision values at each detection (array, length = total predictions).
        rec:   Recall values at each detection (array, length = total predictions).
    """
    # ------------------------------------------------------------------
    # 1. Per-image: compute IoU matrices and greedy TP/FP assignments
    # ------------------------------------------------------------------
    # We'll collect (confidence, tp_flag) for every prediction across all images.
    all_detections = []  # list of (confidence, tp_flag)
    total_gt = 0

    # Iterate over all images
    for img_idx in range(len(all_gt_masks)):
        pred_masks = all_pred_masks[img_idx]
        gt_masks = all_gt_masks[img_idx]
        conf_scores = all_conf_scores[img_idx]

        n_pred = len(pred_masks)
        n_gt = len(gt_masks)
        total_gt += n_gt

        if n_pred == 0:
            continue

        # Ensure conf_scores is a numpy array
        conf_scores = np.asarray(conf_scores, dtype=np.float64)

        # Sort predictions by confidence (descending) within this image
        # This ordering is needed for greedy matching (highest conf matched first), which I think ultralytics also does... i.e. not Hungarian Matching
        order = np.argsort(-conf_scores)

        iou_matrix = _compute_iou_matrix(pred_masks, gt_masks)
        tp_flags = _greedy_match_at_threshold(iou_matrix, order, iou_threshold)

        # Store (confidence, tp_flag) — use the *original* confidence for global sort
        for rank, pred_idx in enumerate(order):
            all_detections.append((conf_scores[pred_idx], tp_flags[rank]))

    # ------------------------------------------------------------------
    # 2. Global sort by confidence (descending)
    # ------------------------------------------------------------------
    if len(all_detections) == 0:
        return 0.0, np.array([]), np.array([])

    # AP calculation : https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/
    # Sort by confidence (descending)
    all_detections.sort(key=lambda x: -x[0])
    tp_array = np.array([d[1] for d in all_detections], dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Cumulative TP / FP → precision-recall curve
    # ------------------------------------------------------------------
    cum_tp = np.cumsum(tp_array)
    cum_fp = np.cumsum(1 - tp_array) # i.e. FP = 1 - TP since each detection is either TP or FP

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt if total_gt > 0 else cum_tp

    # ------------------------------------------------------------------
    # 4. COCO-style 101-point interpolated AP
    # ------------------------------------------------------------------
    ap = _interpolated_ap(precision, recall)

    return ap, precision, recall

def _interpolated_ap(precision, recall, n_points=101):
    """
    COCO-style 101-point interpolated AP.

    Evaluate precision at 101 equally spaced recall levels [0, 0.01, ..., 1.0].
    At each recall level r, the interpolated precision is the maximum precision
    at recall >= r.
    """
    if len(precision) == 0:
        return 0.0

    recall_levels = np.linspace(0, 1, n_points)

    # Make precision monotonically decreasing (right-to-left max)
    # This is the "envelope" of the PR curve
    prec_interp = np.zeros(n_points)
    for i, r in enumerate(recall_levels):
        # Precision at all recall values >= r
        mask = recall >= r
        if mask.any():
            prec_interp[i] = precision[mask].max() # i.e. max precision at recall >= r (precision to the right, i.e. see the webpage)
        else:
            prec_interp[i] = 0.0

    ap = prec_interp.mean()
    return ap

def compute_ap50(all_pred_masks, all_conf_scores, all_gt_masks):
    """Compute AP@50 (single IoU threshold = 0.50)."""
    ap, _, _ = compute_ap_at_threshold(all_pred_masks, all_conf_scores, all_gt_masks,
                                       iou_threshold=0.50)
    return ap

def compute_ap50_95(all_pred_masks, all_conf_scores, all_gt_masks):
    """
    Compute AP@50:95  (COCO primary metric).

    Average AP over 10 IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95.
    """
    thresholds = np.arange(0.50, 1.00, 0.05)  # [0.50, 0.55, ..., 0.95]
    aps = []
    for t in thresholds:
        ap, _, _ = compute_ap_at_threshold(all_pred_masks, all_conf_scores, all_gt_masks,
                                           iou_threshold=t)
        aps.append(ap)
    return np.mean(aps), dict(zip([f"AP@{t:.2f}" for t in thresholds], aps))

def main():
    ## Get the GT masks

    # Load dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

    # Extract masks from all training samples
    train_data = list(ds['train'])
    valid_data = list(ds['valid'])

    full_data = list(ds['train']) + list(ds['valid'])

    all_gt_masks_ids  = _extract_masks(full_data)

    ## Load predicted masks
    PRED_MASKS_FILE = '../../disk/saved_masks/DINO_SAM_mobile/masks.pkl'

    if PRED_MASKS_FILE.endswith('.pkl'):
        with open(PRED_MASKS_FILE, 'rb') as f:
            pred_data = pickle.load(f)
    else:
        pred_data = np.load(PRED_MASKS_FILE)


    all_pred_masks_ids = [(pred_data[key], key) for key in pred_data.keys()]

    # For testing only first 15 elements of gt mask
    #all_gt_masks_ids = all_gt_masks_ids[:5]

   # For evaluation against fine-tuned model, only look at validation samples in prediction (i.e. last 40 samples)
    #all_pred_masks_ids = all_pred_masks_ids[160:]




    # Sort both lists by image ID to ensure alignment
    all_gt_masks_ids.sort(key=lambda x: x[1])
    all_pred_masks_ids.sort(key=lambda x: x[1])




    # Remove sample idx from both lists
    all_gt_masks = [masks for masks, img_id in all_gt_masks_ids]


    if isinstance(all_pred_masks_ids[0][0], list) and len(all_pred_masks_ids[0][0]) == 4:
        all_pred_masks = [masks_and_xyn_and_conf_scores_and_imgs[0] for masks_and_xyn_and_conf_scores_and_imgs, img_id in all_pred_masks_ids]
        all_conf_scores = [masks_and_xyn_and_conf_scores_and_imgs[2] for masks_and_xyn_and_conf_scores_and_imgs, img_id in all_pred_masks_ids]
    else:
        all_pred_masks = [masks_and_conf_scores[0] for masks_and_conf_scores, img_id in all_pred_masks_ids]
        all_conf_scores = [masks_and_conf_scores[1] for masks_and_conf_scores, img_id in all_pred_masks_ids]


    # Calculate metrics for each sample (since each sample is image of multiple objects)
    avg_iou = 0.0
    avg_f1 = 0.0
    avg_precision = 0.0
    avg_recall = 0.0
    for sample_idx in range(len(all_gt_masks)):

       
        
        metrics = calculate_segmentation_metrics(
            all_pred_masks[sample_idx],
            all_gt_masks[sample_idx],
            iou_threshold=0.5,
            mode='pixels'
        )

        print(f"Sample {sample_idx} Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        print("\n")

        avg_iou += metrics['mean_iou']
        avg_f1 += metrics['f1_score']
        avg_precision += metrics['precision']
        avg_recall += metrics['recall']
    
    n_samples = len(all_gt_masks)
    print("Average Metrics Across All Samples:")
    print(f"  Average Mean IoU: {avg_iou / n_samples}")
    print(f"  Average F1 Score: {avg_f1 / n_samples}")
    print(f"  Average Precision: {avg_precision / n_samples}")
    print(f"  Average Recall: {avg_recall / n_samples}")


    # =========================================================================
    # AP / mAP metrics (COCO-style, pooled across all images)
    # =========================================================================
    print("\n" + "=" * 60)
    print("COCO-style AP Metrics (mask IoU, single class)")
    print("=" * 60)

    ap50 = compute_ap50(all_pred_masks, all_conf_scores, all_gt_masks)
    print(f"  AP@50:        {ap50:.4f}")

    ap50_95, per_threshold = compute_ap50_95(all_pred_masks, all_conf_scores, all_gt_masks)
    print(f"  AP@50:95:     {ap50_95:.4f}")
  #  print("  Per-threshold breakdown:")
  #  for name, val in per_threshold.items():
  #      print(f"    {name}: {val:.4f}")




    
        

    


if __name__ == "__main__":
    main()