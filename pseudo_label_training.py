# YOLO-seg model pipeline

from datasets import load_dataset
import numpy as np
from pathlib import Path
from time import time
import pickle
import shutil
from scipy.spatial import ConvexHull
import torch
from ultralytics import YOLO
from ultralytics.models.sam.amg import remove_small_regions
from transformers import pipeline
from SAM_segmentation import store_masks, _filter_red_masks, _filter_bbox_sizes, _filter_mask_shapes, _filter_mask_sizes, filter_overlapping_masks_extended, filter_overlapping_masks_extended_old
from evaluation_segmentation import calculate_segmentation_metrics, _points_to_mask, compute_ap50, compute_ap50_95


def _convex_hull_polygon(poly):
    """Return the convex hull vertices of a (N, 2) normalised polygon array.
    Falls back to the original polygon if fewer than 3 unique points exist."""
    if len(poly) < 3:
        return poly
    try:
        hull = ConvexHull(poly)
        return poly[hull.vertices]
    except Exception:
        return poly

def setup_yolo_dataset_structure_extended(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES, N_VAL_SAMPLES):
    # This is essentially for when we want to also cutoff a test set (i.e. we have train and validation for YOLO training)

    images_dir_train = Path("../../disk/YOLO_dataset_extended/images/train")
    images_dir_val   = Path("../../disk/YOLO_dataset_extended/images/val")
    images_dir_test  = Path("../../disk/YOLO_dataset_extended/images/test")

    for d in [images_dir_train, images_dir_val, images_dir_test]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for idx, (img, img_name) in enumerate(zip(all_imgs, all_pred_ids)):
        img_name = f"{img_name}.png"
        if idx < N_TRAIN_SAMPLES:
            img_path = images_dir_train / img_name
        elif idx < N_TRAIN_SAMPLES + N_VAL_SAMPLES:
            img_path = images_dir_val / img_name
        else:
            img_path = images_dir_test / img_name

        img.save(img_path)

    img_paths = (
        sorted(list(images_dir_train.glob("*.png"))) +
        sorted(list(images_dir_val.glob("*.png")))   +
        sorted(list(images_dir_test.glob("*.png")))
    )

    polygons_per_image = all_pred_xyn

    # Replace val AND test labels with GT labels (not pseudo labels)
    val_labels = [[np.array(label) for label in sample_labels] for sample_labels in val_labels]
    polygons_per_image = polygons_per_image[:N_TRAIN_SAMPLES] + val_labels

    labels_dir_train = Path("../../disk/YOLO_dataset_extended/labels/train")
    labels_dir_val   = Path("../../disk/YOLO_dataset_extended/labels/val")
    labels_dir_test  = Path("../../disk/YOLO_dataset_extended/labels/test")

    for d in [labels_dir_train, labels_dir_val, labels_dir_test]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, polys) in enumerate(zip(img_paths, polygons_per_image)):

        img_name = Path(img_path).stem
        if idx < N_TRAIN_SAMPLES:
            label_file = labels_dir_train / f"{img_name}.txt"
        elif idx < N_TRAIN_SAMPLES + N_VAL_SAMPLES:
            label_file = labels_dir_val / f"{img_name}.txt"
        else:
            label_file = labels_dir_test / f"{img_name}.txt"

        if label_file.exists():
            label_file.unlink()

        lines = []
        for poly in polys:
            coords = poly.flatten()
            line = "0 " + " ".join(map(str, coords))
            lines.append(line)

        label_file.write_text("\n".join(lines))

def setup_yolo_dataset_structure(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES, mode="raspberry", train_labels=None):
    """Build the YOLO folder/label structure for training and validation.

    By default the training split uses pseudo-labels (`all_pred_xyn`) and the
    validation split uses ground-truth labels (`val_labels`).  Pass
    `train_labels` to replace the pseudo-labels with actual GT annotations for
    the training split as well (upper-bound / oracle experiment).
    """

    # Adjust to YOLO format : https://docs.ultralytics.com/datasets/segment/#ultralytics-yolo-format

    # First off, YOLO format requires images to be in a folder structure, with labels in a separate folder.
    # So let us first deal with the images

    if mode == "punnet":
        base_dir = Path("../../disk/YOLO_dataset_punnet")
        class_name = "punnet"
    else:
        base_dir = Path("../../disk/YOLO_dataset")
        class_name = "raspberry"

    images_dir_train = base_dir / "images/train"
    images_dir_val   = base_dir / "images/val"

    # Remove directories if they exist
    if images_dir_train.exists():
        shutil.rmtree(images_dir_train)
    if images_dir_val.exists():
        shutil.rmtree(images_dir_val)

    # Recreate directories
    images_dir_train.mkdir(parents=True, exist_ok=True)
    images_dir_val.mkdir(parents=True, exist_ok=True)

    for idx, (img, img_name) in enumerate(zip(all_imgs, all_pred_ids)):
        img_name = f"{img_name}.png"
        if idx < N_TRAIN_SAMPLES:
            img_path = images_dir_train / img_name
        else:
            img_path = images_dir_val / img_name

        img.save(img_path)

    img_paths = sorted(list(images_dir_train.glob("*.png"))) + sorted(list(images_dir_val.glob("*.png")))

    val_labels = [[np.array(label) for label in sample_labels] for sample_labels in val_labels]

    if train_labels is not None:
        # GT labels for training (oracle / upper-bound experiment)
        train_labels = [[np.array(label) for label in sample_labels] for sample_labels in train_labels]
        polygons_per_image = train_labels + val_labels
    else:
        # Default: pseudo-labels for training, GT for validation
        polygons_per_image = all_pred_xyn[:N_TRAIN_SAMPLES] + val_labels

    labels_dir_train = base_dir / "labels/train"
    labels_dir_val   = base_dir / "labels/val"

    if labels_dir_train.exists():
        shutil.rmtree(labels_dir_train)
    if labels_dir_val.exists():
        shutil.rmtree(labels_dir_val)

    labels_dir_train.mkdir(parents=True, exist_ok=True)
    labels_dir_val.mkdir(parents=True, exist_ok=True)

    for idx, (img_path, polys) in enumerate(zip(img_paths, polygons_per_image)):

        img_name = Path(img_path).stem
        if idx < N_TRAIN_SAMPLES:
            label_file = labels_dir_train / f"{img_name}.txt"
        else:
            label_file = labels_dir_val / f"{img_name}.txt"


        # Check if label file already exists, if so, delete and create a new one (to avoid appending to existing file)
        if label_file.exists():
            label_file.unlink()


        lines = []


        for poly in polys:

            if mode == "punnet":
                poly = _convex_hull_polygon(poly)

            # flatten x,y pairs to get it into wanted YOLO shape : x1 y1 x2 y2 x3 y3 ... xn yn
            coords = poly.flatten()



            # NOTE : for now, just class index 0 since all objects are raspberries ; need to see if we need to adjust
            # NOTE: for raspberry grades or not
            # Creates class_index x1 y1 x2 y2 ... xn yn
            line = "0 " + " ".join(map(str, coords))
            lines.append(line)

        label_file.write_text("\n".join(lines))

    # Write YAML config so training can reference the correct dataset and class name
    yaml_content = (
        f"path: {base_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    (base_dir / f"{class_name}-seg.yaml").write_text(yaml_content)

def evaluate_yolo_iou(model, val_data, val_labels, device=2,
                      filter_red=(False, 0.3),
                      filter_bboxes=(False, None, 3.0),
                      filter_masks_shapes=(False, 0.85),
                      filter_masks_sizes=(False, 0.2, None),
                      filter_holes_islands=False,
                      filter_overlap_masks=(False, 'new')):
    """Compute the same averaged mean IoU used in evaluation_segmentation.py for SAM models.
    This is mainly implemented since we calculate based on pixel level metrics, which is not the standard in YOLO.
    Optionally applies the same post-prediction filters used in model_SAM_extended before computing metrics.
    filter_overlap_masks: (bool, 'new'|'old') — 'new' uses filter_overlapping_masks_extended,
        'old' uses filter_overlapping_masks_extended_old (F2 algorithm)."""
    avg_iou = 0.0
    avg_f1 = 0.0
    avg_precision = 0.0
    avg_recall = 0.0

    all_pred_masks = []
    all_conf_scores = []
    all_gt_masks = []

    depth_pipe = None
    if filter_overlap_masks[0]:
        depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=device if torch.cuda.is_available() else 'cpu')

    times_predict = []
    times_red = []
    times_bbox = []
    times_mask_shape = []
    times_mask_size = []
    times_holes_islands = []
    times_depth = []
    times_overlap = []

    for loop_idx, (sample, gt_polys) in enumerate(zip(val_data, val_labels)):
        img = sample['image']
        width, height = img.size


        # retina_masks=True returns masks at original image resolution, required for the red filter
        _t0 = time()
        results = model.predict(img, device=device, verbose=False, retina_masks=True)
        _t1 = time()
        if loop_idx > 0:
            times_predict.append(_t1 - _t0)

        if results[0].masks is not None:
            img_array = np.array(img)

            # 1. Red filter — discard non-raspberry masks
            if filter_red[0]:
                _t0 = time()
                masks_np = results[0].masks.data.cpu().numpy()
                red_valid_idx = _filter_red_masks(masks_np, img_array, min_red_fraction=filter_red[1])
                results[0].boxes = results[0].boxes[red_valid_idx]
                results[0].masks = results[0].masks[red_valid_idx]
                if loop_idx > 0:
                    times_red.append(time() - _t0)

            # 2. Bbox size filter
            if filter_bboxes[0] and results[0].masks is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    _t0 = time()
                    valid_idx, _ = _filter_bbox_sizes(boxes, upper_multiplier=filter_bboxes[2], lower_multiplier=filter_bboxes[1])
                    results[0].boxes = results[0].boxes[valid_idx]
                    results[0].masks = results[0].masks[valid_idx]
                    if loop_idx > 0:
                        times_bbox.append(time() - _t0)

            # 3. Mask shape filter (rectangularity)
            if filter_masks_shapes[0] and results[0].masks is not None:
                _t0 = time()
                masks_np = results[0].masks.data.cpu().numpy()
                boxes_np = results[0].boxes.xyxy.cpu().numpy()
                shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
                results[0].boxes = results[0].boxes[shape_valid_idx]
                results[0].masks = results[0].masks[shape_valid_idx]
                if loop_idx > 0:
                    times_mask_shape.append(time() - _t0)

            # 4. Mask size filter
            if filter_masks_sizes[0] and results[0].masks is not None:
                _t0 = time()
                masks_np = results[0].masks.data.cpu().numpy()
                size_valid_idx, _ = _filter_mask_sizes(masks_np, upper_multiplier=filter_masks_sizes[2], lower_multiplier=filter_masks_sizes[1])
                results[0].boxes = results[0].boxes[size_valid_idx]
                results[0].masks = results[0].masks[size_valid_idx]
                if loop_idx > 0:
                    times_mask_size.append(time() - _t0)

            # 5. Holes-and-islands cleanup — replaces the YOLO Masks object with a plain tensor
            if filter_holes_islands and results[0].masks is not None:
                _t0 = time()
                masks = results[0].masks.data.cpu().numpy()
                islands_threshold = int(min(m.astype(np.bool_).sum() for m in masks)) - 1
                refined_masks = []
                for mask in masks:
                    mask = mask.astype(np.bool_)
                    refined_mask, _ = remove_small_regions(mask, islands_threshold, mode='islands')
                    refined_mask, _ = remove_small_regions(refined_mask, 20000, mode='holes')
                    refined_masks.append(refined_mask)
                results[0].masks = torch.from_numpy(np.array(refined_masks))
                if loop_idx > 0:
                    times_holes_islands.append(time() - _t0)

            # 6. Overlap filter — guided by depth estimation
            if filter_overlap_masks[0] and results[0].masks is not None:
                _t0 = time()
                depth_array = np.array(depth_pipe(img)["depth"])
                if loop_idx > 0:
                    times_depth.append(time() - _t0)
                _t0 = time()
                masks = results[0].masks.data.cpu().numpy()
                masks_depth_values = np.array([depth_array * mask for mask in masks])
                if filter_overlap_masks[1] == 'old':
                    masks_filtered_dict = filter_overlapping_masks_extended_old(
                        masks,
                        masks_depth_values,
                        overlap_threshold=50,
                        containment_threshold=0.95,
                    )
                else:
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
                if loop_idx > 0:
                    times_overlap.append(time() - _t0)

        if results[0].masks is not None:
            # holes/islands filter replaces the YOLO Masks object with a plain tensor, so .xyn is unavailable
            if isinstance(results[0].masks, torch.Tensor):
                pred_masks = list(results[0].masks.cpu().numpy().astype(bool))
            else:
                pred_masks = [_points_to_mask(poly.flatten(), width, height)
                              for poly in results[0].masks.xyn]
            conf_scores = results[0].boxes.conf.cpu().numpy()
        else:
            pred_masks = []
            conf_scores = np.array([])

        gt_masks = [_points_to_mask(np.array(poly).flatten(), width, height)
                    for poly in gt_polys]

        metrics = calculate_segmentation_metrics(pred_masks, gt_masks, iou_threshold=0.5, mode='pixels')
        avg_iou += metrics['mean_iou']
        avg_f1 += metrics['f1_score']
        avg_precision += metrics['precision']
        avg_recall += metrics['recall']

        all_pred_masks.append(pred_masks)
        all_conf_scores.append(conf_scores)
        all_gt_masks.append(gt_masks)

    n = len(val_data)
    print(f"Average Mean IoU: {avg_iou / n:.4f}")
    print(f"Average F1 Score: {avg_f1 / n:.4f}")
    print(f"Average Precision: {avg_precision / n:.4f}")
    print(f"Average Recall:    {avg_recall / n:.4f}")

    ap50 = compute_ap50(all_pred_masks, all_conf_scores, all_gt_masks)
    ap50_95, _ = compute_ap50_95(all_pred_masks, all_conf_scores, all_gt_masks)
    print(f"AP@50:             {ap50:.4f}")
    print(f"AP@50:95:          {ap50_95:.4f}")

    if times_predict:
        def _avg(lst): return sum(lst) / len(lst) if lst else 0.0

        print("\n--- Timing Summary (excluding first sample) ---")
        print(f"Avg YOLO prediction      : {_avg(times_predict):.3f} s")
        if times_red:
            print(f"Avg red filter           : {_avg(times_red):.3f} s")
        if times_bbox:
            print(f"Avg bbox size filter     : {_avg(times_bbox):.3f} s")
        if times_mask_shape:
            print(f"Avg mask shape filter    : {_avg(times_mask_shape):.3f} s")
        if times_mask_size:
            print(f"Avg mask size filter     : {_avg(times_mask_size):.3f} s")
        if times_holes_islands:
            print(f"Avg holes/islands filter : {_avg(times_holes_islands):.3f} s")
        if times_depth:
            print(f"Avg depth estimation     : {_avg(times_depth):.3f} s")
        if times_overlap:
            print(f"Avg overlap filter       : {_avg(times_overlap):.3f} s")
        full_avg = (_avg(times_predict) + _avg(times_red) + _avg(times_bbox) +
                    _avg(times_mask_shape) + _avg(times_mask_size) +
                    _avg(times_holes_islands) + _avg(times_depth) + _avg(times_overlap))
        print(f"Avg full time (sum)      : {full_avg:.3f} s")
        print("-----------------------------------------------")

def run_yolo_and_store_masks(model, filepath, device=2,
                             filter_red=(False, 0.3),
                             filter_bboxes=(False, None, 3.0),
                             filter_masks_shapes=(False, 0.85),
                             filter_masks_sizes=(False, 0.2, None),
                             filter_holes_islands=False,
                             filter_overlap_masks=(False, 'new')):

    masks_list = []
    xyn_list = []
    conf_scores_list = []
    sample_imgs = []
    img_ids_list = []

    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")

    full_data = list(ds['train']) + list(ds['valid'])

    train_img_paths = sorted(list(Path("../../disk/YOLO_dataset/images/train").glob("*.png")))
    val_img_paths = sorted(list(Path("../../disk/YOLO_dataset/images/val").glob("*.png")))

    all_img_paths = train_img_paths + val_img_paths

    depth_pipe = None
    if filter_overlap_masks[0]:
        depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=device if torch.cuda.is_available() else 'cpu')

    for img_id, sample_path in enumerate(all_img_paths):
        img = full_data[img_id]['image']
        idx = full_data[img_id]['image_id']

        results = model.predict(sample_path, device=device, verbose=False, retina_masks=True)

        if results[0].masks is not None:
            img_array = np.array(img)

            # 1. Red filter — discard non-raspberry masks
            if filter_red[0]:
                masks_np = results[0].masks.data.cpu().numpy()
                red_valid_idx = _filter_red_masks(masks_np, img_array, min_red_fraction=filter_red[1])
                results[0].boxes = results[0].boxes[red_valid_idx]
                results[0].masks = results[0].masks[red_valid_idx]

            # 2. Bbox size filter
            if filter_bboxes[0] and results[0].masks is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    valid_idx, _ = _filter_bbox_sizes(boxes, upper_multiplier=filter_bboxes[2], lower_multiplier=filter_bboxes[1])
                    results[0].boxes = results[0].boxes[valid_idx]
                    results[0].masks = results[0].masks[valid_idx]

            # 3. Mask shape filter (rectangularity)
            if filter_masks_shapes[0] and results[0].masks is not None:
                masks_np = results[0].masks.data.cpu().numpy()
                boxes_np = results[0].boxes.xyxy.cpu().numpy()
                shape_valid_idx = _filter_mask_shapes(masks_np, boxes_np, rectangularity_threshold=filter_masks_shapes[1])
                results[0].boxes = results[0].boxes[shape_valid_idx]
                results[0].masks = results[0].masks[shape_valid_idx]

            # 4. Mask size filter
            if filter_masks_sizes[0] and results[0].masks is not None:
                masks_np = results[0].masks.data.cpu().numpy()
                size_valid_idx, _ = _filter_mask_sizes(masks_np, upper_multiplier=filter_masks_sizes[2], lower_multiplier=filter_masks_sizes[1])
                results[0].boxes = results[0].boxes[size_valid_idx]
                results[0].masks = results[0].masks[size_valid_idx]

            # 5. Holes-and-islands cleanup — replaces the YOLO Masks object with a plain tensor
            if filter_holes_islands and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                islands_threshold = int(min(m.astype(np.bool_).sum() for m in masks)) - 1
                refined_masks = []
                for mask in masks:
                    mask = mask.astype(np.bool_)
                    refined_mask, _ = remove_small_regions(mask, islands_threshold, mode='islands')
                    refined_mask, _ = remove_small_regions(refined_mask, 20000, mode='holes')
                    refined_masks.append(refined_mask)
                results[0].masks = torch.from_numpy(np.array(refined_masks))

            # 6. Overlap filter — guided by depth estimation
            if filter_overlap_masks[0] and results[0].masks is not None:
                depth_array = np.array(depth_pipe(img)["depth"])
                masks = results[0].masks.data.cpu().numpy()
                masks_depth_values = np.array([depth_array * mask for mask in masks])
                if filter_overlap_masks[1] == 'old':
                    masks_filtered_dict = filter_overlapping_masks_extended_old(
                        masks,
                        masks_depth_values,
                        overlap_threshold=50,
                        containment_threshold=0.95,
                    )
                else:
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

        if results[0].masks is None:
            masks_list.append(np.zeros((0, img.size[1], img.size[0]), dtype=bool))
            xyn_list.append([])
            conf_scores_list.append(np.array([]))
        else:
            # holes/islands filter replaces the YOLO Masks object with a plain tensor, so .xyn is unavailable
            if isinstance(results[0].masks, torch.Tensor):
                masks_list.append(results[0].masks.cpu().numpy().astype(bool))
                xyn_list.append([])
            else:
                masks_list.append(results[0].masks.data.cpu().numpy().astype(bool))
                xyn_list.append(results[0].masks.xyn)
            conf_scores_list.append(results[0].boxes.conf.cpu().numpy())

        sample_imgs.append(img)
        img_ids_list.append(idx)

    store_masks(masks_list, img_ids_list, conf_scores_list, xyn_list, sample_imgs, filepath=filepath)


def main():
    # Get raspberry dataset
    ds = load_dataset("FBK-TeV/RaspGrade")


    # Define the given data split : 160 training, 40 validation samples
    # Note that we will take the N_TRAIN_SAMPLES from our large model (e.g. SAM3),
    # but the N_VAL_SAMPLES from the original GT.
    N_TRAIN_SAMPLES = 160
    N_VAL_SAMPLES = 40
  #  N_TEST_SAMPLES = 20

    # Load GT masks for evaluation later on
    train_data = list(ds['train'])
    val_data = list(ds['valid'])
    
    
    val_labels = [sample['labels'] for sample in val_data]

    # Remove punnet mask (class 0)
    val_labels = [[label for label in sample_labels if label[0] != 0] for sample_labels in val_labels]

    # Do not record the raspberry grade, since we are only interested in segmentation performance for now, not classification performance
    val_labels = [[label[1:] for label in sample_labels] for sample_labels in val_labels]

    # GT labels for the training split (oracle / upper-bound experiment)
    train_labels = [sample['labels'] for sample in train_data]
    train_labels = [[label for label in sample_labels if label[0] != 0] for sample_labels in train_labels]
    train_labels = [[label[1:] for label in sample_labels] for sample_labels in train_labels]

    ## Load predicted masks
    PRED_MASKS_FILE = '../../nvme1/thesis/saved_masks/sam3/masks.pkl'
    with open(PRED_MASKS_FILE, 'rb') as f:
        pred_data = pickle.load(f)

    all_pred_masks_ids = [(pred_data[key], key) for key in pred_data.keys()]


    # Sort by image ID to ensure alignment
    all_pred_masks_ids.sort(key=lambda x: x[1])

    all_pred_ids = [img_id for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_pred_masks = [masks_and_xyn_and_imgs[0] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_pred_xyn = [masks_and_xyn_and_imgs[1] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_conf_scores = [masks_and_xyn_and_imgs[2] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]
    all_imgs = [masks_and_xyn_and_imgs[3] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids]


    


    # Option A: train on pseudo-labels from the large model (default)
    setup_yolo_dataset_structure(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES)

    # Option B: train on actual ground-truth labels (oracle / upper-bound experiment)
   # setup_yolo_dataset_structure(all_imgs, all_pred_xyn, all_pred_ids, val_labels, N_TRAIN_SAMPLES, train_labels=train_labels)


    # Load a model
    model = YOLO("../../disk/pretrained_models/yolo26n-seg.pt")  # Load pretrained model
    # Train the model
    results = model.train(data="../../disk/YOLO_dataset/RaspGrade-seg.yaml", epochs=100, imgsz= 640, name = "yolo26n-seg-gt-frozen-0", device = 2, freeze= 0)

    # Load trained model
    model = YOLO("../../disk/pretrained_models/yolo26n-seg-gt-best-640-input-frozen-0.pt") 
    #metrics = model.val(data="../../disk/YOLO_dataset/RaspGrade-seg.yaml", device = 2)
    evaluate_yolo_iou(model, val_data, val_labels, device=2, filter_bboxes = (False, None, 3.0), filter_masks_shapes = (False, 0.85), filter_masks_sizes = (False, 0.2, None), filter_red = (False, 0.3), filter_holes_islands = False, filter_overlap_masks = (False, 'new'))

    run_yolo_and_store_masks(model, filepath="../../nvme1/thesis/saved_masks/yolo_640", device=2)






    '''
    # YOLO for detecting the punnet class

    # GT val labels for punnet: keep only class-0 entries, then strip the class index
    #val_labels_punnet = [sample['labels'] for sample in val_data]
    #val_labels_punnet = [[label for label in sample_labels if label[0] == 0] for sample_labels in val_labels_punnet]
    #val_labels_punnet = [[label[1:] for label in sample_labels] for sample_labels in val_labels_punnet]

    # Load predicted punnet masks (produced by SAM3 on the punnet class)
    #PUNNET_MASKS_FILE = '../../disk/saved_masks/SAM3_punnet/masks.pkl'
    #with open(PUNNET_MASKS_FILE, 'rb') as f:
    #    pred_data_punnet = pickle.load(f)

    #all_pred_masks_ids_punnet = [(pred_data_punnet[key], key) for key in pred_data_punnet.keys()]
    #all_pred_masks_ids_punnet.sort(key=lambda x: x[1])

    #all_pred_ids_punnet    = [img_id for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids_punnet]
    #all_pred_xyn_punnet    = [masks_and_xyn_and_imgs[1] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids_punnet]
    #all_imgs_punnet        = [masks_and_xyn_and_imgs[3] for masks_and_xyn_and_imgs, img_id in all_pred_masks_ids_punnet]

   # setup_yolo_dataset_structure(all_imgs_punnet, all_pred_xyn_punnet, all_pred_ids_punnet, val_labels_punnet, N_TRAIN_SAMPLES, mode="punnet")

    # Load a pretrained model and train on punnet pseudo-labels
   # model_punnet = YOLO("../../disk/pretrained_models/yolo26n-seg.pt")
   # results_punnet = model_punnet.train(data="../../disk/YOLO_dataset_punnet/punnet-seg.yaml", epochs=100, imgsz=640, name="yolo26n-seg-punnet-pseudo-labels-convexhull", device=2)

    # Load trained punnet model and evaluate
  #  model_punnet = YOLO("../../disk/pretrained_models/yolo26n-seg-punnet-pseudo-labels-best.pt")
  #  evaluate_yolo_iou(model_punnet, val_data, val_labels_punnet, device=2)
    '''













if __name__ == "__main__":
    main()
    
