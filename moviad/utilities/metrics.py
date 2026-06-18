from __future__ import annotations

import faiss
import torch
from sympy.codegen.cnodes import sizeof
from torch.nn import functional as F
from sklearn.metrics import *
import numpy as np
from skimage.measure import label, regionprops
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import os
import cv2 as cv
from pathlib import Path
import shutil


from moviad.models.patchcore.product_quantizer import ProductQuantizer


def compute_quantizer_config_size(quantizer: faiss.IndexPQ) -> int:
    centroids_size = quantizer.pq.centroids.size() * np.dtype(np.float32).itemsize
    m_size = np.dtype(np.int32).itemsize
    k_size = np.dtype(np.int32).itemsize
    total_size = centroids_size + m_size + k_size
    return total_size


def compute_product_quantization_efficiency(coreset: np.ndarray, compressed_coreset: np.ndarray,
                                            quantizer: ProductQuantizer) -> (float, float):
    np_array_type = coreset.dtype
    compressed_np_array_type = compressed_coreset.dtype
    original_shape = coreset.shape
    compressed_shape = compressed_coreset.shape
    product_quantized_config_size = compute_quantizer_config_size(quantizer.quantizer)
    original_bitrate = np_array_type.itemsize * np.prod(original_shape) * 8
    compressed_bitrate = (compressed_np_array_type.itemsize * np.prod(compressed_shape) + product_quantized_config_size) * 8
    compression_efficiency = 1 - compressed_bitrate / original_bitrate
    dequantized_coreset = quantizer.decode(compressed_coreset).cpu().numpy()
    distortion = (np.linalg.norm(coreset - dequantized_coreset)/np.linalg.norm(coreset))
    return compression_efficiency, distortion




def visualize_embeddings_tsne(embeddings: np.ndarray, labels: np.ndarray, perplexity: int = 30, n_iter: int = 1000, plot_3d: bool = False, anomaly_threshold: int = 4):
    """
    Visualize tSNE embeddings for whole images. Creates two subplots:
    - Left: each grade as a unique color
    - Right: binary split (normal vs anomalous) based on anomaly_threshold
    If plot_3d=True, creates 3D plots.
    """
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = plt.cm.get_cmap('tab10' if n_labels <= 10 else 'tab20', n_labels)

    n_components = 3 if plot_3d else 2
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=n_iter, random_state=42)
    embeddings_nd = tsne.fit_transform(embeddings)

    projection = {'projection': '3d'} if plot_3d else {}
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121, **projection)
    ax2 = fig.add_subplot(122, **projection)

    # Left: per-grade colors
    for i, label in enumerate(unique_labels):
        mask = labels == label
        kwargs = dict(c=[cmap(i)], label=f'Grade {label}', alpha=0.7, s=30)
        if plot_3d:
            ax1.scatter(embeddings_nd[mask, 0], embeddings_nd[mask, 1], embeddings_nd[mask, 2], **kwargs)
        else:
            ax1.scatter(embeddings_nd[mask, 0], embeddings_nd[mask, 1], **kwargs)

    # Right: binary normal/anomalous
    normal_mask = labels < anomaly_threshold
    anomalous_mask = labels >= anomaly_threshold
    for mask, color, name in [(normal_mask, 'tab:blue', f'Normal (grade < {anomaly_threshold})'),
                               (anomalous_mask, 'tab:red', f'Anomalous (grade >= {anomaly_threshold})')]:
        if not mask.any():
            continue
        kwargs = dict(c=color, label=name, alpha=0.7, s=30)
        if plot_3d:
            ax2.scatter(embeddings_nd[mask, 0], embeddings_nd[mask, 1], embeddings_nd[mask, 2], **kwargs)
        else:
            ax2.scatter(embeddings_nd[mask, 0], embeddings_nd[mask, 1], **kwargs)

    for ax, title in [(ax1, 'Per-Grade'), (ax2, 'Normal vs Anomalous')]:
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        if plot_3d:
            ax.set_zlabel('Dimension 3')
        ax.legend()
        ax.set_title(f't-SNE Embeddings — {title}')

    plt.tight_layout()
    plt.savefig('tsne_visualization.png', dpi=150)




def visualize_patch_embeddings_pairs_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    image_paths: list,
    memory_bank: np.ndarray,
    n_samples: int = 4,
    perplexity: int = 30,
    n_iter: int = 1000
):
    assert n_samples % 2 == 0, "n_samples must be even"
    grades = np.unique(labels)
    assert len(grades) == 2, "Function assumes exactly two grades"

    idx_grade_0 = np.where(labels == grades[0])[0]
    idx_grade_1 = np.where(labels == grades[1])[0]

    rng = np.random.default_rng(42)
    sample_0 = rng.choice(idx_grade_0, n_samples // 2, replace=False)
    sample_1 = rng.choice(idx_grade_1, n_samples // 2, replace=False)

    n_pairs = (n_samples // 2) * (n_samples // 2)
    # 4 rows: pair-only tsne, mem+normal, mem+anomalous, images
    fig, axes = plt.subplots(4, n_pairs, figsize=(6 * n_pairs, 24))
    if n_pairs == 1:
        axes = axes.reshape(4, 1)

    pair_count = 0
    for i, idx_0 in enumerate(sample_0):
        for j, idx_1 in enumerate(sample_1):
            pair_embeddings = np.concatenate([embeddings[idx_0], embeddings[idx_1]], axis=0)
            pair_grades = np.array(
                [labels[idx_0]] * embeddings.shape[1] +
                [labels[idx_1]] * embeddings.shape[1]
            )

            # Row 0: pair-only tSNE
            tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
            embeddings_2d = tsne.fit_transform(pair_embeddings)
            ax = axes[0, pair_count]
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=pair_grades, cmap='viridis', alpha=0.7)
            ax.set_title(f'Non-anom idx {idx_0} vs Anom idx {idx_1}')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=f'Sample {idx_0} (grade {labels[idx_0]})', markerfacecolor='C0', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label=f'Sample {idx_1} (grade {labels[idx_1]})', markerfacecolor='C1', markersize=10)
            ]
            ax.legend(handles=handles)
            fig.colorbar(scatter, ax=ax, label='Grade')

            # Row 1: memory bank + normal sample only
            normal_with_mem = np.concatenate([embeddings[idx_0], memory_bank], axis=0)
            tsne_normal = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
            normal_2d = tsne_normal.fit_transform(normal_with_mem)
            n_patches = embeddings.shape[1]
            ax_n = axes[1, pair_count]
            ax_n.scatter(normal_2d[n_patches:, 0], normal_2d[n_patches:, 1], c='red', alpha=0.15, label='Memory Bank')
            ax_n.scatter(normal_2d[:n_patches, 0], normal_2d[:n_patches, 1], c='C0', alpha=0.5, label=f'Sample {idx_0} (grade {labels[idx_0]})')
            ax_n.set_title(f'Memory Bank + Normal idx {idx_0}')
            ax_n.set_xlabel('Dimension 1')
            ax_n.set_ylabel('Dimension 2')
            ax_n.legend()

            # Row 2: memory bank + anomalous sample only
            anom_with_mem = np.concatenate([embeddings[idx_1], memory_bank], axis=0)
            tsne_anom = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42)
            anom_2d = tsne_anom.fit_transform(anom_with_mem)
            ax_a = axes[2, pair_count]
            ax_a.scatter(anom_2d[n_patches:, 0], anom_2d[n_patches:, 1], c='red', alpha=0.15, label='Memory Bank')
            ax_a.scatter(anom_2d[:n_patches, 0], anom_2d[:n_patches, 1], c='C1', alpha=0.5, label=f'Sample {idx_1} (grade {labels[idx_1]})')
            ax_a.set_title(f'Memory Bank + Anomalous idx {idx_1}')
            ax_a.set_xlabel('Dimension 1')
            ax_a.set_ylabel('Dimension 2')
            ax_a.legend()

            # Row 3: images
            ax_img = axes[3, pair_count]
            img_0 = np.array(Image.open(image_paths[idx_0]).convert('RGB'))
            img_1 = np.array(Image.open(image_paths[idx_1]).convert('RGB'))
            if img_0.shape[0] != 224 or img_0.shape[1] != 224:
                img_0 = np.array(Image.fromarray(img_0).resize((224, 224)))
            if img_1.shape[0] != 224 or img_1.shape[1] != 224:
                img_1 = np.array(Image.fromarray(img_1).resize((224, 224)))
            ax_img.imshow(np.concatenate([img_0, np.ones((224, 10, 3), dtype=np.uint8) * 255, img_1], axis=1))
            ax_img.axis('off')
            path_0 = os.path.basename(image_paths[idx_0]).split("_")
            path_1 = os.path.basename(image_paths[idx_1]).split("_")
            ax_img.set_title(f'{idx_0},path:{path_0[0]}_{path_0[1]}|{idx_1},path:{path_1[0]}_{path_1[1]}')

            pair_count += 1

    plt.tight_layout()
    plt.savefig('tsne_patch_pairs_grid.png')
    plt.close(fig)


def cal_img_roc(img_scores: np.ndarray, gt_list: list) -> tuple[float, float, float]:
    """
    Calculate image-level roc auc score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_list (list)    : list of ground truth labels

    Returns:
        fpr (float)     : false positive rate
        tpr (float)     : true positive rate
        img_roc (float) : img roc auc score
    """

    # for every image in the batch take the max pixel anomaly score

    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)

    return fpr, tpr, img_roc_auc


def cal_pxl_roc(gt_mask: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    """
    Calculate pixel-level roc auc score

    Args:
        gt_mask (np.array) : numpy array of ground truth masks
        scores (np.array)  : numpy array of predicted masks

    Returns:
        fpr (float)     : false positive rate
        tpr (float)     : true positive rate
        img_roc (float) : pixel roc auc score
    """

    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    return fpr, tpr, per_pixel_rocauc


def cal_f1_img(img_scores: np.ndarray, gt_list: list) -> float:
    """
    Calculate image-level f1 score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_list (list)    : list of ground truth labels

    Returns:
        f1 (float)     : f1 score image level
    """

    gt_list = np.asarray(gt_list)

    precision, recall, thresholds = precision_recall_curve(gt_list, img_scores)
    a = 2 * precision * recall
    b = precision + recall

    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    print("Optimal threshold :" ,thresholds[np.argmax(f1)])

    return np.max(f1), thresholds[np.argmax(f1)]


def cal_f1_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Calculate image-level f1 score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_masks (list)   : list of ground truth masks

    Returns:
        f1 (float)     : f1 score pixel level
    """
    gt_masks = np.asarray(gt_masks)

    precision, recall, _ = precision_recall_curve(gt_masks.flatten(), scores.flatten())

    a = 2 * precision * recall
    b = precision + recall

    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    return np.max(f1)


def cal_pr_auc_img(scores: np.ndarray, gt_list: list) -> float:
    """
    Calculate image-level pr auc score

    Args:
        scores (np.array) : numpy array of shape (b 1 h w) with the pixel level anomaly scores
        gt_list (list)    : list of ground truth labels

    Returns:
        pr_auc_img (float)     : pr auc score image level
    """

    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)

    return average_precision_score(gt_list, img_scores)


def cal_pr_auc_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:
    """
    Calculate pixel-level pr auc score

    Args:
        scores (np.array)  : numpy array of predicted masks
        gt_mask (np.array) : numpy array of ground truth masks

    Returns:
        pr_auc_pxl (float) : pro_auc pixel level score
    """

    gt_masks = np.asarray(gt_masks)

    return average_precision_score(gt_masks.flatten(), scores.flatten())


def cal_pro_auc_pxl(scores: np.ndarray, gt_masks: np.ndarray) -> float:
    def rescale(x):
        return (x - x.min()) / (x.max() - x.min())

    """
    Calculate pixel-level pro auc score

    Args:
        scores (np.array)  : numpy array of predicted masks
        gt_mask (np.array) : numpy array of ground truth masks

    Returns:
        per_pixel_roc_auc (float) : pro_auc pixel level score
    """

    # remove the channel dimension
    gt = np.squeeze(gt_masks, axis=1)

    gt[gt <= 0.5] = 0
    gt[gt > 0.5] = 1
    gt = gt.astype(np.bool_)

    max_step = 200
    expect_fpr = 0.3

    # set the max and min scores and the delta step
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    pros_mean = []
    threds = []
    fprs = []

    binary_score_maps = np.zeros_like(scores, dtype=np.bool_)

    for step in range(max_step):
        thred = max_th - step * delta

        # segment the scores with different thresholds
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []
        for i in range(len(binary_score_maps)):

            # label the regions in the ground truth
            label_map = label(gt[i], connectivity=2)

            # calculate some properties for every corresponding region
            props = regionprops(label_map, binary_score_maps[i])

            # calculate the per-regione overlap
            for prop in props:
                pro.append(prop.intensity_image.sum() / prop.area)

        # append the per-region overlap
        pros_mean.append(np.array(pro).mean())

        # calculate the false positive rate
        gt_neg = ~gt
        fpr = np.logical_and(gt_neg, binary_score_maps).sum() / gt_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    fprs = np.array(fprs)

    # select the case when the false positive rates are under the expected fpr
    idx = fprs <= expect_fpr

    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)
    pros_mean_selected = rescale(pros_mean[idx])
    per_pixel_roc_auc = auc(fprs_selected, pros_mean_selected)

    return per_pixel_roc_auc


def save_anomaly_map(dirpath, anomaly_map, pred_score, filepath, x_type, mask):
    """
    Args:
        dirpath     (str)       : Output directory path.
        anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
        filepath    (str)       : Path of the input image.
        x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
        contour     (float)     : Threshold of contour, or None.
    """
    def min_max_norm(image):
        a_min, a_max = image.min(), image.max()
        return (image - a_min) / (a_max - a_min)

    def cvt2heatmap(gray):
        return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

    # Get the image file name.
    filename = os.path.basename(filepath)

    # Load the image file and resize.
    original_image = cv.imread(filepath)
    original_image = cv.resize(original_image, anomaly_map.shape[:2])

    # Normalize anomaly map for easier visualization.
    anomaly_map_norm = cvt2heatmap(255 * min_max_norm(anomaly_map))

    # Overlay the anomaly map to the origimal image.
    output_image = anomaly_map_norm.astype(np.uint8)#anomaly_map_norm.astype(np.uint8) #(anomaly_map_norm / 2 + original_image / 2).astype(np.uint8) 

    # Create a figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    #convert the images to RGB
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

    # Display the input image
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image {x_type}')
    axes[0].axis('off')

    # Display the mask image
    axes[1].imshow(mask.squeeze(), cmap ='gray')
    axes[1].set_title(f'Mask')
    axes[1].axis('off')

    # Display the final image
    axes[2].imshow(output_image)
    axes[2].set_title(f'Heatmap {pred_score}')
    axes[2].axis('off')

    # Show the plot
    plt.savefig(str(dirpath + f"/{x_type}_{pred_score:.4f}_{filename}"))



def save_imgs(dirpath, pred_score, filepath, x_type, mask):
    """
    Same as save anomaly map, just without the anomaly map.
    For models that do not produce an anomaly map, e.g. SINBAD
    
    Args:
        dirpath     (str)       : Output directory path.
        pred_score  (float)     : Predicted anomaly score.
        filepath    (str)       : Path of the input image.
        x_type      (str)       : Label string (e.g. "0_PRED_0", "1_PRED_1").
        mask        (np.ndarray): Segmentation mask.
    """
    filename = os.path.basename(filepath)

    original_image = cv.imread(filepath)
    original_image = cv.resize(original_image, (224, 224))

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)

    axes[0].imshow(original_image)
    axes[0].set_title(f'Original Image {x_type}')
    axes[0].axis('off')

    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title(f'Mask | Score: {pred_score:.4f}')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(str(dirpath + f"/{x_type}_{pred_score:.4f}_{filename}"))
    plt.close()