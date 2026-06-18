from __future__ import annotations
import os
from typing import Union, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import torch

from ..utilities.metrics import *


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import umap
from time import time

def _cls_one_class_analysis(test_cls_last, train_cls, binary_grades, grades, grade_colors, output_prefix='cls'):
    """
    Visualize how well grades 1-5 can be separated in CLS-token space using
    Mahalanobis (LedoitWolf) and kNN-cosine distances from training normals.

    Produces a single 2×2 figure saved to {output_prefix}_grade_separability.png:
      (0,0) PCA scatter of test tokens colored by grade (train as gray background)
      (0,1) Pairwise cosine-distance heatmap between per-grade centroids
      (1,0) Mahalanobis score violin+strip plot per grade
      (1,1) kNN cosine distance violin+strip plot per grade

    Args:
        test_cls_last:  (N_test, dim)  — last-layer CLS tokens from test set
        train_cls:      (N_train, dim) — last-layer CLS tokens from training normals
        binary_grades:  unused (kept for API compatibility)
        grades:         (N_test,) int  — grade labels 1-5
        grade_colors:   dict {grade: color}
        output_prefix:  filename prefix for saved plot
    """
    train_np   = np.asarray(train_cls,      dtype=np.float64)
    test_np    = np.asarray(test_cls_last,  dtype=np.float64)
    grades_arr = np.asarray(grades)
    unique_grades = sorted(np.unique(grades_arr).tolist())

    # ── Mahalanobis (LedoitWolf) ──────────────────────────────────────
    lw = LedoitWolf(assume_centered=False)
    lw.fit(train_np)
    delta = test_np - lw.location_
    maha_scores = np.einsum('ij,jk,ik->i', delta, lw.precision_, delta)
    # clip p99 for readability
    maha_scores_plot = np.clip(maha_scores, None, np.percentile(maha_scores, 99))

    # ── kNN cosine distance (k=5) ─────────────────────────────────────
    train_n = train_np / (np.linalg.norm(train_np, axis=1, keepdims=True) + 1e-8)
    test_n  = test_np  / (np.linalg.norm(test_np,  axis=1, keepdims=True) + 1e-8)
    sim = test_n @ train_n.T
    k = min(5, train_n.shape[0])
    knn_scores = 1.0 - np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)

    # ── PCA 2D ────────────────────────────────────────────────────────
    pca = PCA(n_components=2)
    pca.fit(np.vstack([train_np, test_np]))
    test_pca  = pca.transform(test_np)
    train_pca = pca.transform(train_np)

    # ── Cosine distance between per-grade centroids ───────────────────
    centroids = {}
    for g in unique_grades:
        v = test_n[grades_arr == g].mean(axis=0)
        centroids[g] = v / (np.linalg.norm(v) + 1e-8)
    n_g = len(unique_grades)
    dist_mat = np.array([[1.0 - centroids[g1] @ centroids[g2]
                          for g2 in unique_grades] for g1 in unique_grades])

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) PCA scatter
    ax = axes[0, 0]
    ax.scatter(train_pca[:, 0], train_pca[:, 1],
               c='lightgray', s=12, alpha=0.35, label='Train normals', zorder=1)
    for g in unique_grades:
        mask = grades_arr == g
        ax.scatter(test_pca[mask, 0], test_pca[mask, 1],
                   c=grade_colors[g], s=22, alpha=0.7,
                   label=f'Grade {g} (n={mask.sum()})', zorder=2)
    for g in unique_grades:
        mask = grades_arr == g
        cx, cy = test_pca[mask, 0].mean(), test_pca[mask, 1].mean()
        ax.scatter(cx, cy, c=grade_colors[g], s=200, marker='*',
                   edgecolors='black', linewidths=1.2, zorder=3)
    ax.set_title(f'PCA — test CLS tokens  (var={pca.explained_variance_ratio_.sum():.1%})', fontsize=12)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=8, markerscale=1.4)

    # (0,1) Centroid cosine-distance heatmap
    ax = axes[0, 1]
    im = ax.imshow(dist_mat, cmap='RdYlGn_r', vmin=0, vmax=dist_mat.max())
    plt.colorbar(im, ax=ax, fraction=0.046, label='Cosine distance')
    tick_labels = [f'G{g}' for g in unique_grades]
    ax.set_xticks(range(n_g)); ax.set_xticklabels(tick_labels)
    ax.set_yticks(range(n_g)); ax.set_yticklabels(tick_labels)
    for i in range(n_g):
        for j in range(n_g):
            ax.text(j, i, f'{dist_mat[i, j]:.3f}', ha='center', va='center', fontsize=9)
    ax.set_title('Per-grade centroid cosine distances\n(lower = more similar)', fontsize=12)

    # (1,0) Mahalanobis violin + strip per grade
    ax = axes[1, 0]
    data_maha = [maha_scores_plot[grades_arr == g] for g in unique_grades]
    parts = ax.violinplot(data_maha, positions=unique_grades, showmedians=True, showextrema=False)
    for pc, g in zip(parts['bodies'], unique_grades):
        pc.set_facecolor(grade_colors[g]); pc.set_alpha(0.65)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    rng = np.random.default_rng(0)
    for g, d in zip(unique_grades, data_maha):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.full(len(d), g) + jitter, d, c=grade_colors[g], s=8, alpha=0.35, zorder=2)
    ax.set_xticks(unique_grades); ax.set_xticklabels([f'Grade {g}' for g in unique_grades])
    ax.set_title('Mahalanobis (LW) score per grade', fontsize=12)
    ax.set_ylabel('Score  (↑ = further from train normals)')

    # (1,1) kNN cosine violin + strip per grade
    ax = axes[1, 1]
    data_knn = [knn_scores[grades_arr == g] for g in unique_grades]
    parts = ax.violinplot(data_knn, positions=unique_grades, showmedians=True, showextrema=False)
    for pc, g in zip(parts['bodies'], unique_grades):
        pc.set_facecolor(grade_colors[g]); pc.set_alpha(0.65)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    for g, d in zip(unique_grades, data_knn):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.full(len(d), g) + jitter, d, c=grade_colors[g], s=8, alpha=0.35, zorder=2)
    ax.set_xticks(unique_grades); ax.set_xticklabels([f'Grade {g}' for g in unique_grades])
    ax.set_title(f'kNN cosine distance (k={k}) per grade', fontsize=12)
    ax.set_ylabel('Score  (↑ = further from train normals)')

    fig.suptitle('CLS Token Grade Separability  (last layer)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_grade_separability.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved grade separability plot → {output_prefix}_grade_separability.png")


def visualize_cls_tokens(all_cls_tokens, all_actual_grades, layers_idx=[3, 6], training_cls_tokens=None):
    """
    all_cls_tokens:       np.array (N_test, num_layers, dim_per_layer) — test CLS tokens
    all_actual_grades:    np.array (N_test,) — grade labels 1-5
    layers_idx:           which layers the CLS tokens came from
    training_cls_tokens:  np.array (N_train, dim) — last-layer CLS tokens from training normals
                          (pass model.cls_memory_bank.cpu().numpy()); enables Mahalanobis/kNN analysis
    """
    grades = np.asarray(all_actual_grades)
    grade_colors = {1: '#2ecc71', 2: '#3498db', 3: '#f39c12', 4: '#e74c3c', 5: '#8e44ad'}
    binary_colors = {0: '#2ecc71', 1: '#e74c3c'}  # 0 = normal, 1 = anomalous
    binary_labels = {0: 'Normal (1–3)', 1: 'Anomalous (4–5)'}
    binary_grades = np.where(np.isin(grades, [1, 2, 3]), 0, 1)

    # Per-layer tokens + concatenated version
    keys = list(layers_idx) + ['concat']
    layer_cls = {}
    for i, layer_id in enumerate(layers_idx):
        layer_cls[layer_id] = all_cls_tokens[:, i, :]  # (N, dim)
    layer_cls['concat'] = normalize(
        all_cls_tokens.reshape(all_cls_tokens.shape[0], -1), norm='l2'
    )

    ncols = min(len(keys), 3)
    nrows = (len(keys) + ncols - 1) // ncols

    # Compute all projections once, reuse for both figures
    projections = {}
    for key in keys:
        tokens = layer_cls[key]
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        projections[key] = reducer.fit_transform(tokens)

    def make_figure(coloring, color_map, label_map, suptitle, filename):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.atleast_2d(axes).flatten()

        for idx, key in enumerate(keys):
            proj = projections[key]
            ax = axes[idx]

            for g in sorted(color_map.keys()):
                mask = coloring == g
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=color_map[g], label=label_map[g],
                           s=15, alpha=0.6, edgecolors='none')

            title = f'Layer {key}' if isinstance(key, int) else 'All layers concat'
            ax.set_title(title, fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.legend(fontsize=9, markerscale=2)

        for idx in range(len(keys), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(suptitle, fontsize=15)
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)

    make_figure(
        coloring=grades,
        color_map=grade_colors,
        label_map={g: f'Grade {g}' for g in grade_colors},
        suptitle='DINOv2 CLS Token UMAP — Per Layer (5 grades)',
        filename='cls_umap_per_layer.png',
    )

    make_figure(
        coloring=binary_grades,
        color_map=binary_colors,
        label_map=binary_labels,
        suptitle='DINOv2 CLS Token UMAP — Per Layer (Normal vs Anomalous)',
        filename='cls_umap_per_layer_binary.png',
    )

    # --- One-class classification on CLS tokens (Mahalanobis + kNN) ---
    if training_cls_tokens is not None:
        print("Computing CLS one-class scores (Mahalanobis + kNN)...")
        # last-layer test tokens: index -1 corresponds to the last entry in layers_idx
        test_cls_last = all_cls_tokens[:, -1, :]   # (N_test, dim)
        train_cls_np  = np.asarray(training_cls_tokens, dtype=np.float64)
        _cls_one_class_analysis(
            test_cls_last, train_cls_np,
            binary_grades, grades,
            grade_colors,
            output_prefix='cls',
        )



def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


class Evaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """

    def __init__(self, test_dataloader, device):
        """
        Args:
            test_dataloader (Dataloader): test dataloader, the images should already be normalized
            device (torch.device): device where to run the model
        """
        self.test_dataloader = test_dataloader
        self.device = device

    def evaluate(self, model):
        """
        Args:
            model: a model object on which you can call model.predict(batched_images)
                and returns a tuple of anomaly_maps and anomaly_scores
            output_path (str): path where to store the output masks
        """

        model.eval()

        # Initialize results.
        gt_masks_list, true_img_scores = (list(), list())
        pred_masks, pred_img_scores = (list(), list())
        all_embeddings = list() # NOTE : added
        all_grades = list() # NOTE : added
        all_actual_grades = list() # NOTE : added
        all_image_paths = list() # NOTE : added
        all_cls_tokens = list() # NOTE : added
        test_time_full = 0
    

        
        for images, labels, masks, path, full_mask, actual_grade, mask_unfiltered, og_img, og_mask, og_depth in tqdm(self.test_dataloader, desc="Eval"):
            # get anomaly map and score
            with torch.no_grad():
                

                test_start_time = time()
                out = model((images.to(self.device), full_mask.to(self.device), mask_unfiltered.to(self.device), og_img.to(self.device), og_mask.to(self.device), og_depth.to(self.device))) # NOTE : added full_mask and depth
                test_end_time = time()
                batch_time = test_end_time - test_start_time
                test_time_full += batch_time

                if len(out) == 5: # NOTE : added for patchcore currently
                    anomaly_maps, anomaly_scores, embeddings, memory_bank, cls_tokens = out
                elif len(out) == 1: # NOTE : currently for SINBAD
                    anomaly_scores = out[0]
                else:
                    anomaly_maps, anomaly_scores = out
              #  print(embeddings.shape) # NOTE : added
              #  print(labels.shape) # NOTE : added

                # Split the embeddings such that it is clear which embedding belongs to which image. Assuming each image is split into 28x28 patches, we can split the embeddings into blocks of 784 (28x28) to get the embeddings for each image.
              #  if len(out) == 4: # NOTE : added for patchcore currently
                #    batch_size = images.shape[0]
                 #   num_patches_per_image = 28 * 28
                 #   embeddings = embeddings.view(batch_size, num_patches_per_image, -1) # shape: (batch_size, num_patches_per_image, embedding_dim)
                 #   all_embeddings.extend(embeddings.cpu().numpy()) # NOTE : added

                all_grades.extend(labels.cpu().numpy()) # NOTE : added
                all_image_paths.extend(path) # NOTE : added
                all_actual_grades.extend(actual_grade.cpu().numpy()) # NOTE : added
                if len(out) == 5 and cls_tokens is not None: # NOTE : added for patchcore currently
                    all_cls_tokens.extend(cls_tokens.cpu().numpy()) # NOTE : added




            if len(out) > 1: # Added for SINBAD, since we have no anomaly maps 
                if anomaly_maps.shape[2:] != masks.shape[2:]:
                    raise Exception(
                        "The output anomaly maps should have the same resolution as the target masks."
                        + f"Expected shape: {masks.shape}, got: {anomaly_maps.shape}"
                    )
            



            # add true masks and img anomaly scores
            gt_masks_list.extend(masks.cpu().numpy().astype(int))
            true_img_scores.extend(labels.cpu().numpy())


            if len(out) > 1: # Added for SINBAD, since we have no anomaly maps
                # add predicted masks and img anomaly scores (check for numpy arrays or tensors)
                if isinstance(anomaly_maps, torch.Tensor):
                    pred_masks.extend(anomaly_maps.cpu().numpy())
                    if len(anomaly_scores.size()) != 0:
                        pred_img_scores.extend(anomaly_scores.cpu().numpy())
                    else:
                        # since possible that we get a torch.size([]) for the anomaly score, we need to convert it to a numpy array and then get the item (.extend doesn't work directly here)
                        pred_img_scores.append(anomaly_scores.item())
                else:
                    pred_masks.extend(anomaly_maps)
                    pred_img_scores.extend(anomaly_scores)
            else:  # SINBAD: no anomaly maps, only scores
                if isinstance(anomaly_scores, torch.Tensor):
                    pred_img_scores.extend(anomaly_scores.cpu().numpy())
                else:
                    pred_img_scores.extend(anomaly_scores)

       # if len(out) == 4:
          #  all_embeddings = np.asarray(all_embeddings) # NOTE : added
       #     memory_bank = memory_bank.cpu().numpy() # NOTE : added

        
        print(f"Average Inference time per Raspberry: {test_time_full / len(pred_img_scores):.6f} seconds")

        all_grades = np.asarray(all_grades) # NOTE : added
        all_actual_grades = np.asarray(all_actual_grades) # NOTE : added
        if len(out) == 5 and cls_tokens is not None: # NOTE : added for patchcore currently
            all_cls_tokens = np.asarray(all_cls_tokens) # NOTE : added
          #  print(all_cls_tokens.shape) # NOTE : added
          #  training_cls = model.cls_memory_bank.cpu().numpy() if hasattr(model, 'cls_memory_bank') and model.cls_memory_bank is not None else None
          #  visualize_cls_tokens(all_cls_tokens, all_actual_grades, layers_idx=[3,6,11], training_cls_tokens=training_cls)
   

      #  if len(out) == 4:
            # Average the embeddings for each image : Given that the embeddings are in the shape (num_images, num_patches_per_image, embedding_dim), we can average the embeddings for each image by taking the mean over the num_patches_per_image dimension.
         #   image_embeddings = np.mean(all_embeddings, axis=1) # shape: (num_images, embedding_dim)

        # print(f"Image Embeddings shape: {image_embeddings.shape}, Grades shape: {all_grades.shape}") # NOTE : added
         #   visualize_embeddings_tsne(image_embeddings, all_actual_grades, plot_3d = False)
           # visualize_patch_embeddings_pairs_tsne(all_embeddings, all_grades, all_image_paths, memory_bank)

        gt_masks_list = np.asarray(gt_masks_list)
        true_img_scores = np.asarray(true_img_scores)
        pred_masks = np.asarray(pred_masks)
        pred_img_scores = np.asarray(pred_img_scores)

        # Added for SINBAD, since we have no anomaly maps
        if len(out) > 1:
            pred_masks = min_max_norm(pred_masks)

        """Image-level AUROC"""
        fpr, tpr, img_roc_auc = cal_img_roc(pred_img_scores, true_img_scores)

        """Pixel-level AUROC"""
        fpr, tpr, pxl_roc_auc = 0,0,0#cal_pxl_roc(gt_masks_list, pred_masks)

        """F1 Score Image-level"""
        img_f1, opt_thresh= cal_f1_img(pred_img_scores, true_img_scores) #  opt_thresh 

        """F1 Score Pixel-level"""
        pxl_f1 = 0#cal_f1_pxl(pred_masks, gt_masks_list)

        """Image-level PR-AUC"""
        img_pr_auc = cal_pr_auc_img(pred_img_scores, true_img_scores)

        """Pixel-level PR-AUC"""
        pxl_pr_auc = 0#cal_pr_auc_pxl(pred_masks, gt_masks_list)

        """Pixel-level AU-PRO"""
        pxl_au_pro = 0#cal_pro_auc_pxl(np.squeeze(pred_masks, axis=1), gt_masks_list)

        # TODO: Implement Add False-alarm rate

        metrics = {
            "img_roc_auc": img_roc_auc,
            "pxl_roc_auc": pxl_roc_auc,
            "img_f1": img_f1,
            "pxl_f1": pxl_f1,
            "img_pr_auc": img_pr_auc,
            "pxl_pr_auc": pxl_pr_auc,
            "pxl_au_pro": pxl_au_pro
        }

        return metrics, opt_thresh

           
    def evaluate_single_images(self, model):

        model.eval()

        # compute the threshold as equal precision and recall on the test dataset
        pred_anom_score_lst, true_anom_score_lst = [], []
        pred_anom_map_lst, gt_anom_mask_lst = [], []
        allpaths = []
        for images, labels, masks, paths in tqdm(self.test_dataloader):
            with torch.no_grad():
                anomaly_maps, anomaly_scores = model(images.to(self.device))

            if isinstance(anomaly_maps, torch.Tensor):
                anomaly_maps = anomaly_maps.cpu().numpy()
                anomaly_scores = anomaly_scores.cpu().numpy()

            gt_masks_list = masks.cpu().numpy().astype(int)
            true_img_scores = labels.cpu().numpy()

            pred_anom_score_lst.extend(anomaly_scores)
            true_anom_score_lst.extend(true_img_scores)
            pred_anom_map_lst.extend(anomaly_maps)
            gt_anom_mask_lst.extend(gt_masks_list)
            allpaths.extend(paths)

        pred_anom_score_lst = np.asarray(pred_anom_score_lst)
        true_anom_score_lst = np.asarray(true_anom_score_lst)
        pred_anom_map_lst = np.asarray(pred_anom_map_lst)
        gt_anom_mask_lst = np.asarray(gt_anom_mask_lst)

        pred_anom_map_lst = min_max_norm(pred_anom_map_lst)

        # the threshold is the value that minimizes the difference between precision and recall
        precision, recall, thresholds = precision_recall_curve(
            gt_anom_mask_lst.flatten(), pred_anom_map_lst.flatten()
        )
        threshold = thresholds[np.argmin(np.abs(precision - recall))]

        pred_mask_lst = (pred_anom_map_lst > threshold).astype(int)

        print(
            len(pred_anom_score_lst),
            len(true_anom_score_lst),
            len(pred_mask_lst),
            len(gt_anom_mask_lst),
            len(allpaths),
        )

        metrics = []
        for pred_anom_score, true_anom_score, pred_anom_mask, gt_anom_mask, path in zip(
            pred_anom_score_lst,
            true_anom_score_lst,
            pred_mask_lst,
            gt_anom_mask_lst,
            allpaths,
        ):
            gt_anom_mask = gt_anom_mask.flatten()
            pred_anom_mask = pred_anom_mask.flatten()

            precision = precision_score(gt_anom_mask, pred_anom_mask, zero_division=0)
            recall = recall_score(gt_anom_mask, pred_anom_mask, zero_division=0)
            f1 = f1_score(gt_anom_mask, pred_anom_mask, zero_division=0)

            false_alarm_rate = np.sum(
                (gt_anom_mask == 0) & (pred_anom_mask == 1)
            ) / np.sum(gt_anom_mask == 0)

            metrics.append(
                {
                    "pred_anom_score": pred_anom_score,
                    "true_anom_score": true_anom_score,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "false_alarm_rate": false_alarm_rate,
                    "path": path,
                }
            )
        metrics = pd.DataFrame(metrics)

        return metrics, threshold, gt_anom_mask_lst, pred_anom_map_lst


    @staticmethod
    def get_threshold(gt: np.ndarray, score: np.ndarray) -> float:
        """
        Calculate the segmentation threshold

        Args:
            gt (np.array)    : ground truth masks
            score (np.array) : predicted masks

        Returns:
            threshold (float) : segmentation threshold
        """

        gt_mask = np.asarray(gt)
        precision, recall, thresholds = precision_recall_curve(
            gt_mask.flatten(), score.flatten()
        )
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)

        # consider the threshold with the highest f1 score
        threshold = thresholds[np.argmax(f1)]

        return threshold
    
def append_results(
    output_path: Union[str, os.PathLike],
    category: str,
    seed: Optional[int],
    img_roc_auc: float,
    per_pixel_rocauc: float,
    f1_img: float,
    f1_pxl: float,
    pr_auc_img: float,
    pr_auc_pxl: float,
    au_pro_pxl: float,
    ad_model: str,
    feature_layers: str,
    backbone: str,
    weights: Optional[str],
    bootstrap_layer: Optional[int],
    epochs: Optional[int],
    input_img_size: Optional[tuple[int, int]],
    output_img_size: Optional[tuple[int, int]],
):
    """
    Save the results of the evaluation in a file
    """
    df = pd.DataFrame(
        {
            "category": [category],
            "seed": [seed],
            "img_roc_auc": [img_roc_auc],
            "per_pixel_rocauc": [per_pixel_rocauc],
            "f1_img": [f1_img],
            "f1_pxl": [f1_pxl],
            "pr_auc_img": [pr_auc_img],
            "pr_auc_pxl": [pr_auc_pxl],
            "au_pro_pxl": [au_pro_pxl],
            "ad_model": [ad_model],
            "feature_layers": [feature_layers],
            "backbone": [backbone],
            "weights": [weights],
            "eval_datetime": [pd.Timestamp.now()],
            "bootstrap_layer": [bootstrap_layer],
            "epochs": [epochs],
        }
    )
    if os.path.isfile(output_path):
        old_df = pd.read_csv(output_path)
        df = pd.concat([old_df, df], ignore_index=True)
    df.to_csv(output_path, index=False)
