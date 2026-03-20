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
import umap

def visualize_cls_tokens(all_cls_tokens, all_actual_grades, layers_idx=[3, 6]):
    """
    all_cls_tokens: np.array of shape (num_images, num_layers, dim_per_layer)
    all_actual_grades: np.array of shape (num_images,)
    layers_idx: which layers the CLS tokens came from
    """
    grades = np.asarray(all_actual_grades)
    normal_mask = np.isin(grades, [1, 2, 3])
    grade_colors = {1: '#2ecc71', 2: '#3498db', 3: '#f39c12', 4: '#e74c3c', 5: '#8e44ad'}

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

    # --- UMAP ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, key in enumerate(keys):
        tokens = layer_cls[key]
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        proj = reducer.fit_transform(tokens)

        ax = axes[idx]
        for g in sorted(grade_colors.keys()):
            mask = grades == g
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=grade_colors[g], label=f'Grade {g}',
                       s=15, alpha=0.6, edgecolors='none')

        title = f'Layer {key}' if isinstance(key, int) else 'All layers concat'
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=9, markerscale=2)

    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('DINOv2 CLS Token UMAP — Per Layer', fontsize=15)
    plt.tight_layout()
    plt.savefig('cls_umap_per_layer.png', dpi=200, bbox_inches='tight')
    plt.show()

    # --- Cosine distance from normal centroid ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, key in enumerate(keys):
        tokens = layer_cls[key]
        centroid = tokens[normal_mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cos_dist = 1.0 - tokens @ centroid

        ax = axes[idx]
        for g in sorted(grade_colors.keys()):
            g_mask = grades == g
            ax.hist(cos_dist[g_mask], bins=40, alpha=0.5,
                    color=grade_colors[g], label=f'Grade {g}', density=True)

        title = f'Layer {key}' if isinstance(key, int) else 'All layers concat'
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Cosine distance from normal centroid')
        ax.set_ylabel('Density')
        if idx == 0:
            ax.legend(fontsize=9)

    for idx in range(len(keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('CLS Cosine Distance from Normal Centroid — Per Layer', fontsize=15)
    plt.tight_layout()
    plt.savefig('cls_cosine_dist_per_layer.png', dpi=200, bbox_inches='tight')
   # plt.show()

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
    
        for images, labels, masks, path, full_mask, actual_grade, og_img, og_mask, og_depth in tqdm(self.test_dataloader, desc="Eval"):
            # get anomaly map and score
            with torch.no_grad():

                out = model((images.to(self.device), full_mask.to(self.device), og_img.to(self.device), og_mask.to(self.device), og_depth.to(self.device))) # NOTE : added full_mask and depth

                if len(out) == 5: # NOTE : added for patchcore currently
                    anomaly_maps, anomaly_scores, embeddings, memory_bank, cls_tokens = out
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

            if anomaly_maps.shape[2:] != masks.shape[2:]:
                raise Exception(
                    "The output anomaly maps should have the same resolution as the target masks."
                    + f"Expected shape: {masks.shape}, got: {anomaly_maps.shape}"
                )
            



            # add true masks and img anomaly scores
            gt_masks_list.extend(masks.cpu().numpy().astype(int))
            true_img_scores.extend(labels.cpu().numpy())

            # add predicted masks and img anomaly scores (check for numpy arrays or tensors)
            if isinstance(anomaly_maps, torch.Tensor):
                pred_masks.extend(anomaly_maps.cpu().numpy())
                pred_img_scores.extend(anomaly_scores.cpu().numpy())
            else:
                pred_masks.extend(anomaly_maps)
                pred_img_scores.extend(anomaly_scores)

       # if len(out) == 4:
          #  all_embeddings = np.asarray(all_embeddings) # NOTE : added
       #     memory_bank = memory_bank.cpu().numpy() # NOTE : added

        all_grades = np.asarray(all_grades) # NOTE : added
        all_actual_grades = np.asarray(all_actual_grades) # NOTE : added
        if len(out) == 5 and cls_tokens is not None: # NOTE : added for patchcore currently
            all_cls_tokens = np.asarray(all_cls_tokens) # NOTE : added

            visualize_cls_tokens(all_cls_tokens, all_actual_grades, layers_idx=[3,6])
       # exit()

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

        pred_masks = min_max_norm(pred_masks)

        """Image-level AUROC"""
        fpr, tpr, img_roc_auc = cal_img_roc(pred_img_scores, true_img_scores)

        """Pixel-level AUROC"""
        fpr, tpr, pxl_roc_auc = 0,0,0#cal_pxl_roc(gt_masks_list, pred_masks)

        """F1 Score Image-level"""
        img_f1 = cal_f1_img(pred_img_scores, true_img_scores)

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

        return metrics

           
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
