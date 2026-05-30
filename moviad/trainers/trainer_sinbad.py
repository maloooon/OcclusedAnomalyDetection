"""
Trainer for SINBAD model.

Follows the same pattern as TrainerPatchCore:
1. Extract patch features from training data
2. Fit the model (set descriptors + scorer)
3. Evaluate

Key difference: PatchCore collects all patch embeddings into a flat memory bank.
SINBAD keeps per-image structure — it needs [C, P] per image to build
per-image set descriptors. So we collect a list of per-image features
rather than concatenating into one big tensor.
"""

from __future__ import annotations

import torch
import numpy as np
from tqdm import tqdm
import pickle

from moviad.trainers.trainer import Trainer, TrainerResult



class TrainerSINBAD(Trainer):
    """
    Trainer for the SINBAD model.

    Training procedure:
        1. Extract patch features [B, C, P] from all training images
        2. Apply mask filtering to keep only foreground patches
        3. Fit CumulativeSetFeatures (random projection thresholds) on training patches
        4. Compute per-image set descriptors for all training images
        5. Fit MahalanobisScorer on training descriptors
        6. Evaluate on test set

    No gradient-based training — this is purely fit-and-score.
    """

    def __init__(
        self,
        sinbad_model,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: str,
        save_path=None,
        logger=None,
    ):
        super().__init__(
            sinbad_model,
            train_dataloader,
            test_dataloader,
            device,
            logger,
            save_path,
        )

    def train(self):
        """
        Fit the SINBAD model and evaluate.

        Steps:
            1. Collect per-image patch features + masks from training set
            2. Fit set feature extractor (CDF thresholds)
            3. Compute training set descriptors
            4. Fit Mahalanobis scorer
            5. Evaluate
        """
        # Import here to avoid circular imports — adjust path as needed
        from moviad.models.sinbad.sinbad import CumulativeSetFeatures, MahalanobisScorer

        self.model.train()  # sets self.training = True for forward()

        # ---- Step 1: Collect per-image patch features ----
        all_patch_features = []  # list of [C, P] tensors
        all_patch_masks = []     # list of [P] bool tensors or None

        print("SINBAD: Extracting patch features...")
        with torch.no_grad():
            for batch in tqdm(iter(self.train_dataloader)):
                # batch is (image, mask) tuple from SingleRaspberryDataset
                if isinstance(batch, tuple) and len(batch) == 2:
                    images, masks = batch
                    # Forward in training mode returns (patch_features, patch_mask)
                    # patch_features: [B, C, P], patch_mask: [B, P] or None
                    result = self.model((images, masks))
                    patch_features, patch_mask = result
                else:
                    # No mask available
                    images = batch
                    patch_features, patch_mask = self.model(images)

                B = patch_features.shape[0]
                for b in range(B):
                    all_patch_features.append(patch_features[b])  # [C, P]
                    if patch_mask is not None:
                        all_patch_masks.append(patch_mask[b])  # [P]
                    else:
                        all_patch_masks.append(None)

        n_train = len(all_patch_features)
        n_channels = all_patch_features[0].shape[0]
        print(f"SINBAD: Collected {n_train} images, {n_channels} channels per patch")

        # ---- Step 2: Fit set feature extractor ----
        # We need to fit the CDF thresholds on *all* training patch features.
        # To do this, we need to project all patches and compute quantiles.
        # The original SINBAD fits on the first batch only — we do better by
        # fitting on a large sample across the full training set.

        set_extractor = CumulativeSetFeatures(
            n_channels=n_channels,
            n_projections=self.model.n_projections,
            n_quantiles=self.model.n_quantiles,
        )

        # Collect a representative sample of patches for fitting thresholds.
        # We stack all foreground patches into [1, C, P_total] for fit().
        # If this is too large for memory, subsample.
        print("SINBAD: Fitting CDF thresholds...")
        all_fg_patches = []
        for i in range(n_train):
            feats = all_patch_features[i]  # [C, P]
            if all_patch_masks[i] is not None:
                fg = feats[:, all_patch_masks[i]]  # [C, P_fg]
            else:
                fg = feats
            all_fg_patches.append(fg)

        # Concatenate all foreground patches: [C, P_total]
        all_fg_concat = torch.cat(all_fg_patches, dim=1)  # [C, P_total]

        # Subsample if too many patches (>500k should be fine, but cap at 1M)
        # TODO : see if it works without
      #  max_patches_for_fit = 1_000_000
       # if all_fg_concat.shape[1] > max_patches_for_fit:
       #     perm = torch.randperm(all_fg_concat.shape[1])[:max_patches_for_fit]
       #     all_fg_concat = all_fg_concat[:, perm]
       #     print(f"  Subsampled to {max_patches_for_fit} patches for threshold fitting")

        # fit() expects [N, C, P] — wrap in batch dim
        set_extractor.fit(all_fg_concat.unsqueeze(0))
        print(f"  Thresholds computed (min/max quantiles over {all_fg_concat.shape[1]} patches)")

        # Store on model
        self.model.set_feature_extractor = set_extractor

        # ---- Step 3: Compute training set descriptors ----
        print("SINBAD: Computing training set descriptors...")
        train_descriptors = self.model.compute_masked_descriptors_from_collected(
            all_patch_features, all_patch_masks
        )
        print(f"  Descriptor shape: {train_descriptors.shape}")  # [N_train, n_proj * n_quant]

        # ---- Step 4: Fit Mahalanobis scorer ----
        print("SINBAD: Fitting Mahalanobis scorer...")
        scorer = MahalanobisScorer(
            shrinkage=self.model.shrinkage,
            mode=self.model.scoring_mode,
        )
        scorer.fit(train_descriptors)
        self.model.scorer = scorer

        # Quick sanity check: score training set
        train_scores = scorer.score(train_descriptors)
        print(f"  Training scores — mean: {train_scores.mean():.4f}, "
              f"std: {train_scores.std():.4f}, max: {train_scores.max():.4f}")

        # ---- Step 5: Save if requested ----
    
        if self.save_path:
            save_path_pkl = self.save_path.replace('.pt', '.pkl').replace('.pth', '.pkl')
            if not save_path_pkl.endswith('.pkl'):
                save_path_pkl = save_path_pkl + '.pkl'
            
            state = self.model.state_dict_sinbad()
            with open(save_path_pkl, 'wb') as f:
                pickle.dump(state, f)
            
            # Sanity check
            with open(save_path_pkl, 'rb') as f:
                state_check = pickle.load(f)
            print(f"  Sanity check — n_channels: {state_check['n_channels']}, keys: {list(state_check.keys())}")
            print(f"  Model saved to {save_path_pkl}")


        print("Starting Evaluation...")
        # ---- Step 6: Evaluate ----
        self.model.eval()

        # Move to GPU for feature extraction during eval
        gpu_device = torch.device(self.device)
        self.model.to(gpu_device)
        self.evaluator.device = gpu_device

        metrics = self.evaluator.evaluate(self.model)

        if self.logger is not None:
            self.logger.log(metrics)

        print("SINBAD: End training performances:")
        self.print_metrics(metrics)

        # Cleanup
        del all_patch_features, all_patch_masks, all_fg_patches, all_fg_concat
        del train_descriptors
        torch.cuda.empty_cache()

        return TrainerResult(**metrics)