"""


Key differences from the original SINBAD code:
- No multi-crop ensemble (raspberries are centered & segmented)
- Single pyramid level per run (multi-level ensembling can be done externally, right now NOT implemented)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sklearn.covariance import ShrunkCovariance



class CumulativeSetFeatures:
    """
    Computes set-level descriptors via random projection + CDF histograms.

    Given a set of patch features for an image [C, P], projects them into
    n_projections random directions, then builds a CDF histogram with
    n_quantiles bins along each projection. The result is a fixed-length
    descriptor of size (n_projections * n_quantiles) per image, regardless
    of how many patches P the image has.

    This is the core of SINBAD: it captures the *distribution* of patch
    features rather than individual patch anomaly scores.

    Input shape:  [N, C, P]  — N images, C channels, P patches (set elements)
    Output shape: [N, n_projections * n_quantiles]
    """

    def __init__(self, n_channels: int, n_projections: int = 1000, n_quantiles: int = 5):
        self.n_channels = n_channels
        self.n_projections = n_projections
        self.n_quantiles = n_quantiles
        # Random projection matrix as conv1d weight: [out_channels, in_channels, kernel=1]
        self.projections = torch.randn(self.n_projections, self.n_channels, 1)
        self.min_vals = None
        self.max_vals = None

    def fit(self, X: torch.Tensor):
        """
        Compute quantile thresholds from training data.

        Args:
            X: [N, C, P] training patch features
        """
        # Project: [N, n_proj, P]
        a = F.conv1d(X, self.projections)
        # Reshape to [N*P, n_proj] for quantile computation
        a = a.permute(0, 2, 1).reshape(-1, self.n_projections)
        self.min_vals = torch.quantile(a, 0.01, dim=0)
        self.max_vals = torch.quantile(a, 0.99, dim=0)

    def forward(self, X: torch.Tensor) -> np.ndarray:
        """
        Compute set-level descriptors.

        Args:
            X: [N, C, P] patch features

        Returns:
            cdf: [N, n_projections * n_quantiles] set descriptors as numpy array
        """
        assert self.min_vals is not None, "Call .fit() before .forward()"

        # Project: [N, n_proj, P]
        a = F.conv1d(X, self.projections)

        N = a.shape[0]
        cdf = torch.zeros(N, self.n_projections, self.n_quantiles)

        for q in range(self.n_quantiles):
            threshold = self.min_vals + (self.max_vals - self.min_vals) * (q + 1) / (self.n_quantiles + 1) # Calculates a threshold for EACH projection ! (and with e.g. 5 quantiles we then have 5 lists of len (n_projection), such that we have quantiles for each projection)
            # threshold: [n_proj] -> [1, n_proj, 1] for broadcasting
            # a < threshold: [N, n_proj, P] boolean -> mean over P gives CDF value
            cdf[:, :, q] = (a < threshold.unsqueeze(0).unsqueeze(2)).float().mean(2) # adds the fraction of values that fall in this current histogram "bin" (for each projection)

        cdf = cdf.reshape(N, -1).numpy()
        return cdf

class MahalanobisScorer:
    """
    Mahalanobis-distance scorer with shrunk covariance.

    Fits a Gaussian on training descriptors, scores test descriptors
    by Mahalanobis distance to the training mean. Uses shrunk covariance
    for regularization (essential when descriptor dim >> n_samples).

    This replaces the original SINBAD's kNN_shrunk + faiss setup with a
    simpler, dependency-light implementation. The original also does
    kNN with k=1 after whitening, which is equivalent to Mahalanobis
    distance to the nearest training sample. We provide both options.
    """

    def __init__(self, shrinkage: float = 0.1, mode: str = 'knn'):
        """
        Args:
            shrinkage: Regularization parameter for ShrunkCovariance
            mode: 'knn' (distance to nearest train sample, like original SINBAD)
                  or 'mean' (distance to training mean, simpler)
        """
        self.shrinkage = shrinkage
        self.mode = mode
        self.cov_inv = None
        self.train_descriptors_whitened = None
        self.train_mean = None

    def fit(self, descriptors: np.ndarray):
        # TODO : fix. Do with faiss as in original
        """
        Fit the scorer on training descriptors.

        Args:
            descriptors: [N_train, D] training set descriptors
        """
        cov = ShrunkCovariance(shrinkage=self.shrinkage).fit(descriptors).covariance_
        try:
            self.cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.cov_inv = np.linalg.pinv(cov)

        self.train_mean = descriptors.mean(axis=0)

        if self.mode == 'knn':
            # Whiten training descriptors for L2-based kNN
            self.train_descriptors_whitened = descriptors @ self.cov_inv #descriptors.dot(self.cov_inv) #
            

    def score(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Score test descriptors.

        Args:
            descriptors: [N_test, D] test set descriptors

        Returns:
            scores: [N_test] anomaly scores (higher = more anomalous)
        """
        if self.mode == 'mean':
            # Mahalanobis distance to training mean
            diff = descriptors - self.train_mean
            # (x - mu)^T Sigma^{-1} (x - mu)
            scores = np.sum(diff @ self.cov_inv * diff, axis=1)
            return scores

        elif self.mode == 'knn':
            # TODO : fix. in OG done via faiss, see if we can also just implement faiss here to match original
            # Distance to nearest training sample in whitened space (like original SINBAD)
            whitened_test = descriptors @ self.cov_inv
            # Brute-force 1-NN in whitened space
            # For ~3000 train × ~1000 test × 5000 dims, this is manageable
            # If memory is an issue, batch this

            # in OG K is still a parameter, but it is set to 1 (i.e. we just search the closest neighbour) (see Appendix in paper)
            scores = np.zeros(len(descriptors))
            batch_size = 128
            for i in range(0, len(descriptors), batch_size):
                batch = whitened_test[i:i + batch_size] # Grab a batch of test descriptors
                # [batch, D] vs [N_train, D] -> [batch, N_train]
                dists = np.sum((batch[:, None, :] - self.train_descriptors_whitened[None, :, :]) ** 2, axis=2)
                
                scores[i:i + batch_size] = dists.min(axis=1)
            return scores

        else:
            raise ValueError(f"Unknown scoring mode: {self.mode}")


'''
class MahalanobisScorer:
    # faiss not working??
    def __init__(self, shrinkage: float = 0.1):
        self.shrinkage = shrinkage
        self.cov_inv = None

    def fit(self, descriptors: np.ndarray):
        cov = ShrunkCovariance(shrinkage=self.shrinkage).fit(descriptors).covariance_
        try:
            self.cov_inv = np.linalg.inv(cov)
        except:
            self.cov_inv = np.linalg.pinv(cov)

        whitened = descriptors.dot(self.cov_inv)
        self.index = faiss.IndexFlatL2(whitened.shape[1])
        self.index.add(np.ascontiguousarray(whitened.astype('float32')))

    def score(self, descriptors: np.ndarray) -> np.ndarray:
        whitened = descriptors.dot(self.cov_inv)
        D, I = self.index.search(np.ascontiguousarray(whitened.astype('float32')), 1)
        return D[:, 0]
'''

class SINBAD(nn.Module):
    """
    SINBAD model for the MoViAD pipeline.

    Architecture:
        1. Feature extraction via CustomFeatureExtractor (shared with PatchCore)
        2. Mask-aware patch selection (only foreground patches)
        3. Random projection + CDF histogram → fixed-length set descriptor per image
        4. Mahalanobis-distance scoring against training descriptors

    This model does NOT produce anomaly maps — it outputs only image-level scores.
    For the thesis, this is a standalone baseline that tests whether distributional
    (set-level) features detect grade 5 anomalies better than patch-level max-aggregation.
    """

    def __init__(
        self,
        device: torch.device,
        input_size: tuple[int, int],
        feature_extractor,
        n_projections: int = 1000,
        n_quantiles: int = 5,
        shrinkage: float = 0.1,
        scoring_mode: str = 'knn',
    ):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.feature_extractor = feature_extractor
        self.n_projections = n_projections
        self.n_quantiles = n_quantiles
        self.shrinkage = shrinkage
        self.scoring_mode = scoring_mode

        # These are set during training
        self.set_feature_extractor: CumulativeSetFeatures | None = None
        self.scorer: MahalanobisScorer | None = None

        # Determined from the first forward pass
        self._n_channels: int | None = None

    def forward(self, input_tensor):
        """
        Training mode: returns patch features [B, C, P] for the trainer to collect.
        Eval mode: returns (None, pred_scores, None, None, None) matching PatchCore output format.

        The None placeholders are for anomaly_maps, embedding, memory_bank, cls_tokens —
        SINBAD doesn't produce these, but the evaluator may expect the tuple structure.
        """
        # Unpack input tuple
        mask = None
        if isinstance(input_tensor, tuple):
            if len(input_tensor) == 2:
                input_tensor, mask = input_tensor
                batch_og, mask_og, depth_og, mask_unfiltered = None, None, None, None
            elif len(input_tensor) == 3:
                input_tensor, mask, mask_unfiltered = input_tensor
                batch_og, mask_og, depth_og = None, None, None
            elif len(input_tensor) == 6:
                input_tensor, mask, mask_unfiltered, batch_og, mask_og, depth_og = input_tensor

        # Extract features
        with torch.no_grad():
            if "dino" in self.feature_extractor.model_name:
                features, cls_tokens = self.feature_extractor(input_tensor.to(self.device))
            else:
                features = self.feature_extractor(input_tensor.to(self.device))

        # Process features into [B, C, P] format
        if isinstance(features, dict):
            features = list(features.values())

        if "dino" in self.feature_extractor.model_name:
            # Per-layer L2 normalization
            features = [F.normalize(f, p=2, dim=1) for f in features]
            embedding = torch.cat(features, dim=1)  # [B, C_total, H, W]
        else:
            # CNN path: resize to common spatial dims
            H_max = max(f.shape[2] for f in features)
            W_max = max(f.shape[3] for f in features)
            resizer = nn.Upsample(size=(H_max, W_max), mode="nearest")
            features = [resizer(f) for f in features]
            embedding = torch.cat(features, dim=1)  # [B, C_total, H, W]

        B, C, H, W = embedding.shape
        self._n_channels = C

        if self.training:
            # Return [B, C, H*W] for trainer to collect
            # Also return mask info so trainer can filter
            patch_features = embedding.reshape(B, C, H * W).cpu()

            return patch_features, None

        else:
            # Eval mode: compute set descriptors and score
            assert self.set_feature_extractor is not None, "Model not trained"
            assert self.scorer is not None, "Model not trained"

            patch_features = embedding.reshape(B, C, H * W).cpu()

            descriptors = self.set_feature_extractor.forward(patch_features)

            # Score
            pred_scores = self.scorer.score(descriptors)
            pred_scores = torch.tensor(pred_scores, dtype=torch.float32)

            # No anomaly maps
            return (pred_scores,)

    def _compute_masked_descriptors(self, patch_features: Tensor, patch_mask: Tensor) -> np.ndarray:
        """
        Compute set descriptors with mask filtering.

        For each image, selects only foreground patches, then computes
        the CDF histogram descriptor. Since images may have different numbers
        of foreground patches, we process them individually.

        Args:
            patch_features: [B, C, P]
            patch_mask: [B, P] boolean

        Returns:
            descriptors: [B, n_projections * n_quantiles] numpy array
        """
        B = patch_features.shape[0]
        D = self.n_projections * self.n_quantiles
        descriptors = np.zeros((B, D))

        for i in range(B):
            fg_mask = patch_mask[i].cpu()  # [P] bool
            fg_features = patch_features[i, :, fg_mask]  # [C, P_fg]

            if fg_features.shape[1] == 0:
                # No foreground patches — score as maximally anomalous
                # (shouldn't happen with proper masks)
                descriptors[i] = 0.0
                continue

            # Add batch dim for conv1d: [1, C, P_fg]
            fg_features = fg_features.unsqueeze(0)
            desc = self.set_feature_extractor.forward(fg_features)  # [1, D]
            descriptors[i] = desc[0]

        return descriptors

    def compute_masked_descriptors_from_collected(
        self,
        all_patch_features: list[Tensor],
        all_patch_masks: list[Tensor | None],
    ) -> np.ndarray:
        """
        Compute set descriptors for a list of collected per-image features.
        Used by the trainer after collecting all training embeddings.

        Args:
            all_patch_features: list of [C, P] tensors (one per image)
            all_patch_masks: list of [P] boolean tensors (or None)

        Returns:
            descriptors: [N, n_projections * n_quantiles]
        """
        N = len(all_patch_features)
        D = self.n_projections * self.n_quantiles
        descriptors = np.zeros((N, D))

        for i in range(N):
            feats = all_patch_features[i]  # [C, P]

            if all_patch_masks[i] is not None:
                fg_mask = all_patch_masks[i]  # [P] bool
                feats = feats[:, fg_mask]  # [C, P_fg]

            if feats.shape[1] == 0:
                continue

            feats = feats.unsqueeze(0)  # [1, C, P_fg]
            desc = self.set_feature_extractor.forward(feats)  # [1, D]
            descriptors[i] = desc[0]

        return descriptors

    def state_dict_sinbad(self):
        """Custom state dict for saving (nn.Module state_dict won't capture our objects)."""
        return {
            'n_projections': self.n_projections,
            'n_quantiles': self.n_quantiles,
            'shrinkage': self.shrinkage,
            'scoring_mode': self.scoring_mode,
            'n_channels': self._n_channels,
            'projections': self.set_feature_extractor.projections if self.set_feature_extractor else None,
            'min_vals': self.set_feature_extractor.min_vals if self.set_feature_extractor else None,
            'max_vals': self.set_feature_extractor.max_vals if self.set_feature_extractor else None,
            'cov_inv': self.scorer.cov_inv if self.scorer else None,
            'train_mean': self.scorer.train_mean if self.scorer else None,
            'train_descriptors_whitened': self.scorer.train_descriptors_whitened if self.scorer else None,
        }

    def load_sinbad(self, state: dict):
        """Load from custom state dict."""
        self._n_channels = state['n_channels']
        self.n_projections = state['n_projections']
        self.n_quantiles = state['n_quantiles']
        self.shrinkage = state['shrinkage']
        self.scoring_mode = state['scoring_mode']

        self.set_feature_extractor = CumulativeSetFeatures(
            self._n_channels, self.n_projections, self.n_quantiles
        )
        self.set_feature_extractor.projections = state['projections']
        self.set_feature_extractor.min_vals = state['min_vals']
        self.set_feature_extractor.max_vals = state['max_vals']

        self.scorer = MahalanobisScorer(self.shrinkage, self.scoring_mode)
        self.scorer.cov_inv = state['cov_inv']
        self.scorer.train_mean = state['train_mean']
        self.scorer.train_descriptors_whitened = state['train_descriptors_whitened']