import torch
from torch import Tensor


class StructCore:
    """
    StructCore: Structure-Aware Image-Level Scoring for Anomaly Detection.

    StructCore augments a base anomaly score (e.g. PatchCore's weighted max score)
    with a structure-aware score derived from a compact 3D descriptor of the
    pixel-level anomaly map. The descriptor captures distributional and spatial
    characteristics of the anomaly map that are invisible to max pooling alone.

    The 3D descriptor is defined as:
        θ(S) = [σ_S, topk_mean_S, TV(S)]
    where:
        σ_S        : standard deviation of all pixel scores (global dispersion)
        topk_mean_S: mean of the top 1% pixel scores (tail concentration)
        TV(S)      : total variation of the map (spatial roughness)

    During training, descriptors and base scores from train-good samples are
    accumulated. After training, fit() computes per-dimension mean and std of
    the descriptor distribution, and an automatic scale-matching weight
    (auto_lambda) between the base score and structural score.

    During inference, a diagonal Mahalanobis distance is computed between the
    test image's descriptor and the fitted normal statistics. The final hybrid
    score is:
        S_hyb = S_base + auto_lambda * D_struct

    """

    def __init__(self, top_k_ratio: float = 0.01):
        """
        Args:
            top_k_ratio (float): Fraction of pixels used for tail concentration.
                                 Default 0.01 means top 1% of pixels, following
                                 the StructCore paper.
        """
        self.top_k_ratio = top_k_ratio

        # Fitted normal statistics — set by fit(), used by score()
        self.mean: Tensor | None = None          # per-dimension mean of train-good descriptors, (3,)
        self.std: Tensor | None = None           # per-dimension std  of train-good descriptors, (3,)
        self.auto_lambda: Tensor | None = None   # scale-matching weight, scalar

        # Accumulation buffers — filled by accumulate(), consumed by fit()
        self._descriptors: list[Tensor] = []     # list of (B, 3) tensors
        self._base_scores: list[Tensor] = []     # list of (B,)  tensors

    def _compute_descriptor(self, anomaly_map: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Compute the structural descriptor θ(S) from an anomaly map.

        Args:
            anomaly_map (Tensor): Pixel-level anomaly map, shape (B, 1, H, W).
            mask (Tensor | None): Optional binary ROI mask, shape (B, 1, H, W) or (B, H, W).
                When provided, all three descriptors (sigma, topk_mean, TV) are computed
                exclusively over mask pixels so that background zeros do not contaminate
                any of the statistics.

        Returns:
            Tensor: Structural descriptor of shape (B, 3).
        """
        B = anomaly_map.shape[0]

        # Remove channel dim: (B, 1, H, W) -> (B, H, W)
        S = anomaly_map.squeeze(1)

        # Flatten for per-image statistics: (B, H*W)
        s_flat = S.reshape(B, -1)
        N = s_flat.shape[1]

        if mask is not None:
            if mask.ndim == 4:
                mask = mask.squeeze(1)                     # (B, H, W)
            mask_sq = (mask > 0).float()                   # (B, H, W)
            mask_flat = mask_sq.reshape(B, -1).bool()      # (B, H*W)
            N_mask = mask_flat.sum(dim=1).float().clamp(min=1)  # (B,)

            sigmas, topk_means = [], []
            for b in range(B):
                s_b = s_flat[b][mask_flat[b]]
                if s_b.numel() == 0:
                    s_b = s_flat[b]
                sigmas.append(s_b.std(correction=0))
                k = max(1, int(s_b.numel() * self.top_k_ratio))
                topk_means.append(torch.topk(s_b, k).values.mean())
            sigma = torch.stack(sigmas)
            topk_mean = torch.stack(topk_means)

            # TV: only adjacent pairs where both pixels are inside the mask
            mask_h = mask_sq[:, 1:, :].bool() & mask_sq[:, :-1, :].bool()
            mask_w = mask_sq[:, :, 1:].bool() & mask_sq[:, :, :-1].bool()
            diff_h = ((S[:, 1:, :] - S[:, :-1, :]).abs() * mask_h.float()).sum(dim=(1, 2))
            diff_w = ((S[:, :, 1:] - S[:, :, :-1]).abs() * mask_w.float()).sum(dim=(1, 2))
            tv = (diff_h + diff_w) / N_mask
        else:
            sigma = s_flat.std(dim=1)
            k = max(1, int(N * self.top_k_ratio))
            topk_vals, _ = torch.topk(s_flat, k, dim=1)
            topk_mean = topk_vals.mean(dim=1)
            diff_h = (S[:, 1:, :] - S[:, :-1, :]).abs().sum(dim=(1, 2))
            diff_w = (S[:, :, 1:] - S[:, :, :-1]).abs().sum(dim=(1, 2))
            tv = (diff_h + diff_w) / N


        # sigma, topk_mean, tv
        return torch.stack([sigma, topk_mean, tv], dim=1)

    def accumulate(self, anomaly_map: Tensor, base_score: Tensor, mask: Tensor | None = None) -> None:
        """
        Accumulate structural descriptors and base scores from train-good samples.

        Call this once per batch during the training loop, after the anomaly map
        and base score have been computed. The accumulated data is used by fit()
        to estimate normal statistics.

        Args:
            anomaly_map (Tensor): Pixel-level anomaly map, shape (B, 1, H, W).
            base_score  (Tensor): Image-level base score, i.e. in general the MAX anomaly score of the anomaly map shape (B,)
        """
        descriptor = self._compute_descriptor(anomaly_map, mask=mask)
        self._descriptors.append(descriptor.detach().cpu())
        self._base_scores.append(base_score.detach().cpu())

    def fit(self) -> None:
        """
        Fit normal statistics from accumulated train-good descriptors and base scores.

        Must be called after the full training loop, once all train-good batches
        have been processed with accumulate(). Computes and stores:
            - self.mean        : per-dimension mean of descriptors over train-good samples
            - self.std         : per-dimension std  of descriptors over train-good samples
            - self.auto_lambda : scale-matching weight between base score and structural score

        auto_lambda is defined as std(base_scores_train) / std(d_struct_train), which
        ensures both terms contribute equally in terms of variance in the hybrid score.

        After fitting, accumulation buffers are cleared to free memory.

        Raises:
            AssertionError: If accumulate() was never called before fit().
        """
        assert len(self._descriptors) > 0, \
            "No descriptors accumulated. Call accumulate() during training first."

        # Concatenate all accumulated batches: (N_train, 3)
        all_descriptors = torch.cat(self._descriptors, dim=0)

        # Fit per-dimension mean and std defining the normal descriptor distribution
        self.mean = all_descriptors.mean(dim=0)                  # (3,)
        self.std  = all_descriptors.std(dim=0).clamp(min=1e-6)  # (3,) clamped to avoid division by zero

        # ------------------------------------------------------------------
        # Compute auto_lambda.
        # Recompute d_struct on train-good samples using the just-fitted stats,
        # then take the ratio of std(base_scores) / std(d_struct).
        # This scales d_struct so its variance matches that of base_score,
        # preventing either term from dominating the hybrid score purely
        # due to scale differences rather than discriminative content.
        # ------------------------------------------------------------------
        # TODO : is this correct ? 
        d_struct_train = ((all_descriptors - self.mean) / self.std).norm(dim=1)  # (N_train,)

        if len(self._base_scores) > 0:
            base_scores_train = torch.cat(self._base_scores, dim=0)  # (N_train,)
            std_base   = base_scores_train.std().clamp(min=1e-6)
            std_struct = d_struct_train.std().clamp(min=1e-6)
            self.auto_lambda = std_base / std_struct
        else:
            # Fallback: no scale matching, equal weighting
            self.auto_lambda = torch.tensor(1.0)

        # Free accumulation buffers
        self._descriptors = []
        self._base_scores = []

        print(f"[StructCore] Fitted on {all_descriptors.shape[0]} train-good samples.")
        print(f"[StructCore] Descriptor mean : {self.mean}")
        print(f"[StructCore] Descriptor std  : {self.std}")
        print(f"[StructCore] Auto lambda      : {self.auto_lambda:.4f}")

    def score(self, anomaly_map: Tensor, base_score: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Compute the StructCore hybrid image-level anomaly score.

        Computes the structural descriptor of the test anomaly map, measures
        its diagonal Mahalanobis distance from the fitted normal distribution,
        and adds the scaled result to the base score:

            S_hyb = S_base + auto_lambda * D_struct

        where D_struct = || (θ(S) - μ) / σ ||_2

        Must be called after fit() has been called.

        Args:
            anomaly_map (Tensor): Pixel-level anomaly map, shape (B, 1, H, W).
            base_score  (Tensor): Image-level base score, i.e. in general the MAX anomaly score of the anomaly map shape (B,)

        Returns:
            Tensor: Hybrid image-level anomaly scores, shape (B,).

        Raises:
            AssertionError: If fit() has not been called yet.
        """

        device = anomaly_map.device


        # Move fitted statistics to same device as input
        mean        = self.mean.to(device)         # (3,)
        std         = self.std.to(device)          # (3,)
        auto_lambda = self.auto_lambda.to(device)  # scalar

        # Compute structural descriptor for the test batch
        descriptor = self._compute_descriptor(anomaly_map, mask=mask)  # (B, 3)

        # Diagonal Mahalanobis distance: standardize each dimension, take L2 norm
        # Measures how far the test descriptor is from the normal descriptor distribution
        d_struct = ((descriptor - mean) / std).norm(dim=1)  # (B,)

        # Hybrid score: base score augmented by scaled structural distance
        return base_score + auto_lambda * d_struct