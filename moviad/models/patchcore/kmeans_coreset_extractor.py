import torch
from scipy.cluster.vq import kmeans, kmeans2
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing_extensions import override

from moviad.models.patchcore.kcenter_greedy import CoresetExtractor


class MiniBatchKMeansCoresetExtractor(CoresetExtractor):
    def __init__(self, quantized, device: torch.device, sampling_ratio: float = 0.1,
                 k: int = 30000, batch_size: int = 16, n_init: int = 100) -> None:
        super().__init__(quantized, device, sampling_ratio, k)
        self.batch_size = batch_size
        self.n_init = 100
        self.kmeans_extractor = MiniBatchKMeans(n_clusters=self.k, random_state=42, batch_size=self.batch_size,
                                                n_init=self.n_init, compute_labels=False)


    def partial_fit(self, embeddings_batch: torch.tensor) -> None:
        assert self.kmeans_extractor is not None, "KMeans extractor is not initialized"
        # assert embeddings_batch.shape[0] == self.batch_size, "Batch size mismatch"
        embeddings_batch_array = embeddings_batch.cpu().numpy()
        self.kmeans_extractor.partial_fit(embeddings_batch_array)

    def extract_coreset(self, embeddings: torch.Tensor = None) -> torch.Tensor:
        coreset = self.kmeans_extractor.cluster_centers_
        return torch.tensor(coreset).to(self.device)


class KMeansCoresetExtractor(CoresetExtractor):

    def get_coreset_idx_randomp(self, z_lib, n: int = 1000, k: int = 30000, eps: float = 0.90, float16: bool = True,
                                force_cpu: bool = False):
        super().get_coreset_idx_randomp(z_lib, n, k, eps, float16, force_cpu)

    def __init__(self, quantized, device: torch.device, sampling_ratio: float = 0.1,
                 k: int = 30000) -> None:
        super().__init__(quantized, device, sampling_ratio, k)

    def extract_coreset(self, embeddings: torch.Tensor) -> torch.Tensor:
        kmeans_extractor = KMeans(n_clusters=self.k, random_state=42, verbose=1, n_init="auto")
        kmeans_extractor.fit(embeddings.cpu().numpy())
        coreset = kmeans_extractor.cluster_centers_
        return torch.tensor(coreset).to(self.device)
