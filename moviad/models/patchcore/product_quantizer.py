from typing import Union

import faiss
import numpy as np
import torch

class ProductQuantizer:
    quantizer: faiss.IndexPQ
    dim = 1
    subspaces: int
    centroid_bits: int = 8

    def __init__(self, subspaces = None, centroids_per_subspace: int = 8):
        self.centroid_bits = centroids_per_subspace
        self.subspaces = subspaces

    def fit(self, input : Union[torch.Tensor, np.ndarray], dim=1) -> None:
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()
        self.dim = dim
        self.subspaces = self.__compute_optimal_m(input)
        self.centroid_bits = self.__compute_optimal_k(input)

        self.quantizer = faiss.IndexPQ(input.shape[dim], self.subspaces, self.centroid_bits)
        self.quantizer.train(input)
        self.quantizer.add(input)

    def encode(self, input : Union[torch.Tensor, np.ndarray], dim = 0) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()

        self.subspaces = self.__compute_optimal_m(input) if self.subspaces is None else self.subspaces

        n = input.shape[dim]
        code_size = self.quantizer.sa_code_size()
        compressed = np.zeros((n, code_size), dtype=np.uint8)

        self.quantizer.sa_encode(input, compressed)

        return torch.tensor(compressed, dtype=torch.float32)

    def decode(self, input : Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(input, torch.Tensor):
            input = input.cpu().numpy()

        if input.dtype != np.uint8:
            input = input.astype(np.uint8)

        decompressed = np.zeros((input.shape[0], self.quantizer.d), dtype=np.float32)
        self.quantizer.sa_decode(input, decompressed)
        return torch.tensor(decompressed, dtype=torch.float32)

    def __compute_optimal_m(self, input: np.ndarray) -> int:
        """
            Compute optimal number of subspaces for product quantizer based on input shape

            Args:
                input: Input data to compute the optimal number of subspaces

            Returns:
                m: Optimal number of subspaces
        """
        d = input.shape[self.dim]
        divisors = [m for m in range(1, d + 1) if d % m == 0]
        valid_m = [m for m in divisors]
        suggested_m = 8 if 8 in valid_m else min(valid_m)

        return suggested_m

    def __compute_optimal_k(self, input: np.ndarray, safety_factor=4) -> int:
        """
            Compute optimal nbits for product quantizer based on number of points

            Args:
                num_points: Number of training points available
                safety_factor: Multiple of centroids needed for stable training (typically 2-5)

            Returns:
                nbits: Optimal number of bits for the product quantizer
            """
        num_points = input.shape[0]
        max_nbits = int(np.log2(num_points / safety_factor))
        return max(4, min(8, max_nbits))

    def save(self, path: str) -> None:
        faiss.write_index(self.quantizer, path)

    def load(self, path: str) -> None:
        self.quantizer = faiss.read_index(path)
