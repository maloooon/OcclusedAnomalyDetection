"""PyTorch model for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
#from memory_profiler import profile
from torch import Tensor, nn

from .product_quantizer import ProductQuantizer
from ...models.patchcore.anomaly_map import AnomalyMapGenerator
from ...utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from ...utilities.get_sizes import *
from ...utilities.filters import filter_holes_batched, compute_hole_mask_patchgrid, suppress_removed_mask_regions, compute_protrusion_weight_patchgrid

SEED = 32
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PatchCore(nn.Module):
    """Patchcore Module."""

    def __init__(
        self,
        device:torch.device,
        input_size: tuple[int],
        feature_extractor: CustomFeatureExtractor,
        num_neighbors: int = 9,
        apply_quantization: bool = False,
        k: int = 10000,
        struct_core_instance = None,
        scoring_mode = 'MAXMEAN_1',
        mask_border_filter_thickness = 0,
        filter_post = 'NONE',
        cls_token_viz_bool = False,
        protrusion_damping_radius: int = 0,
        protrusion_damping_gamma: float = 1,

    ) -> None:

        """
        Constructor of the patch-core model

        Args: 
            device (torch.device): device to be used during the training
            input_size (tuple[int]): size of the input images
            feature_extractor (CustomFeatureExtractor): feature extractor to be used 
            num_neighbors (int): number of neighbors to be considered in the k-nn search
        """

        super().__init__()

        self.num_neighbors = num_neighbors
        self.device = device
        self.input_size = input_size
        self.struct_core_instance = struct_core_instance
        self.scoring_mode = scoring_mode
        self.filter_post = filter_post
        self.mask_border_filter_thickness = mask_border_filter_thickness
        self.protrusion_damping_radius = protrusion_damping_radius
        self.protrusion_damping_gamma = protrusion_damping_gamma


        self.cls_token_viz_bool = cls_token_viz_bool
        self.cls_memory_bank: Tensor | None = None
      #  self.cls_alpha: float = 1


        self.feature_extractor = feature_extractor
        self.feature_pooler = torch.nn.AvgPool2d(3,1,1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.k = k
        self.register_buffer("memory_bank", Tensor())
        self.memory_bank: Tensor
        self.apply_quantization = apply_quantization
        if apply_quantization:
            self.product_quantizer = ProductQuantizer()

    def load(self, model_state_dict_patch, quantizer_state_dict_path):
        """
        Load the Patchcore model

        Parameters:
        ----------
            model_state_dict_patch (dict): model state dictionary
            quantizer_state_dict_patch (dict): quantizer state dictionary
        """

        self.load_model(model_state_dict_patch)
        if quantizer_state_dict_path is not None:
            self.product_quantizer = ProductQuantizer()
            self.product_quantizer.load(quantizer_state_dict_path)
            self.apply_quantization = True

    def forward(self, input_tensor: Tensor) -> Tensor | dict[str, Tensor]:
        """Return Embedding during training, or a tuple of anomaly map and anomaly score during testing.
    
        Steps performed:
        1. Get features from a CNN.
        2. Generate embedding based on the features.
        3. Compute anomaly map in test mode, return the embedding in training mode
    
        Args:
            input_tensor (Tensor): Input tensor
    
        Returns:
            Tensor | dict[str, Tensor]: Embedding for training,
                anomaly map and anomaly score for testing.
        """
    
    
        if isinstance(input_tensor, tuple):
            if len(input_tensor) == 2:
                # This is the case for struct core collection.
                # NOTE : currently, no struct core instance is set ;
                # NOTE : we simply want to collect all anomaly maps/base scores of PatchCore!
                # We need the mask such that we can collect the descriptor values
                # based on only the raspberry mask area (i.e. we do not care for the black background)
                input_tensor, mask = input_tensor
                batch_og, masK_og, depth_og = None, None, None
            elif len(input_tensor) == 6:
                # For evaluation, we need the mask for only looking at the raspberry
                # and the og values for the hole/darkness filtering
                input_tensor, mask, mask_unfiltered, batch_og, mask_og , depth_og  = input_tensor
            
            elif len(input_tensor) == 3:
                input_tensor, mask, mask_unfiltered = input_tensor
                batch_og, mask_og, depth_og = None, None, None
 
    

    
    
        #extract the features for the input tensor
        self.memory_bank.to(self.device)
        with torch.no_grad():
            if "dino" in self.feature_extractor.model_name:
                features, cls_tokens = self.feature_extractor(input_tensor.to(self.device))
               # print(cls_tokens[0].shape) # [batch_size, 1, dimension] for each extracted layer 

            else:
                 features = self.feature_extractor(input_tensor.to(self.device))
     
    
    
        #concatenate the embeddings
        if isinstance(features, dict):
            features = list(features.values())

    
        
        if "dino" in self.feature_extractor.model_name:
            # Per-layer L2 normalization (StructCore protocol)
            features = [F.normalize(f, p=2, dim=1) for f in features]
            cls_tokens = [F.normalize(cls_token, p=2, dim=-1) for cls_token in cls_tokens]
            if self.cls_token_viz_bool:
                ### TODO : FOR TESTING CURRENTLY ###
                # Discard features of the last layer ; we just want the cls token information of that layer
                features = features[:-1]
                ### TODO : FOR TESTING CURRENTLY ###
            # No smoothing — tokens already have global context
            embedding = torch.cat(features, dim=1)
            cls_tokens = torch.cat(cls_tokens, dim=1)

            if self.cls_token_viz_bool:
                cls_vec = cls_tokens[:,-1,:].squeeze(1)  # (B, dim), already L2-normed per layer ; just the last layers cls token, which has the most global information
        
            
        else:
            # CNN: apply local neighborhood aggregation
            # Apply smoothing (3x3 average pooling) to the features.
            smoothing = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            features  = [smoothing(feature) for feature in features]
    
            # Compute maximum shape.
            H_max = max([f.shape[2] for f in features])
            W_max = max([f.shape[3] for f in features])

           # print(H_max, W_max) #wrn : 28x28, #mobilenet : 28x28 
      
    
            # Create resize function instance.
            resizer = torch.nn.Upsample(size=(H_max, W_max), mode="nearest")
    
            # Apply resize function for all input tensors.
            features = [resizer(f) for f in features]
    
            embedding =  torch.cat(features, dim=1)
    
        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)
    
        #embedding shape: (#num_patches, emb_dim)
    
        if self.training:
            if self.cls_token_viz_bool:
                output = (embedding, cls_vec)
            else:
                output = embedding
        else:
    
            if self.feature_extractor.quantized:
                embedding = torch.int_repr(embedding).to(torch.float64)
    
            # apply nearest neighbor search
            if self.apply_quantization:
                patch_scores, locations = self.nearest_neighbors_quantized(embedding=embedding, n_neighbors=1)
            else:
                patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
    
            # reshape to batch dimension 
            patch_scores = patch_scores.reshape((batch_size, -1))
            locations = locations.reshape((batch_size, -1))
    
            # --------------------------------------------------------- #
            if mask is not None:
                # Ensure (batch_size, 1, H, W)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
    
                # Resize mask to match patch grid dimensions (width, height)
                mask_resized = torch.nn.functional.interpolate(
                    mask.float(), size=(width, height), mode="nearest"
                )  # (B, 1, width, height)
    
                # Build the effective mask per sample: mask minus contour border
                effective_mask = torch.zeros_like(mask_resized, dtype=torch.float32)
    
                for i in range(mask_resized.shape[0]):
                    sample_mask = mask_resized[i, 0].cpu().numpy().astype(np.uint8)  # (width, height)
    
                    if self.mask_border_filter_thickness > 0:
                        # Find external contours of the raspberry region
                        contours, _ = cv.findContours(
                            sample_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                        )
                        # Draw contour band — thickness draws half inward, half outward,
                        # but since outside is already 0 in the mask, the AND below
                        # ensures only the inward portion matters.
                        contour_band = np.zeros_like(sample_mask)
                        cv.drawContours(
                            contour_band, contours, -1, 255, thickness=self.mask_border_filter_thickness
                        )
                        # Subtract contour band from the mask
                        inner_mask = sample_mask.copy()
                        inner_mask[contour_band == 255] = 0
                    else:
                        inner_mask = sample_mask
    
                    effective_mask[i, 0] = torch.from_numpy(
                        inner_mask.astype(np.float32)
                    ).to(patch_scores.device)
    
                # Binarize (in case mask had values other than 0/1)
                effective_mask = (effective_mask > 0).float()
    
                # Flatten and zero out everything outside the effective mask
              #  effective_mask_flat = effective_mask.reshape(batch_size, -1)
               # patch_scores = patch_scores * effective_mask_flat
            # --------------------------------------------------------- #

            if 'HOLE_DARKNESS' in self.filter_post:
                thresh_depth, thresh_dark = self.filter_post.split('_')[2:4]
                hole_mask_patch = compute_hole_mask_patchgrid(
                    batch_og, mask_og, depth_og,
                    patch_h=width, patch_w=height,
                    depth_threshold_percentile=int(thresh_depth),
                    brightness_threshold_percentile=int(thresh_dark),
                ).to(effective_mask.device)
                effective_mask = effective_mask * (1.0 - hole_mask_patch)

            if self.protrusion_damping_radius > 0 and mask_unfiltered is not None:
                protrusion_weight = compute_protrusion_weight_patchgrid(
                    mask, mask_unfiltered,
                    patch_h=width, patch_w=height,
                    influence_radius=self.protrusion_damping_radius,
                    gamma=self.protrusion_damping_gamma,
                ).to(effective_mask.device)
                effective_mask = effective_mask * protrusion_weight

            effective_mask_flat = effective_mask.reshape(batch_size, -1)
            patch_scores = patch_scores * effective_mask_flat

            # compute the anomaly score of the images
            pred_scores = self.compute_anomaly_score(patch_scores, locations, embedding)

            # reshape to w,h
            patch_scores = patch_scores.reshape((batch_size, 1, width, height))

            # get the anomaly map
            anomaly_maps = self.anomaly_map_generator(patch_scores, image_size = self.input_size)

    
            if self.struct_core_instance is not None and self.scoring_mode == 'STRUCTCORE':
            
                pred_scores = self.struct_core_instance.score(anomaly_maps, pred_scores)

            


            # ── CLS-based image-level score ────────────────────────────
          #  if "dino" in self.feature_extractor.model_name and self.cls_memory_bank is not None and self.cls_token_scoring_bool:
          #      cls_memory = self.cls_memory_bank.to(self.device)  # (N_train, dim)
                # Cosine similarity → distance; both sides already L2-normed
           #     sim   = cls_vec @ cls_memory.T          # (B, N_train)
           #     k     = 3 
           #     topk  = sim.topk(k, dim=1).values       # (B, k) highest similarities
           #     cls_dist = 1.0 - topk.mean(dim=1)       # (B,) ∈ [0, 2], lower = more normal
           #     cls_score = cls_dist
                # Normalise to roughly [0,1] using training statistics stored at fit time
              #  cls_score = (cls_dist - self.cls_score_mean) / (self.cls_score_std + 1e-8)
              #  cls_score = cls_score.clamp(min=0.0)
             #   pred_scores = cls_score
            #    #((1.0 - self.cls_alpha) * pred_scores + self.cls_alpha * cls_score)
     
            # ──────────────────────────────────────────────────────────
            
            if "dino" in self.feature_extractor.model_name:
                output = (anomaly_maps, pred_scores, embedding, self.memory_bank, cls_tokens) 
            else:
                output = (anomaly_maps, pred_scores, embedding, self.memory_bank, None)
    
        return output




    def generate_embedding(self, features: dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: dict[str:Tensor]: Hierarchical feature map from a CNN 

        Returns:
            Embedding vector [Tensor]
        """

        embeddings = [features[self.layers[0]]]

        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size = embeddings[0].shape[-2], mode = "bilinear")
            if self.quantize:
                layer_embedding = layer_embedding.quantize()
            embeddings.append(layer_embedding)

        embeddings = torch.cat(embeddings, 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: Tensor) -> Tensor:
        """
        Reshape Embedding.

        Reshapes Embedding to the following format:
        [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

        Args:
            embedding (Tensor): Embedding tensor extracted from CNN features.

        Returns:
            Tensor: Reshaped embedding tensor.
        """

        embedding_size = embedding.size(1)
        embedding = embedding.permute(0,2,3,1).reshape(-1, embedding_size)
        return embedding

    def euclidean_distance(x: Tensor, y: Tensor, quantized:bool) -> Tensor:
        """
        Calculates pair-wise distance between row vectors in x and those in y.

        Args:
            x: input tensor 1
            y: input tensor 2
            quantized: bool, True if x and y are quantized tensors

        Returns:
            Matrix of distances between row vectors in x and y.
        """

        if quantized:
            return torch.cdist(x.dequantize(), y.dequantize())
        else:
            return torch.cdist(x, y)

    def nearest_neighbors(self, embedding: Tensor, n_neighbors: int, memory_bank: torch.Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Nearest neighbors using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        if memory_bank is None:
            memory_bank = self.memory_bank

        distances = PatchCore.euclidean_distance(embedding, memory_bank, quantized=self.feature_extractor.quantized)

        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k = n_neighbors, largest = False, dim = 1)

        return patch_scores, locations


    def nearest_neighbors_quantized(self, embedding: Tensor, n_neighbors: int) -> tuple[Tensor, Tensor]:
        """
        Nearest neighbors using brute force method and euclidean norm.

        Args:
            embedding (Tensor): Features to compare the distance with the memory bank.
            n_neighbors (int): Number of neighbors to look at

        Returns:
            Tensor: Patch scores.
            Tensor: Locations of the nearest neighbor(s).
        """
        self.memory_bank = self.memory_bank.to(self.device)

        # Top 100 nearest neighbors
        quantized_embedding = self.product_quantizer.encode(embedding)
        quantized_embedding = quantized_embedding.to(self.device)
        distances = PatchCore.euclidean_distance(quantized_embedding, self.memory_bank,
                                                 quantized=self.feature_extractor.quantized)

        # Top 100 nearest neighbors
        top_100_patch_scores, top_100_locations = distances.topk(k=100, largest=False, dim=1)

        # Decode the top 100 neighbors from the memory bank
        top_100_neighbors = self.memory_bank[top_100_locations]
        patch_scores = []
        locations = []
        for embedding_index in range(top_100_neighbors.size(0)):
            neighbours = top_100_neighbors[embedding_index]
            decoded_neighbors = self.product_quantizer.decode(neighbours)
            embedding_value = embedding[embedding_index].unsqueeze(0)
            decoded_neighbors = decoded_neighbors.to(self.device)
            neighbour_distances = PatchCore.euclidean_distance(embedding_value, decoded_neighbors, quantized=False)
            top_patch_score, top_location = neighbour_distances.topk(k=n_neighbors, largest=False, dim=1)
            patch_scores.append(top_patch_score)
            locations.append(top_location)
        patch_scores = torch.cat(patch_scores, dim=0).squeeze()
        locations = torch.cat(locations, dim=0).squeeze()
        return patch_scores, locations


    def compute_anomaly_score(self, patch_scores: Tensor, locations: Tensor, embedding: Tensor) -> Tensor:
        """
        Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            locations: Memory bank locations of the nearest neighbor for each patch location
            embedding: The feature embeddings that generated the patch scores

        Returns:
            Tensor: Image-level anomaly scores
        """
        memory_bank = self.memory_bank
        if self.apply_quantization:
            assert self.product_quantizer is not None
            memory_bank = self.product_quantizer.decode(memory_bank)
            memory_bank = memory_bank.to(self.device)

        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)

        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]

        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper

        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = memory_bank[nn_index, :]  # m^* in the paper

        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = memory_bank.shape[0]  # edge case when memory bank is too small
        if self.apply_quantization:
            _, support_samples = self.nearest_neighbors_quantized(
                nn_sample,
                n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
            )
        else:
            _, support_samples = self.nearest_neighbors(
                nn_sample,
                n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
            )

        # 4. Find the distance of the patch features to each of the support samples
        distances = PatchCore.euclidean_distance(max_patches_features.unsqueeze(1), memory_bank[support_samples], self.feature_extractor.quantized)

        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]


        if "MAXMEAN" in self.scoring_mode:
            k = float(self.scoring_mode.split('_')[-1])

            weighted_max = weights * score
            mean_scores = patch_scores.mean(1)
            lambda_weight = k
            return lambda_weight * weighted_max + (1 - lambda_weight) * mean_scores

        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper

    def save_model(self, output_path):
        """
        Save the Patchcore model

        Parameters:
        ----------
            output_path (str): where the model will be saved
        """

        model_state_dict = self.state_dict()
        if self.cls_memory_bank is not None:
            model_state_dict["cls_memory_bank"] = self.cls_memory_bank
        if self.apply_quantization:
            assert self.product_quantizer is not None
            self.product_quantizer.save(output_path + "/product_quantizer.bin")
        torch.save(model_state_dict, output_path)

    def save_anomaly_map(self, dirpath, anomaly_map, pred_score, filepath, x_type, mask):
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
        output_image = (anomaly_map_norm / 2 + original_image / 2).astype(np.uint8)

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
        plt.savefig(str(dirpath + f"/{x_type}_{filename}.jpg"))

    def get_model_size_and_macs(self):
        sizes = {}

        # get feature extractor size, params, and macs

        macs, params = get_model_macs(self.feature_extractor.model)
        sizes["feature_extractor"] = {
            "size" : get_torch_model_size(self.feature_extractor.model),
            "params" : params,
            "macs" : macs
        }

        # get MB size and shape
        sizes["memory_bank"] = {
            "size" : get_tensor_size(self.memory_bank),
            "type" : str(self.memory_bank.dtype),
            "shape" : self.memory_bank.shape
        }

        total_size = sizes["feature_extractor"]["size"] + sizes["memory_bank"]["size"]

        return sizes, total_size

    def load_model(self, path):

        """
        Load the Patchcore memory bank

        Parameters:
        ----------
            path (str): where the pt file containing the memory bank is stored
        """

        state_dict = torch.load(path)

        ## TODO: MemoryBank quantization

        if "memory_bank" not in state_dict.keys():
            raise RuntimeError("Memory Bank tensor not in model checkpoint")

        # load the memory bank
        self.memory_bank = state_dict["memory_bank"]
        self.cls_memory_bank = state_dict.get("cls_memory_bank", None)