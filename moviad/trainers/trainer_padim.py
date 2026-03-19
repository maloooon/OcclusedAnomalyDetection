import os
from tqdm import tqdm
import torch

from moviad.models.padim.padim import Padim
from moviad.trainers.trainer import Trainer, TrainerResult


class TrainerPadim(Trainer):

    def __init__(
        self, 
        model: Padim,  
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device,
        apply_diagonalization=False,
        logger=None,
    ):
        """
        Args:
            device: one of the following strings: 'cpu', 'cuda', 'cuda:0', ...
        """
        super().__init__(
            model, 
            train_dataloader, 
            test_dataloader, 
            device, 
            logger
        )
        self.apply_diagonalization = apply_diagonalization

    def train(self):
        print(f"Train Padim. Backbone: {self.model.backbone_model_name}")

        self.model.train()

        if self.logger is not None:
            self.logger.watch(self.model)

        # 1. get the feature maps from the backbone
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.model.layers_idxs
        }
        for x in tqdm(
            self.train_dataloader, "| feature extraction | train | %s |" 
        ):
            outputs = self.model(x.to(self.device))
            assert isinstance(outputs, dict)
            for layer, output in outputs.items():
                layer_outputs[layer].extend(output)

        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.model.raw_feature_maps_to_embeddings(layer_outputs)
        
        # 3. fit the multivariate Gaussian distribution
        if self.apply_diagonalization:
            self.model.fit_multivariate_diagonal_gaussian(embedding_vectors, update_params=True, logger=self.logger)
        else:
            self.model.fit_multivariate_gaussian(embedding_vectors, update_params=True, logger=self.logger)
        
        metrics = self.evaluator.evaluate(self.model)

        if self.logger is not None:
            self.logger.log(
                metrics
            )

        print("End training performances:")
        self.print_metrics(metrics)

        return TrainerResult(**metrics)