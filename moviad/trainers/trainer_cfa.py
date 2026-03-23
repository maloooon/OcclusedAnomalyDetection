import os

import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm

from moviad.models.cfa.cfa import CFA
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult, Trainer

SEED = 32
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TrainerCFA(Trainer):
    """
    This class contains the code for training the CFA model

    Args:
        cfa_model (CFA): model to be trained
        feature_extractor (CustomFeatureExtractor): feature extractor to be used
        train_dataloader (torch.utils.data.DataLoader): train dataloader
        test_dataloder (torch.utils.data.DataLoader): test dataloader
        device (str): device to be used for the training
    """

    def __init__(
        self,
        cfa_model: CFA,
        feature_extractor: CustomFeatureExtractor,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloder: torch.utils.data.DataLoader,
        device: str,
        logger=None,
        save_path: str = None,
        saving_criteria: callable = None
    ):
        super().__init__(
            cfa_model,
            train_dataloader,
            test_dataloder,
            device,
            logger=logger,
            save_path=save_path,
            saving_criteria=saving_criteria
        )
        self.feature_extractor = feature_extractor

    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):
        """
        Train the model by first extracting the features from the batches, transform them
        with the patch descriptor and then apply the CFA loss

        Args:
            epochs (int) : number of epochs for the training
            evaluation_epoch_interval: optional, number of epochs between evaluations
        """

        params = [{'params': self.model.parameters()}]
        learning_rate = 1e-3
        weight_decay = 5e-4
        optimizer = AdamW(params=params,
                          lr=learning_rate,
                          weight_decay=weight_decay,
                          amsgrad=True)

        best_metrics = {}
        best_metrics["img_roc_auc"] = 0
        best_metrics["pxl_roc_auc"] = 0
        best_metrics["img_f1"] = 0
        best_metrics["pxl_f1"] = 0
        best_metrics["img_pr_auc"] = 0
        best_metrics["pxl_pr_auc"] = 0
        best_metrics["pxl_au_pro"] = 0

        if self.logger is not None:
            self.logger.config.update(
                {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "optimizer": "AdamW"
                },
                allow_val_change=True
            )
            self.logger.watch(self.model, log='all', log_freq=10)

        for epoch in range(epochs):

            print(f"EPOCH: {epoch}")

            self.model.train()
            batch_loss = 0
            for batch in tqdm(self.train_dataloader):
                optimizer.zero_grad()

                loss = self.model(batch.to(self.device))
                batch_loss += loss.item()
                if self.logger is not None:
                    self.logger.log({"loss": loss.item()})
                loss.backward()
                optimizer.step()

            avg_batch_loss = batch_loss / len(self.train_dataloader)
            if self.logger is not None:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                metrics = self.evaluator.evaluate(self.model)

                if self.saving_criteria(best_metrics, metrics) and self.save_path is not None:
               # if self.save_path is not None:
                    print("Saving model...")
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Model saved to {self.save_path}")

                # update the best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, metrics)

                print("Trainer training performances:")
                Trainer.print_metrics(metrics)

                if self.logger is not None:
                    self.logger.log(best_metrics)

        print("Best training performances:")
        Trainer.print_metrics(best_metrics)

        if self.logger is not None:
            self.logger.log(
                best_metrics
            )

        best_results = TrainerResult(
            **best_metrics
        )

        results = TrainerResult(
            **metrics
        )


        return results, best_results
