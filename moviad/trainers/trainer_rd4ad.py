from typing import Union, List
import os

from tqdm import *
import numpy as np
import torch
import torch.nn.functional as F

from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult, Trainer

class TrainerRD4AD(Trainer):

    @staticmethod
    def loss_function(teacher_features: List[torch.Tensor], student_features: List[torch.Tensor]):

        cos_loss = torch.nn.CosineSimilarity()
        loss = 0

        #iterate over the feature extraction layers batches
        #every feature maps shape is (B C H W)
        for i in range(len(teacher_features)):
            loss += torch.mean(
                1 - cos_loss(
                    teacher_features[i].view(teacher_features[i].shape[0],-1),
                    student_features[i].view(student_features[i].shape[0],-1)
                )
            )
        return loss


    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        self.model.train()

        optimizer = torch.optim.Adam(
            list(self.model.decoder.parameters())+list(self.model.bn.parameters()),
            lr=RD4AD.DEFAULT_PARAMETERS["learning_rate"],
            betas=RD4AD.DEFAULT_PARAMETERS["betas"],
        )

        best_metrics = {}
        best_metrics["img_roc_auc"] = 0
        best_metrics["pxl_roc_auc"] = 0
        best_metrics["img_f1"] = 0
        best_metrics["pxl_f1"] = 0
        best_metrics["img_pr_auc"] = 0
        best_metrics["pxl_pr_auc"] = 0
        best_metrics["pxl_au_pro"] = 0

        if self.logger:
            self.logger.config.update(
                {
                    "learning_rate": RD4AD.DEFAULT_PARAMETERS["learning_rate"],
                    "optimizer": "Adam"
                },
                allow_val_change=True
            )
            self.logger.watch(self.model, log='all', log_freq=10)

        for epoch in trange(epochs):
            
            self.model.train()

            print(f"EPOCH: {epoch}")

            #train the model
            batch_loss = 0
            for batch in tqdm(self.train_dataloader):

                batch = batch.to(self.device)
                teacher_features, bn_features, student_features = self.model(batch)

                loss = TrainerRD4AD.loss_function(teacher_features, student_features)

                batch_loss += loss.item()
                if self.logger:
                    self.logger.log({"loss": loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_batch_loss = batch_loss / len(self.train_dataloader)
            if self.logger:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })
            print(f"Avg loss on epoch {epoch}: {avg_batch_loss}")

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                metrics = self.evaluator.evaluate(self.model)
                
                if self.saving_criteria(best_metrics, metrics) and self.save_path is not None: 
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
            self.logger.log(best_metrics)

        best_results = TrainerResult(
            **best_metrics
        )

        results = TrainerResult(
            **metrics
        )


        return results, best_results
