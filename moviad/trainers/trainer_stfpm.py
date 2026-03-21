from tqdm import *
import copy

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from moviad.models.stfpm.stfpm import STFPM
from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult, Trainer

SEED = 32
import numpy as np
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TrainerSTFPM(Trainer):

    """
    This class contains the code for training the STFPM model
    """

    @staticmethod
    def stfpm_loss(teacher_features, student_features):
        return torch.sum((teacher_features - student_features) ** 2, 1).mean()

    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        optimizer = torch.optim.SGD(
            self.model.student.model.parameters(),
            STFPM.DEFAULT_PARAMETERS["learning_rate"], 
            momentum=STFPM.DEFAULT_PARAMETERS["momentum"], 
            weight_decay=STFPM.DEFAULT_PARAMETERS["weight_decay"]
        )

        best_metrics = {}
        best_metrics["img_roc_auc"] = 0
        best_metrics["pxl_roc_auc"] = 0
        best_metrics["img_f1"] = 0
        best_metrics["pxl_f1"] = 0
        best_metrics["img_pr_auc"] = 0
        best_metrics["pxl_pr_auc"] = 0
        best_metrics["pxl_au_pro"] = 0

        # log the training configurations
        if self.logger:
            self.logger.config.update(
                {
                    "epochs": epochs,
                    "learning_rate": STFPM.DEFAULT_PARAMETERS["learning_rate"],
                    "weight_decay":STFPM.DEFAULT_PARAMETERS["weight_decay"],
                    "optimizer": "SGD",
                    "momentum": STFPM.DEFAULT_PARAMETERS["momentum"],
                },
                allow_val_change=True
            )
            self.logger.watch(self.model, log='all', log_freq=10)

        for epoch in trange(epochs):

            self.model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = 0
            #train the model
            for batch in tqdm(self.train_dataloader):

                batch = batch.to(self.device)
                teacher_features, student_features = self.model(batch)

                for i in range(len(student_features)):

                    teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                    student_features[i] = F.normalize(student_features[i], dim=1)
                    loss = TrainerSTFPM.stfpm_loss(teacher_features[i], student_features[i])

                avg_batch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_batch_loss /= len(self.train_dataloader)
            if self.logger:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                metrics = self.evaluator.evaluate(self.model)
                
                if self.saving_criteria(best_metrics, metrics) and self.save_path is not None: 
                    print("Saving model...")
                    torch.save(self.model.student.model.state_dict(), self.save_path)
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