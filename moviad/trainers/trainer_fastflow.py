from typing import List

from tqdm import tqdm 
import torch
from torch import Tensor
from torch import nn

from moviad.utilities.evaluator import Evaluator
from moviad.trainers.trainer import TrainerResult, Trainer


class TrainerFastFlow(Trainer):

    """
    This class contains the code for training the FastFlow model
    """

    @staticmethod
    def fastflow_loss(hidden_variables: List[Tensor], jacobians: List[Tensor]) -> Tensor:
        """Calculate the Fastflow loss.

        Args:
            hidden_variables (List[Tensor]): Hidden variables from the fastflow model. f: X -> Z
            jacobians (List[Tensor]): Log of the jacobian determinants from the fastflow model.

        Returns:
            Tensor: Fastflow loss computed based on the hidden variables and the log of the Jacobians.
        """
        loss = torch.tensor(0.0, device=hidden_variables[0].device)  # pylint: disable=not-callable
        for (hidden_variable, jacobian) in zip(hidden_variables, jacobians):
            loss += torch.mean(0.5 * torch.sum(hidden_variable**2, dim=(1, 2, 3)) - jacobian)
        return loss
    
    def train(self, epochs:int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        self.optimizer = torch.optim.Adam(
            self.model.parameters()
        )

        best_metrics = {}
        best_metrics["img_roc_auc"] = 0
        best_metrics["pxl_roc_auc"] = 0
        best_metrics["img_f1"] = 0
        best_metrics["pxl_f1"] = 0
        best_metrics["img_pr_auc"] = 0
        best_metrics["pxl_pr_auc"] = 0
        best_metrics["pxl_au_pro"] = 0


        if self.logger is not None:
            pass #TODO: add configuration logging

        for epoch in range(epochs):

            self.model.train()

            avg_batch_loss = 0.0
            print("Epoch: ", epoch)
            for batch in tqdm(self.train_dataloader):
                batch = batch.to(self.device)
                hidden_variables, jacobians = self.model(batch)
                loss = TrainerFastFlow.fastflow_loss(hidden_variables, jacobians)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                avg_batch_loss += loss.item()
            
            avg_batch_loss /= len(self.train_dataloader)

            if self.logger is not None:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })

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

        

        