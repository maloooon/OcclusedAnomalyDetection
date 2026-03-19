import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from moviad.trainers.trainer import Trainer, TrainerResult
from moviad.models.components.simplenet.loss import SSNLoss
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet 

from tqdm import tqdm

class TrainerSuperSimpleNet(Trainer):


    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        optimizer = AdamW(
            [
                {
                    "params": self.model.adaptor.parameters(),
                    "lr": SuperSimpleNet.DEFAULT_PARAMETERS["adaptor_learning_rate"],
                    "weight_decay": SuperSimpleNet.DEFAULT_PARAMETERS["adaptor_weight_decay"],
                },
                {
                    "params": self.model.segdec.parameters(),
                    "lr": SuperSimpleNet.DEFAULT_PARAMETERS["segdec_learning_rate"],
                    "weight_decay": SuperSimpleNet.DEFAULT_PARAMETERS["segdec_weight_decay"],
                },
            ],
        )

        scheduler = MultiStepLR(
            optimizer,
            milestones=SuperSimpleNet.DEFAULT_PARAMETERS["milestones_scheduler"],
            gamma=SuperSimpleNet.DEFAULT_PARAMETERS["gamma_scheduler"],
        )

        loss_fn = SSNLoss()
        best_metrics = {}
        best_metrics["img_roc_auc"] = 0
        best_metrics["pxl_roc_auc"] = 0
        best_metrics["img_f1"] = 0
        best_metrics["pxl_f1"] = 0
        best_metrics["img_pr_auc"] = 0
        best_metrics["pxl_pr_auc"] = 0
        best_metrics["pxl_au_pro"] = 0

        for epoch in range(epochs):
            
            print(f"EPOCH: {epoch}")

            self.model.train()

            avg_batch_loss = 0
            for batch in tqdm(self.train_dataloader):

                batch = batch.to(self.device)
                B,C,H,W = batch.shape
                masks = torch.zeros(B, 1, H, W).to(self.device)
                labels = torch.zeros(B).to(self.device)

                anomaly_map, anomaly_score, masks, labels = self.model(
                    images=batch,
                    masks=masks,
                    labels=labels,
                )
                loss = loss_fn(pred_map=anomaly_map, pred_score=anomaly_score, target_mask=masks, target_label=labels)

                avg_batch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_batch_loss = avg_batch_loss / len(self.train_dataloader)
            if self.logger is not None:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_loss": avg_batch_loss
                })

            if (epoch + 1) % evaluation_epoch_interval == 0 and epoch != 0:
                print("Evaluating model...")
                metrics = self.evaluator.evaluate(self.model)

                if self.saving_criteria and self.save_path is not None and self.saving_criteria(best_metrics, metrics):
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