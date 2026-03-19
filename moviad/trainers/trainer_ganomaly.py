import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from moviad.trainers.trainer import Trainer, TrainerResult
from moviad.models.components.ganomaly.loss import GeneratorLoss, DiscriminatorLoss

from tqdm import tqdm

class TrainerGanomaly(Trainer):



    def train(self, epochs: int, evaluation_epoch_interval: int = 10) -> (TrainerResult, TrainerResult):

        learning_rate = 0.0002
        beta1 = 0.5
        beta2 = 0.999

        d_opt = Adam(
            self.model.discriminator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
        )
        g_opt = Adam(
            self.model.generator.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
        )

        g_loss_fn = GeneratorLoss()
        d_loss_fn = DiscriminatorLoss()


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

            avg_g_loss = 0
            avg_d_loss = 0
            for batch in tqdm(self.train_dataloader):

                # forward pass
                padded, fake, latent_i, latent_o = self.model(batch.to(self.device))
                pred_real, _ = self.model.discriminator(padded)

                # generator update
                pred_fake, _ = self.model.discriminator(fake)
                g_loss = g_loss_fn(latent_i, latent_o, padded, fake, pred_real, pred_fake)

                g_opt.zero_grad()
                g_loss.backward(retain_graph=True)
                g_opt.step()
                avg_g_loss += g_loss.item()

                # discrimator update
                pred_fake, _ = self.model.discriminator(fake.detach())
                d_loss = d_loss_fn(pred_real, pred_fake)

                d_opt.zero_grad()
                d_loss.backward(retain_graph=True)
                d_opt.step()
                avg_d_loss += d_loss.item()

            avg_g_loss /= len(self.train_dataloader)
            avg_d_loss /= len(self.train_dataloader)
            if self.logger is not None:
                self.logger.log({
                    "current_epoch" : epoch,
                    "avg_batch_discriminator_loss": avg_d_loss,
                    "avg_batch_generator_loss": avg_g_loss,
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