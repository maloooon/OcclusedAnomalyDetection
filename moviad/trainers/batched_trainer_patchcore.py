import torch
from tqdm import tqdm
from typing_extensions import override
from moviad.models.patchcore.kmeans_coreset_extractor import MiniBatchKMeansCoresetExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore


class BatchPatchCoreTrainer(TrainerPatchCore):
    def __init__(self, patchore_model: PatchCore, train_dataloader: torch.utils.data.DataLoader,
                 test_dataloder: torch.utils.data.DataLoader, device: str, logger=None, cluster_batch_size=16):
        super().__init__(patchore_model, train_dataloader, test_dataloder, device, logger=logger)
        self.cluster_batch_size = cluster_batch_size

    @override
    def train(self):
        self.coreset_extractor = MiniBatchKMeansCoresetExtractor(False, self.device, k=self.patchore_model.k, batch_size=self.cluster_batch_size)
        with torch.no_grad():

            if self.logger is not None:
                self.logger.watch(self.patchore_model)

            print("Embedding Extraction:")
            for batch in tqdm(iter(self.train_dataloader)):
                if isinstance(batch, tuple):
                    embedding = self.patchore_model(batch[0].to(self.device))
                else:
                    embedding = self.patchore_model(batch.to(self.device))

                self.coreset_extractor.partial_fit(embedding)
                del embedding

            coreset = self.coreset_extractor.extract_coreset()

            if self.patchore_model.apply_quantization:
                assert self.patchore_model.product_quantizer is not None, "Product Quantizer not initialized"

                self.patchore_model.product_quantizer.fit(coreset)
                coreset = self.patchore_model.product_quantizer.encode(coreset)

            self.patchore_model.memory_bank = coreset

            img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = self.evaluator.evaluate(self.patchore_model)

            if self.logger is not None:
                self.logger.log({
                    "train_loss": 0,
                    "val_loss": 0,
                    "img_roc": img_roc,
                    "pxl_roc": pxl_roc,
                    "f1_img": f1_img,
                    "f1_pxl": f1_pxl,
                    "img_pr": img_pr,
                    "pxl_pr": pxl_pr,
                    "pxl_pro": pxl_pro,
                })

            print("End training performances:")
            print(f"""
                        img_roc: {img_roc} \n
                        pxl_roc: {pxl_roc} \n
                        f1_img: {f1_img} \n
                        f1_pxl: {f1_pxl} \n
                        img_pr: {img_pr} \n
                        pxl_pr: {pxl_pr} \n
                        pxl_pro: {pxl_pro} \n
                    """)
