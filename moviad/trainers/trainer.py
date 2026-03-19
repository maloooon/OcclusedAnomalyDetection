import torch

from moviad.utilities.evaluator import Evaluator


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        logger: any,
        save_path: str = None,
        saving_criteria: callable = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.logger = logger
        self.save_path = save_path
        self.saving_criteria = saving_criteria
        self.evaluator = Evaluator(self.eval_dataloader, self.device)


    @staticmethod
    def update_best_metrics(best_metrics, metrics):
        best_metrics["img_roc_auc"] = metrics["img_roc_auc"] if metrics["img_roc_auc"] > best_metrics["img_roc_auc"] else best_metrics["img_roc_auc"]
        best_metrics["pxl_roc_auc"] = metrics["pxl_roc_auc"] if metrics["pxl_roc_auc"] > best_metrics["pxl_roc_auc"] else best_metrics["pxl_roc_auc"]
        best_metrics["img_f1"]      = metrics["img_f1"] if metrics["img_f1"] > best_metrics["img_f1"] else best_metrics["img_f1"]
        best_metrics["pxl_f1"]      = metrics["pxl_f1"] if metrics["pxl_f1"] > best_metrics["pxl_f1"] else best_metrics["pxl_f1"]
        best_metrics["img_pr_auc"]  = metrics["img_pr_auc"] if metrics["img_pr_auc"] > best_metrics["img_pr_auc"] else best_metrics["img_pr_auc"]
        best_metrics["pxl_pr_auc"]  = metrics["pxl_pr_auc"] if metrics["pxl_pr_auc"] > best_metrics["pxl_pr_auc"] else best_metrics["pxl_pr_auc"]
        best_metrics["pxl_au_pro"]  = metrics["pxl_au_pro"] if metrics["pxl_au_pro"] > best_metrics["pxl_au_pro"] else best_metrics["pxl_au_pro"]
        return best_metrics

    @staticmethod
    def print_metrics(metrics):
        print(f"""
            img_roc: {metrics["img_roc_auc"]} \n
            pxl_roc: {metrics["pxl_roc_auc"]} \n
            f1_img:  {metrics["img_f1"]} \n
            f1_pxl:  {metrics["pxl_f1"]} \n
            img_pr:  {metrics["img_pr_auc"]} \n
            pxl_pr:  {metrics["pxl_pr_auc"]} \n
            pxl_pro: {metrics["pxl_au_pro"]} \n
        """)

        

    def train(self, epochs: int, evaluation_epoch_interval: int):
        pass


class TrainerResult:
    img_roc_auc: float
    pxl_roc_auc: float
    img_f1: float
    pxl_f1: float
    img_pr_auc: float
    pxl_pr_auc: float
    pxl_au_pro: float

    def __init__(
        self,
        img_roc_auc,
        pxl_roc_auc,
        img_f1,
        pxl_f1,
        img_pr_auc,
        pxl_pr_auc,
        pxl_au_pro
    ):
        self.img_roc_auc = img_roc_auc
        self.pxl_roc_auc = pxl_roc_auc
        self.img_f1 = img_f1
        self.pxl_f1 = pxl_f1
        self.img_pr_auc = img_pr_auc
        self.pxl_pr_auc = pxl_pr_auc
        self.pxl_au_pro = pxl_au_pro