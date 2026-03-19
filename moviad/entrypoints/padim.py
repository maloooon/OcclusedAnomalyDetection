import os, random
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from dataclasses import dataclass

from moviad.common.args import Args
from moviad.datasets.iad_dataset import IadDataset
from moviad.entrypoints.common import load_datasets
from moviad.models.padim.padim import Padim
from moviad.trainers.trainer_padim import PadimTrainer
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.evaluator import Evaluator, append_results
from moviad.utilities.configurations import TaskType, Split

BATCH_SIZE = 2
IMAGE_INPUT_SIZE = (224, 224)
OUTPUT_SIZE = (224, 224)


@dataclass
class PadimArgs(Args):
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    category: str = None
    backbone: str = None
    ad_layers: list = None
    model_checkpoint_save_path: str = None
    diagonal_convergence: bool = False
    results_dirpath: str = None
    logger = None


def train_padim(args: PadimArgs, logger=None) -> None:
    train_dataset, test_dataset = load_datasets(args.dataset_config, args.dataset_type, args.category)
    padim = Padim(
        args.backbone,
        args.category,
        device=args.device,
        diag_cov=args.diagonal_convergence,
        layers_idxs=args.ad_layers,
    )
    padim.to(args.device)
    trainer = PadimTrainer(
        model=padim,
        device=args.device,
        save_path=args.model_checkpoint_save_path,
        data_path=None,
        class_name=args.category,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, pin_memory=True, drop_last=True
    )

    trainer.train(train_dataloader, logger)

    # evaluate the model
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    evaluator = Evaluator(test_dataloader=test_dataloader, device=args.device)

    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    torch.cuda.empty_cache()

    if logger is not None:
        logger.log({
            "img_roc": img_roc,
            "pxl_roc": pxl_roc,
            "f1_img": f1_img,
            "f1_pxl": f1_pxl,
            "img_pr": img_pr,
            "pxl_pr": pxl_pr,
            "pxl_pro": pxl_pro,
        })

    print("Evaluation performances:")
    print(f"""
            img_roc: {img_roc}
            pxl_roc: {pxl_roc}
            f1_img: {f1_img}
            f1_pxl: {f1_pxl}
            img_pr: {img_pr}
            pxl_pr: {pxl_pr}
            pxl_pro: {pxl_pro}
            """)


def test_padim(args: PadimArgs, logger=None) -> None:

    padim = Padim(
        args.backbone,
        args.category,
        device=args.device,
        layers_idxs=args.ad_layers,
    )
    path = padim.get_model_savepath(args.model_checkpoint_path)
    padim.load_state_dict(
        torch.load(path, map_location=args.device), strict=False
    )
    padim.to(args.device)
    print(f"Loaded model from path: {path}")

    # Evaluator
    padim.eval()

    test_dataloader = DataLoader(
        args.test_dataset, batch_size=args.batch_size, shuffle=True
    )

    # evaluate the model
    evaluator = Evaluator(test_dataloader=test_dataloader, device=args.device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(padim)

    if logger is not None:
        logger.log({
            "img_roc": img_roc,
            "pxl_roc": pxl_roc,
            "f1_img": f1_img,
            "f1_pxl": f1_pxl,
            "img_pr": img_pr,
            "pxl_pr": pxl_pr,
            "pxl_pro": pxl_pro,
        })

    print("Evaluation performances:")
    print(f"""
        img_roc: {img_roc}
        pxl_roc: {pxl_roc}
        f1_img: {f1_img}
        f1_pxl: {f1_pxl}
        img_pr: {img_pr}
        pxl_pr: {pxl_pr}
        pxl_pro: {pxl_pro}
        """)


def save_results(results_dirpath: str, category: str, seed: int, scores: tuple, backbone: str, ad_layers: tuple,
                 img_input_size: tuple, output_size: tuple):
    metrics_savefile = Path(
        results_dirpath, f"metrics_{backbone}.csv"
    )
    # check if the metrics path exists
    dirpath = os.path.dirname(metrics_savefile)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # save the scores
    append_results(
        metrics_savefile,
        category,
        seed,
        *scores,
        "padim",  # ad_model
        ad_layers,
        backbone,
        "IMAGENET1K_V2",  # NOTE: hardcoded, should be changed
        None,  # bootstrap_layer
        -1,  # epochs (not used)
        img_input_size,
        output_size,
    )
