import random
import argparse
import pathlib
import tempfile
from dataclasses import dataclass

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from moviad.common.args import Args
from moviad.datasets.builder import DatasetFactory, DatasetType, DatasetConfig
from moviad.datasets.iad_dataset import IadDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.entrypoints.common import load_datasets
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.cfa.cfa import CFA
from moviad.trainers.trainer_cfa import TrainerCFA
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator


@dataclass
class CFAArguments(Args):
    batch_size: int = 4 # default value
    category: str = None
    backbone: str = None
    ad_layers: list = None
    epochs: int = 10
    save_path: str = "./temp.pt"
    model_checkpoint_path: str = f"./patch.pt"
    visual_test_path: str = None


def train_cfa(args: CFAArguments, logger=None):
    train_dataset, test_dataset = load_datasets(args.dataset_config, args.dataset_type, args.category)
    if args.contamination_ratio:
        train_dataset.contaminate(test_dataset, args.contamination_ratio)
        contamination = train_dataset.compute_contamination_ratio()
        logger.config.update({
            "contamination": contamination
        }, allow_val_change=True) if logger is not None else None

    print(f"Training CFA for category: {args.category} \n")
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=True)

    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device)

    cfa_model = CFA(feature_extractor, args.backbone, args.device)
    cfa_model.initialize_memory_bank(train_dataloader)
    cfa_model = cfa_model.to(args.device)

    trainer = TrainerCFA(cfa_model, args.backbone, feature_extractor, train_dataloader, test_dataloader, args.category,
                         args.device, logger)
    results, best_results = trainer.train(args.epochs)

    # save the model
    if args.save_path:
        torch.save(cfa_model.state_dict(), args.save_path)


    return results, best_results


def test_cfa(args: CFAArguments, logger=None):
    dataset_factory = DatasetFactory(args.dataset_config)
    test_dataset = dataset_factory.build(args.dataset_type, args.dataset_split.TEST, args.category)
    test_dataset.load_dataset()

    if logger is not None:
        logger.config.update({
            "ad_model": "cfa",
            "dataset": type(test_dataset).__name__,
            "category": args.category,
            "backbone": args.backbone,
            "ad_layers": args.ad_layers,
        }, allow_val_change=True)

    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(args.backbone, args.ad_layers, args.device, True, False, None)
    cfa_model = CFA(feature_extractor, args.backbone, args.device)
    cfa_model.load_model(args.model_checkpoint_path)
    cfa_model.to(args.device)
    cfa_model.eval()

    evaluator = Evaluator(test_dataloader, args.device)
    img_roc, pxl_roc, f1_img, f1_pxl, img_pr, pxl_pr, pxl_pro = evaluator.evaluate(cfa_model)

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

    # chek for the visual test
    if args.visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(args.visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        for images, labels, masks, paths in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores = cfa_model(images.to(args.device))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            for i in range(anomaly_maps.shape[0]):
                cfa_model.save_anomaly_map(dirpath, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i], labels[i],
                                           masks[i])
