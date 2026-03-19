import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional, List, Any

import torch
from torch.utils.data import Dataset, DataLoader

from moviad.common.args import Args
from moviad.datasets.iad_dataset import IadDataset
from moviad.entrypoints.common import load_datasets
from moviad.models.stfpm.stfpm import Stfpm
from moviad.trainers.trainer_stfpm import train_param_grid_search
from moviad.utilities.evaluator import Evaluator, append_results


@dataclass
class STFPMArgs(Args):
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    epochs = [10]
    categories: list[str] = None
    backbone: str = None
    save_path: str = None  # Default save path (no path by default)
    model_checkpoint_path: str = None  # Path for loading a model checkpoint
    visual_test_path: str = None  # Path for saving visual test outputs
    ad_layers: List[int | str] = field(default_factory=list)
    img_input_size: tuple[int, int] = (224, 224)
    img_output_size: tuple[int, int] = (224, 224)
    early_stopping: float = False
    checkpoint_dir = "./checkpoints"
    log_dirpath = "./logs"
    normalize_dataset: bool = True
    student_bootstrap_layer = [0]
    seeds = [3]
    results_dirpath: str = './results'
    input_sizes: dict = field(default_factory=lambda: {
        "mcunet-in3": (176, 176),
        "mobilenet_v2": (224, 224),
        "phinet_1.2_0.5_6_downsampling": (224, 224),
        "wide_resnet50_2": (224, 224),
        "micronet-m1": (224, 224),
    })
    ad_model: str = None
    trained_models_filepaths: Optional[List[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "epochs": self.epochs,
            "categories": self.categories,
            "backbone": self.backbone,
            "backbone_model_name": self.backbone,
            "contamination_ratio": self.contamination_ratio,
            "batch_size": self.batch_size,
            "device": self.device,
            "save_path": self.save_path,
            "model_checkpoint_path": self.model_checkpoint_path,
            "visual_test_path": self.visual_test_path,
            "ad_layers": self.ad_layers,
            "img_input_size": self.img_input_size,
            "img_output_size": self.img_output_size,
            "early_stopping": self.early_stopping,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dirpath": self.log_dirpath,
            "normalize_dataset": self.normalize_dataset,
            "student_bootstrap_layer": self.student_bootstrap_layer,
            "seeds": self.seeds,
            "results_dirpath": self.results_dirpath,
            "input_sizes": self.input_sizes,
            "trained_models_filepaths": self.trained_models_filepaths,
            "ad_model": self.ad_model
        }


def train_stfpm(params: STFPMArgs, logger=None, evaluate=True) -> None:
    print(f"Training with params: {params}")
    train_dataset, test_dataset = load_datasets(params.dataset_config, params.dataset_type, params.categories[0])
    params.train_dataset = train_dataset
    params.test_dataset = test_dataset
    params.epochs = params.epochs * len(params.ad_layers)
    trained_models_filepaths = train_param_grid_search(params.to_dict(), logger)
    if evaluate:
        test_stfpm(params, logger)
    m = "\n".join(trained_models_filepaths)
    print(f"Trained models:{m}")


def test_stfpm(params: STFPMArgs, logger=None) -> None:
    if params.trained_models_filepaths is None:
        params.trained_models_filepaths = glob(
            os.path.join(params.checkpoint_dir, "**/*.pth.tar"), recursive=True
        )
    if len(params.trained_models_filepaths) == 0:
        raise ValueError(f"No trained models found in {params.checkpoint_dir}")

    if not os.path.exists(params.results_dirpath):
        os.makedirs(params.results_dirpath)

    models = list(params.input_sizes.keys())

    for checkpoint_path in params.trained_models_filepaths:
        torch.manual_seed(0)

        # get category from dirname
        category = os.path.basename(os.path.dirname(checkpoint_path))

        # get backbone model name from filename
        backbone_model_name = [
            m for m in models if m in os.path.basename(checkpoint_path)
        ][0]
        img_input_size = params.input_sizes[backbone_model_name]

        print(f"backbone model name: {backbone_model_name}")
        print(f"img_input_size: {img_input_size}")
        print(f"Category: {category}")

        test_dataloader = DataLoader(
            params.test_dataset, batch_size=params.batch_size, shuffle=True
        )
        print(f"Length test dataset: {len(params.test_dataset)}")

        # load the model snapshot

        model = Stfpm(input_size=params.img_input_size,
                      output_size=params.img_output_size)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=params.device), strict=False
        )

        if logger is not None:
            logger.watch(model)

        model.to(params.device)

        # evaluate the model
        evaluator = Evaluator(test_dataloader=test_dataloader, device=params.device)
        scores = evaluator.evaluate(model, logger)

        # save the scores
        metrics_filename = os.path.join(
            params.results_dirpath,
            f"metrics_{backbone_model_name}.csv",
        )
        append_results(
            metrics_filename,
            category,
            model.seed,
            *scores,
            params.ad_model,
            str(model.ad_layers),
            backbone_model_name,
            model.weights_name,
            model.student_bootstrap_layer,
            model.epochs,
            img_input_size,
            params.img_output_size,
        )

        torch.cuda.empty_cache()


def visualize_stfpm(params: STFPMArgs):
    if params.trained_models_filepaths is None:
        trained_models_filepaths = glob(
            os.path.join(params.checkpoint_dir, "**/*.pth.tar"), recursive=True
        )
    if len(trained_models_filepaths) == 0:
        raise ValueError(f"No trained models found in {params.checkpoint_dir}")

    if not os.path.exists(params.results_dirpath):
        os.makedirs(params.results_dirpath)

    models = list(params.input_sizes.keys())

    models_to_load = []
    for checkpoint_path in trained_models_filepaths:
        if params.model_name not in checkpoint_path:
            continue
        found = False
        for cat in params.categories:
            if cat in checkpoint_path:
                found = True
                break
        if not found:
            continue
        if params.boot_layer is not None:
            if f"boots{params.boot_layer}" not in checkpoint_path:
                continue
        else:
            if "boots" in checkpoint_path:
                continue
        models_to_load.append(checkpoint_path)

    print("-" * 20, "Models to load", "-" * 20)
    print(models_to_load)

    assert len(models_to_load) > 0, "No models to load"

    for checkpoint_path in models_to_load:
        torch.manual_seed(0)

        # get category from dirname
        category = os.path.basename(os.path.dirname(checkpoint_path))

        # get backbone model name from filename
        backbone_model_name = [
            m for m in models if m in os.path.basename(checkpoint_path)
        ][0]
        img_input_size = params.input_sizes[backbone_model_name]

        print(f"backbone model name: {backbone_model_name}")
        print(f"img_input_size: {img_input_size}")
        print(f"Category: {category}")

        # test_dataloader = DataLoader(
        #     test_dataset, batch_size=batch_size, shuffle=True
        # )
        print(f"Length test dataset: {len(params.test_dataset)}")

        # load the model snapshot
        model = Stfpm()
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=params.device), strict=False
        )
        model.to(params.device)

        # save the scores
        visualization_path = os.path.join(
            params.feature_maps_dir,
            f"{category}_{backbone_model_name}_lay{model.ad_layers}_share{model.student_bootstrap_layer}",
        )
        # 3 files: teacher_maps, student_maps, diff_maps
        teacher_maps, student_maps, diff_maps = [], [], []
        anomaly_maps, original_imgs, labels = [], [], []
        masks = []
        model.attach_hooks(teacher_maps, student_maps)

        for image, label, mask, path in params.test_dataset:
            model.eval()
            image = image.unsqueeze(0)
            with torch.no_grad():
                anomaly_map, anomaly_score = model(image.to(params.device))
                anomaly_maps.append(anomaly_map.cpu().numpy())
                original_imgs.append(image.cpu().numpy())
                labels.append(label)
                masks.append(mask.cpu().numpy())

        import pickle

        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)

        print("num inferences: len(teacher_maps)", len(teacher_maps))
        print("len(student_maps)", len(student_maps))

        print("Len of teacher_maps[0]", len(teacher_maps[0]))
        print("Len of student_maps[0]", len(student_maps[0]))

        with open(os.path.join(visualization_path, "teacher_maps.pkl"), "wb") as f:
            pickle.dump(teacher_maps, f)
        with open(os.path.join(visualization_path, "student_maps.pkl"), "wb") as f:
            pickle.dump(student_maps, f)
        with open(os.path.join(visualization_path, "anomaly_maps.pkl"), "wb") as f:
            pickle.dump(anomaly_maps, f)
        with open(os.path.join(visualization_path, "original_imgs.pkl"), "wb") as f:
            pickle.dump(original_imgs, f)
        with open(os.path.join(visualization_path, "labels.pkl"), "wb") as f:
            pickle.dump(labels, f)
        with open(os.path.join(visualization_path, "masks.pkl"), "wb") as f:
            pickle.dump(masks, f)
