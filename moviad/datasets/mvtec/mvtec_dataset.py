import math
from enum import Enum
from typing import Optional

from numpy.ma.core import indices
from torchvision.transforms.functional import InterpolationMode

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset

from moviad.backbones.micronet.utils import compute_mask_contamination
from moviad.datasets.iad_dataset import IadDataset
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.utilities.configurations import TaskType, Split, LabelName

IMG_EXTENSIONS = (".png", ".PNG")

CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class MvtecClassEnum(Enum):
    BOTTLE = "bottle"
    CABLE = "cable"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    PILL = "pill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"


IMG_SIZE = (3, 900, 900)

"""Create MVTec AD samples by parsing the MVTec AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    This function creates a dataframe to store the parsed information based on the following format:
"""


class MVTecDataset(IadDataset):
    """MVTec dataset class.

    Args:
        task (TaskType): Task type, ``classification``, ``detection`` or ``segmentation``.
        root (Path | str): Path to the root of the dataset.
            Defaults to ``./datasets/MVTec``.
        category (str): Sub-category of the dataset, e.g. 'bottle'
            Defaults to ``bottle``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Split of the dataset, usually Split.TRAIN or Split.TEST
            Defaults to ``None``

    """

    def __init__(
            self,
            task: TaskType,
            root: str,
            category: str,
            split: Split,
            norm: bool = True,
            img_size=(224, 224),
            gt_mask_size: Optional[tuple] = None,
            preload_imgs: bool = True,
    ) -> None:
        super(MVTecDataset)

        gt_mask_size = img_size if gt_mask_size is None else gt_mask_size

        self.img_size = img_size
        self.gt_mask_size = gt_mask_size

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples: pd.DataFrame = None
        self.preload_imgs = preload_imgs

        if norm:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:
            t_list = [
                transforms.ToTensor(),
                transforms.Resize(img_size, antialias=True),
            ]

        self.transform_image = transforms.Compose(t_list)

        self.transform_mask = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    gt_mask_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )

    def compute_contamination_ratio(self) -> float:
        if self.samples is None:
            raise ValueError("Dataset is not loaded")

        contaminated_samples = self.samples[self.samples["label_index"] == LabelName.ABNORMAL.value]
        if contaminated_samples.empty:
            return 0

        total_contamination_ratio = 0
        for index, row in contaminated_samples.iterrows():
            if not Path(row["mask_path"]).exists():
                raise ValueError("Mask file does not exist")

            mask = Image.open(row["mask_path"]).convert("L")
            mask = self.transform_mask(mask)
            total_contamination_ratio += compute_mask_contamination(mask)
        return total_contamination_ratio / len(contaminated_samples)

    def is_loaded(self) -> bool:
        return self.samples is not None

    def contains(self, item) -> bool:
        return self.samples['image_path'].eq(item['image_path']).any()

    def load_dataset(self):
        if self.is_loaded():
            print("Dataset already loaded")
            return

        root = Path(self.root_category)

        samples_list = [
            (str(root),) + f.parts[-3:]
            for f in root.glob(r"**/*")
            if f.suffix in IMG_EXTENSIONS
        ]

        if not samples_list:
            msg = f"Found 0 images in {root}"
            raise RuntimeError(msg)

        samples = pd.DataFrame(
            samples_list, columns=["path", "split", "label", "image_path"]
        )

        # Modify image_path column by converting to absolute path
        samples["image_path"] = (
                samples.path
                + "/"
                + samples.split
                + "/"
                + samples.label
                + "/"
                + samples.image_path
        )

        # Create label index for normal (0) and anomalous (1) images.
        samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
        samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
        samples.label_index = samples.label_index.astype(int)

        if self.split == Split.TEST:

            # separate masks from samples
            mask_samples = samples.loc[samples.split == "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )
            samples = samples[samples.split != "ground_truth"].sort_values(
                by="image_path", ignore_index=True
            )

            # assign mask paths to anomalous test images
            samples["mask_path"] = ""
            samples.loc[
                (samples.split == "test") & (samples.label_index == LabelName.ABNORMAL),
                "mask_path",
            ] = mask_samples.image_path.to_numpy()

            # assert that the right mask files are associated with the right test images
            abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
            if (
                    len(abnormal_samples)
                    and not abnormal_samples.apply(
                lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1
            ).all()
            ):
                msg = """Mismatch between anomalous images and ground truth masks. Make sure t
                he mask files in 'ground_truth' folder follow the same naming convention as the
                anomalous images in the dataset (e.g. image: '000.png', mask: '000.png' or '000_mask.png')."""
                raise Exception(msg)

        self.samples = samples[samples.split == self.split].reset_index(drop=True)
        if self.preload_imgs:
            self.data = [
                self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                for index in range(len(self.samples))
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> int:
        if type(source) != MVTecDataset:
            raise ValueError("Dataset should be of type MVTecDataset")
        if self.samples is None:
            raise ValueError("Destination dataset is not loaded")
        if source.samples is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(math.floor(len(self.samples) * ratio))
        contaminated_entries_indices = source.samples[source.samples["label_index"] == LabelName.ABNORMAL.value].index
        if len(contaminated_entries_indices) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(
                f"Source dataset does not contain enough abnormal entries to contaminate the destination dataset. "
                f"Source dataset contains {len(contaminated_entries_indices)} abnormal entries, "
                f"while {contamination_set_size} are required."
            )

        contaminated_entries_indices = np.random.choice(contaminated_entries_indices, contamination_set_size,
                                                        replace=False)
        for index in contaminated_entries_indices:
            entry_metadata = source.samples.iloc[index]
            if source.preload_imgs:
                entry = source.data[index]
                self.data.append(entry)
            else:
                entry = self.transform_image(
                    Image.open(self.samples.iloc[index].image_path).convert("RGB")
                )
                self.data.append(entry)
                source.data = [e for e in source.data if hash(e) != hash(entry)]

            self.samples = pd.concat([self.samples, pd.DataFrame([entry_metadata])], ignore_index=True)
            index_label = source.samples.index[index]

        source.samples = source.samples.drop(contaminated_entries_indices).reset_index(drop=True)
        source.data = [source.data[i] for i in range(len(source.data)) if i not in contaminated_entries_indices]
        return contamination_set_size

    def __getitem__(self, index: int):
        """
        Args:
            index (int) : index of the element to be returned

        Returns:
            image (Tensor) : tensor of shape (C,H,W) with values in [0,1]
            label (int) : label of the image
            mask (Tensor) : tensor of shape (1,H,W) with values in [0,1]
            path (str) : path of the input image
        """

        # open the image and get the tensor
        if self.preload_imgs:
            image = self.data[index]
        else:
            image = self.transform_image(
                Image.open(self.samples.iloc[index].image_path).convert("RGB")
            )

        if self.split == Split.TRAIN:
            return image
        else:
            # return also the label, the mask and the path
            label = self.samples.iloc[index].label_index
            path = self.samples.iloc[index].image_path
            if label == LabelName.ABNORMAL:
                mask = Image.open(self.samples.iloc[index].mask_path).convert("L")
                mask = self.transform_mask(mask)

            else:
                mask = torch.zeros(1, *self.gt_mask_size)

            return image, label, mask.int(), path
