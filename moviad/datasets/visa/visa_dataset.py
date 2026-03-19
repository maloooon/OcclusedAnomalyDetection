from typing import Optional

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from moviad.datasets.iad_dataset import IadDataset
from moviad.datasets.exceptions.exceptions import DatasetTooSmallToContaminateException
from moviad.datasets.visa.visa_data import VisaData, VisaAnomalyClass
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import Split, LabelName


class VisaDataset(IadDataset):
    root_path: str
    csv_path: str
    split: Split
    class_name: str
    data: VisaData

    def __init__(self, root_path: str, csv_path: str, split: Split, class_name: str,
                 gt_mask_size: Optional[tuple] = None, image_size=(224,224), transform=None):
        self.root_path = root_path
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        self.class_name = class_name
        self.gt_mask_size = gt_mask_size
        self.dataframe = pd.read_csv(csv_path)
        self.dataframe = self.dataframe[self.dataframe["split"] == split.value]
        self.dataframe = self.dataframe[self.dataframe["object"] == class_name]
        self.category = class_name

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.PILToTensor(),
                transforms.Resize(
                    image_size,
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ConvertImageDtype(torch.float32),
            ])

    def set_category(self, category: str):
        self.category = category

    def load_dataset(self):
        self.__load__()

    def compute_contamination_ratio(self) -> float:
        if self.data is None or self.data.data is None:
            raise ValueError("Dataset is not loaded")
        return self.data.compute_contamination_ratio()

    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> int:
        if type(source) != VisaDataset:
            raise ValueError("Dataset should be of type VisaDataset")
        if self.data is None or self.data.data is None:
            raise ValueError("Destination dataset is not loaded")
        if source.data is None or source.data.data is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(len(self.data) * ratio)
        contaminated_entries = [entry for entry in source.data.data if entry.label == VisaAnomalyClass.ANOMALY]
        if len(contaminated_entries) < contamination_set_size:
            raise DatasetTooSmallToContaminateException(f"Source dataset does not have enough contaminated entries to contaminate the dataset. "
                             f"Found {len(contaminated_entries)} entries, but needed {contamination_set_size} entries")
        contaminated_entries = np.random.choice(contaminated_entries, contamination_set_size, replace=False).tolist()
        self.data.images.extend(contaminated_entries)
        source.data.data = [entry for entry in source.data.data if entry not in contaminated_entries]
        return contamination_set_size


    def __load__(self):
        self.data = VisaData(meta=self.dataframe, data=self.dataframe)
        self.data.load_images(self.root_path, split=self.split)

    def __len__(self):
        return len(self.data.images)

    def __getitem__(self, item):
        image_data_entry = self.data.images[item]
        image = image_data_entry.image
        mask = image_data_entry.mask

        if self.split == Split.TRAIN:
            if self.transform:
                image = self.transform(image)
            return image

        if self.split == Split.TEST:
            label = LabelName.NORMAL.value if image_data_entry.label == VisaAnomalyClass.NORMAL else LabelName.ABNORMAL.value
            path = str(image_data_entry.image_path)
            if mask is not None:
                mask = self.transform(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size, dtype=torch.float32)
            if self.transform:
                image = self.transform(image)

            return image, label, mask, path
