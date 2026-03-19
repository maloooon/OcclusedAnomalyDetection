from abc import abstractmethod
from enum import Enum

from torch.utils.data.dataset import Dataset
from moviad.utilities.configurations import TaskType, Split

class IadDataset(Dataset):
    task : TaskType
    split: Split
    category: str
    dataset_path: str
    contamination_ratio: float

    @abstractmethod
    def is_loaded(self) -> bool:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def set_category(self, category: str):
        self.category = category

    @abstractmethod
    def compute_contamination_ratio(self) -> float:
        pass

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> int:
        pass

    @abstractmethod
    def contains(self, entry) -> bool:
        pass

    @staticmethod
    def get_argpars_parameters(parser):
        parser.add_argument("--normalize_dataset", action="store_true", help="If the dataset needs to be normalized to ImageNet mean and std")
        parser.add_argument("--batch_size", type=int, help="Batch size for the dataloader")
        return parser
