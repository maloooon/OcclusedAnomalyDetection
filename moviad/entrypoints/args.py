from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch

@dataclass
class StfpmTrainingParams:
    """Parameters for the STFPM training process."""
    train_dataset: Optional[torch.utils.data.Dataset]
    test_dataset: Optional[torch.utils.data.Dataset] = None
    categories: List[str] = None
    ad_layers: List[List[int]] = None
    ad_model: Optional[str] = None
    results_dirpath: str = './results'
    epochs: int = 10
    seeds: List[int] = None
    batch_size: int = 32
    backbone_model_name: str = None
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_sizes: Dict[str, Tuple[int, int]] = None
    img_input_size: Tuple[int, int] = None
    img_output_size: Tuple[int, int] = None
    early_stopping: Optional[float] = None  # e.g., 0.01 or None
    student_bootstrap_layer: List[Optional[int]] = None
    checkpoint_dir: str = './checkpoints'
    normalize_dataset: bool = True
    dataset_path: str = None
    log_dirpath: str = './logs'
    trained_models_filepaths: Optional[List[str]] = None

    def __post_init__(self):
        """Post-initialization logic for adding derived attributes."""
        # Set defaults for uninitialized attributes based on logic in the original implementation
        if self.backbone_model_name and self.input_sizes:
            self.img_input_size = self.input_sizes.get(self.backbone_model_name, (224, 224))
        if self.student_bootstrap_layer is None:
            self.student_bootstrap_layer = [False]


@dataclass
class TrainingArguments:
    """Arguments for training and testing.
    """
    mode: str
    dataset_path: str
    category: str
    backbone: str
    ad_layers: List[str]
    save_path: str
    epochs: int
    visual_test_path: str
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed: int = 42
    batch_size: int = 4
    model_checkpoint_path: Optional[str] = None
