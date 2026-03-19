from moviad.datasets.builder import DatasetFactory, DatasetConfig, DatasetType
from moviad.datasets.iad_dataset import IadDataset
from moviad.utilities.configurations import Split


def load_datasets(dataset_config: DatasetConfig, dataset_type: DatasetType, dataset_category: str, image_size: (int, int) = None)\
        -> (IadDataset, IadDataset):
    if image_size is None:
        image_size = dataset_config.image_size
    dataset_factory = DatasetFactory(dataset_config)
    train_dataset = dataset_factory.build(dataset_type, Split.TRAIN, dataset_category, image_size)
    test_dataset = dataset_factory.build(dataset_type, Split.TEST, dataset_category, image_size)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    return train_dataset, test_dataset