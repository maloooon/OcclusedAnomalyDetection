import gc
import torch
from dataclasses import dataclass
from moviad.common.args import Args
from moviad.datasets.iad_dataset import IadDataset
from moviad.entrypoints.common import load_datasets
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.trainers.trainer_rd4ad import TrainerRD4AD


@dataclass
class RD4ADArgs(Args):
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    category: str = None
    backbone: str = None
    ad_layers: list = None
    img_input_size: tuple = (224, 224)
    batch_size: int = 2
    epochs: int = 10
    device: torch.device = None
    save_path: str = None


def train_rd4ad(args: RD4ADArgs, logger=None) -> None:

    train_dataset, test_dataset = load_datasets(args.dataset_config, args.dataset_type, args.category, image_size=args.img_input_size)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=True)

    # define the model
    model = RD4AD(args.device,args.img_input_size)
    model.to(args.device)
    trainer = TrainerRD4AD(model, train_dataloader, test_dataloader, args.device, logger=logger)
    trainer.train(args.epochs)

    # save the model
    if args.save_path and args.save_path != "":
        torch.save(model.state_dict(), args.save_path)

    # force garbage collector in case
    del model
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()
