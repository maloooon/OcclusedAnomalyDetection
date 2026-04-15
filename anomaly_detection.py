import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pickle

from moviad.common.common_utils import obsolete
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.fastflow.fastflow import create_fastflow
from moviad.models.cfa.cfa import CFA
from moviad.models.stfpm.stfpm import STFPM
from moviad.models.rd4ad.rd4ad import RD4AD
from moviad.models.sinbad.sinbad import SINBAD
from moviad.models.padim.padim import Padim
from moviad.models.supersimplenet.supersimplenet import SuperSimpleNet
from moviad.models.ganomaly.ganomaly import Ganomaly
from moviad.trainers.trainer_rd4ad import TrainerRD4AD
from moviad.trainers.trainer_cfa import TrainerCFA
from moviad.trainers.trainer_stfpm import TrainerSTFPM
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.trainers.trainer_fastflow import TrainerFastFlow
from moviad.trainers.trainer_sinbad import TrainerSINBAD
from moviad.trainers.trainer_padim import TrainerPadim
from moviad.trainers.trainer_ganomaly import TrainerGanomaly
from moviad.trainers.trainer_supersimplenet import TrainerSuperSimpleNet
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator
from moviad.models.patchcore.product_quantizer import ProductQuantizer
from moviad.utilities.metrics import save_anomaly_map, save_imgs

from torchvision.transforms.functional import InterpolationMode

from synthetic_occlusion import SyntheticOcclusion


from create_dataset import _center_object

import random
import argparse
import gc
import pathlib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
import pickle
import random

import cv2 


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

from create_dataset import _center_object, data_split_non_anomalous

from moviad.utilities.struct_core import StructCore

from image_manipulation import find_holes

from ss_cutout import CutoutReconstructionModel, apply_cutout, train_cutout_reconstruction



class SingleRaspberryDataset(Dataset):
    # CHANGE: removed filter_test_bool from the signature entirely.
    # If you have call sites passing filter_test_bool=..., just remove that argument.
    def __init__(self, dataset_path: str, split=None, synthetic_augmentation=False, synthetic_augmentation_mode = 'replace',
                 AD_model=None, backbone_model=None, struct_core_collection_bool=False,
                 pass_og_bool=True):
        """
        Args:
            dataset_path: path to the 'processed' folder (contains normal/, anomalous/, splits/)
            synthetic_augmentation_mode : 'replace' replaces og training samples with occluded variants, 'augment' keeps og training samples and adds occluded variants (only for train split)
            split: 'train' or 'test' (or None)
 
        """
        self.dataset_path = dataset_path
        self.split = split
        self.synthetic_augmentation = synthetic_augmentation
        self.synthetic_augmentation_mode = synthetic_augmentation_mode
        self.struct_core_collection_bool = struct_core_collection_bool
        self.pass_og_bool = pass_og_bool
 
        # CHANGE: Removed these — no longer scanning external directories at runtime
        # self.removed_raspberries_darkness_path = Path('../../disk/removed_dark_raspberries')
        # self.removed_raspberries_size_path = Path('../../disk/removed_size_raspberries')
 
        if AD_model == 'ganomaly':
            transform_sizes = 256
        else:
            transform_sizes = 224 
 
        if self.synthetic_augmentation:
            from synthetic_occlusion import SyntheticOcclusion
            self.synthetic_occlusion = SyntheticOcclusion(
                base_path=Path(self.dataset_path),
                sample_folders=['anomalous', 'normal']
            )
 
        self.transform_img = transforms.Compose([
            transforms.Resize((transform_sizes, transform_sizes), antialias=True,
                             interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
 
        self.transform_mask = transforms.Compose([
            transforms.Resize((transform_sizes, transform_sizes), antialias=True,
                             interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
 
        # --- Data loading (unchanged logic, but now the pkls may already be filtered) ---
        normal_path = Path(self.dataset_path) / 'normal' / 'normal_samples.pkl'
        anomalous_path = Path(self.dataset_path) / 'anomalous' / 'anomalous_samples.pkl'
 
        self.train_indices_non_anom = pickle.load(
            open(Path(self.dataset_path) / 'splits' / 'train_normal_indices.pkl', 'rb'))
        self.test_indices_non_anom = pickle.load(
            open(Path(self.dataset_path) / 'splits' / 'test_normal_indices.pkl', 'rb'))
 
        with open(normal_path, 'rb') as f:
            normal_data = pickle.load(f)
        with open(anomalous_path, 'rb') as f:
            anomalous_data = pickle.load(f)
 
        normal_test = [normal_data[i] for i in self.test_indices_non_anom]
        normal_train = [normal_data[i] for i in self.train_indices_non_anom]
 
        if self.split == 'train':
            self.data = normal_train
        else:
            self.data = anomalous_data + normal_test


        if synthetic_augmentation and self.split == 'train' and synthetic_augmentation_mode == 'augment':
            # This mode is mainly for a model such as PatchCore (meaning we create a fixed augmentation dataset
            # and actually add the synthetic samples to the dataset, instead of creating synthetic samples on the fly 
            #and replacing the original ones during training as in 'replace' mode). Since PatchCore has no epochs
            # that is runs through, this could be a nice variant to research.
            augmented_data = []

            for item in self.data:
                if random.random() < 0.5:
                    wanted_size_range = (random.randint(5,7) / 10), (random.randint(8,10) / 10)

                    new_img, new_mask, grade = self.synthetic_occlusion.multi_raspberry_occlusion(
                        wanted_size_range=wanted_size_range,
                        randomize_scale_bool=(False, 0.5, 0.9),
                        randomize_rotation_bool=(False, -180, 180),
                        visualize_bool=False,
                        reassign_source_target_bool=False,
                        sampling_mode=('N_largest', 50),
                        k=2,
                        chosen_initial_raspberry=(item['image'], item['mask'], item['grade'])
                    )

                    if new_img is None:
                        continue

                    # --- same post-processing as in __getitem__ ---
                    new_img = np.asarray(new_img, dtype=np.uint8)
                    new_img, new_mask = self.synthetic_occlusion.clean_mask_and_img(new_img, new_mask)
                    new_mask = np.asarray(new_mask, dtype=bool)
                    new_img, new_mask = _center_object(new_img, new_mask)

                    new_depth = item['depth'].copy()
                    new_depth[~new_mask] = 0

                    # --- create new sample dict ---
                    new_item = {
                        'img_path': item['img_path'],  # or mark as synthetic if you want
                        'grade': grade,
                        'mask': new_mask,
                        'depth': new_depth,
                        'image': new_img
                    }

                    augmented_data.append(new_item)

            # extend dataset
            self.data = self.data + augmented_data
 
        self.img_paths = [item['img_path'] for item in self.data]
        self.grades = [item['grade'] for item in self.data]
        self.masks = [item['mask'] for item in self.data]
        self.depths = [item['depth'] for item in self.data]
        self.img_arrays = [item['image'] for item in self.data]
 
 
    def __len__(self):
        return len(self.img_paths)
 

    def __getitem__(self, idx):
        synthetic_added = False
 
        if self.synthetic_augmentation and self.synthetic_augmentation_mode == 'replace' and random.random() < 0.5:
            
         
            wanted_size_range = (random.randint(5,7) / 10), (random.randint(8,10) / 10)
 
            new_img, new_mask, grade = self.synthetic_occlusion.multi_raspberry_occlusion(
                wanted_size_range=wanted_size_range,
                randomize_scale_bool=(False, 0.5, 0.9),
                randomize_rotation_bool=(False, -180, 180),
                visualize_bool=False,
                reassign_source_target_bool=False,
                sampling_mode=('N_largest', 50),
                k=2,
                chosen_initial_raspberry=(self.img_arrays[idx], self.masks[idx], self.grades[idx])
            )
 
            if new_img is None:
                # In this case no viable occlusion pattern was found 
                pass
            else:
                synthetic_added = True
                new_img = np.asarray(new_img, dtype=np.uint8)
                new_img, new_mask = self.synthetic_occlusion.clean_mask_and_img(new_img, new_mask)
                new_mask = np.asarray(new_mask, dtype=bool)
                new_img, new_mask = _center_object(new_img, new_mask)
 

                # Saving the image is nice for visualization purposes, but obviously a bottleneck during training loop..
               # synthetic_img_path = Path(self.dataset_path) / 'synthetic' / f'synthetic_{Path(self.img_paths[idx]).stem}.png'
              
                img = Image.fromarray(new_img, 'RGB')
               # img.save(synthetic_img_path)
                img_array = np.asarray(img, dtype=np.uint8)
 
                og_img = img_array.copy()
                mask = new_mask
                og_mask = mask.copy()
                depth = self.depths[idx].copy()
                depth[~mask] = 0
                og_depth = depth.copy()
                mask = Image.fromarray(mask.astype(np.uint8) * 255)
 
        if not synthetic_added:
            #img_file = self.img_paths[idx]
           # img = Image.open(img_file).convert("RGB")
            img = Image.fromarray(self.img_arrays[idx])
            og_img = self.img_arrays[idx].copy()
            mask = self.masks[idx]
            og_mask = mask.copy()
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            depth = self.depths[idx]
            og_depth = depth.copy()
 
        img = self.transform_img(img)
        mask = self.transform_mask(mask)
 
        if self.split == 'test':
            img_path = self.img_paths[idx]
            grade = self.grades[idx]
            if grade > 3:
                error_mask = mask
                needed_grade = 1
                actual_grade = grade
            else:
                error_mask = torch.zeros(img.shape[1], img.shape[2])
                needed_grade = 0
                actual_grade = grade
            if len(error_mask.shape) == 2:
                error_mask = error_mask.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if self.pass_og_bool:
                return img, needed_grade, error_mask.int(), img_path, mask, actual_grade, og_img, og_mask, og_depth
            else:
                return img, needed_grade, error_mask.int(), img_path, mask, actual_grade, img, img, img
        else:
            if self.struct_core_collection_bool:
                return img, mask
            else:
                return img

def pretrain_backbone_cutout(dataset_path, device, backbone, save_path,
                             unfreeze_from=10, epochs=30, lr=1e-4,
                             batch_size=32, n_holes=3, hole_size_range=(32, 64),
                             target_path='full_no_filters'):


    assert backbone == 'mobilenet_v2', "Currently only implemented for MobileNetv2"

    train_set_path = dataset_path / Path(f'{target_path}/processed')



    train_dataset = SingleRaspberryDataset(
        train_set_path, split='train',
        synthetic_augmentation=False,
        AD_model='patchcore',
        backbone_model=backbone,
        struct_core_collection_bool=True,  # such that we return (img, mask)
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CutoutReconstructionModel(unfreeze_from=unfreeze_from, device = device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = torch.nn.MSELoss(reduction='none')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Cutout pretraining: {epochs} epochs, unfreeze from features.{unfreeze_from}")
    print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = torch.nn.MSELoss(reduction='none')

    trained_model = train_cutout_reconstruction(model, train_dataloader, device, epochs=epochs, lr=lr, optimizer=optimizer, criterion=criterion)

    # Save encoder weights only, mapped to torchvision naming
    encoder_state = {f'features.{k}': v for k, v in trained_model.encoder.state_dict().items()}
    torch.save(encoder_state, save_path)
    print(f"Saved fine-tuned encoder to {save_path}")

    del model, optimizer, train_dataloader, train_dataset
    torch.cuda.empty_cache()

def train_model(dataset_path : str, backbone : str, ad_layers : list, save_path : str, device : torch.device, max_dataset_size : int = None, mode = 'patchcore', target_path = 'full_no_filters', pass_og_bool = False, custom_weights_path = None, synthetic_augmentation_bool = False, synthetic_augmentation_mode = 'replace'):

    mode = mode.lower()
    # initialize the feature extractor
    if mode != 'stfpm':
        feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, custom_weights_path = custom_weights_path)
    else:
        teacher = CustomFeatureExtractor(backbone, ad_layers, device, frozen = True, custom_weights_path = custom_weights_path)
        student = CustomFeatureExtractor(backbone, ad_layers, device, frozen = False)

    # Only normal samples for training
    train_set_path = dataset_path  / Path(f'{target_path}/processed')
    # Create the synthetic folder if it does not exist ; if it exists, clear it out
    synthetic_folder = Path(train_set_path) / 'synthetic'
    if synthetic_folder.exists():
        shutil.rmtree(synthetic_folder)
    synthetic_folder.mkdir(parents=True, exist_ok=True)

    train_dataset = SingleRaspberryDataset(train_set_path, split = 'train', synthetic_augmentation = synthetic_augmentation_bool, synthetic_augmentation_mode = synthetic_augmentation_mode, AD_model = mode, backbone_model = backbone)


    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Only anomalous samples for testing
    test_set_path = dataset_path  / Path(f'{target_path}/processed')
    test_dataset = SingleRaspberryDataset(test_set_path, split = 'test', synthetic_augmentation = False, AD_model = mode, backbone_model = backbone, pass_og_bool = pass_og_bool) # filter_test_bool = filter_test_bool, 

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    print(f"Length train dataset: {len(train_dataset)}")
    print(f"Length test dataset: {len(test_dataset)}")
    

    # TODO : change if len(2) for dino to backbone.model_name  because else it messes up (i.e. STFPM fucked up with len(2) ...)
    # NOTE : Ganomaly & supersimplenet & STFPM have no device, therefore also run on cuda:0 ...
    if mode == 'patchcore':
        model = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor, k = 70000, num_neighbors = 500)
    elif mode == 'cfa':
        model = CFA(feature_extractor, backbone, device)
        model.initialize_memory_bank(train_dataloader)
    elif mode == 'fastflow':
        model = create_fastflow((224,224), backbone, device)
    elif mode == 'rd4ad':
        model = RD4AD(backbone, device, input_size = (224,224))
    elif mode == 'stfpm':
        model = STFPM(teacher, student)
    elif mode == 'padim':
        diagonal_convergence = False
        model = Padim(backbone, class_name = 'raspberry', device = device, diag_cov = diagonal_convergence, layers_idxs = ad_layers)
    elif mode == 'ganomaly':
        model = Ganomaly(input_size = (256,256), num_input_channels = 3, n_features = 64, latent_vec_size = 100, extra_layers = 0, add_final_conv_layer = True)
    elif mode == 'supersimplenet':
        model = SuperSimpleNet(feature_extractor)
    elif mode == 'sinbad':
        model = SINBAD(
            device=device,
            input_size=(224, 224),
            feature_extractor=feature_extractor,
            n_projections=200,   # paper default
            n_quantiles=5,        # paper default
            shrinkage=0.1,        # paper default
            scoring_mode='knn',   # 'knn' (original) 
        )


    model.to(device)
    model.train()

    if mode == 'patchcore':
        trainer = TrainerPatchCore(model, train_dataloader, test_dataloader, device)
    elif mode == 'cfa':
        trainer = TrainerCFA(model, feature_extractor, train_dataloader, test_dataloader, device, saving_criteria = saving_criteria, save_path = save_path)
    elif mode == 'fastflow':
        trainer = TrainerFastFlow(model, train_dataloader, test_dataloader, device, logger = None, saving_criteria = saving_criteria, save_path = save_path)
    elif mode == 'rd4ad':
        trainer = TrainerRD4AD(model, train_dataloader, test_dataloader, device, logger = None, saving_criteria = saving_criteria, save_path = save_path)
    elif mode == 'stfpm':
        trainer = TrainerSTFPM(model, train_dataloader, test_dataloader, device, logger = None, saving_criteria = saving_criteria, save_path = save_path)
    elif mode == 'padim':
        trainer = TrainerPadim(model, train_dataloader, test_dataloader, device, apply_diagonalization = False, logger = None)
    elif mode == 'ganomaly':
        trainer = TrainerGanomaly(model, train_dataloader, test_dataloader, device, logger = None, saving_criteria = saving_criteria, save_path = save_path)
    elif mode == 'supersimplenet':
        trainer = TrainerSuperSimpleNet(model, train_dataloader, test_dataloader, device, logger = None, saving_criteria = saving_criteria, save_path = save_path)
    elif mode == 'sinbad':
        trainer = TrainerSINBAD(model, train_dataloader, test_dataloader, device,save_path=save_path, logger=None)
    if mode not in ('patchcore', 'padim', 'sinbad'):
        trainer.train(epochs = 30, evaluation_epoch_interval=1)
    else:
        trainer.train()

    # save the model
    if save_path:

        # Can save at the very end since we do not have typical epoch training (i.e. do not need to save best results during training, just the one result at the end)
        if mode == 'patchcore' or mode == 'padim':
            print("Saving model ...")
            torch.save(model.state_dict(), save_path)


    # force garbage collector in case
    del model
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

def struct_core_collection(dataset_path : str, backbone : str, ad_layers : list, model_checkpoint_path : str, device : torch.device, max_dataset_size : int = None, mode = 'patchcore', target_path = 'full_no_filters'):
    """
    After creating the memory bank/ training a model, collect descriptors for StructCore based on training data
    NOTE : Currently only implemented for PatchCore
    """

    struct_core = StructCore()


    # initialize the feature extractor
    if mode != 'stfpm':
        feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    else:
        teacher = CustomFeatureExtractor(backbone, ad_layers, device, frozen = True)
        student = CustomFeatureExtractor(backbone, ad_layers, device, frozen = False)


    train_set_path = dataset_path  / Path(f'{target_path}/processed')
    train_dataset = SingleRaspberryDataset(train_set_path, split = 'train', synthetic_augmentation = False, AD_model = mode, backbone_model = backbone, struct_core_collection_bool = True)


    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # load the model
    if mode == 'patchcore':
        model = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor, k = 70000, num_neighbors = 500)
    elif mode == 'stfpm':
        model = STFPM(teacher, student)

    

    if mode == 'patchcore' or mode == 'cfa':
        model.load_model(model_checkpoint_path)
    else:
        if mode == 'stfpm':
            model.student.model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        else:
            state_dict = torch.load(model_checkpoint_path, map_location=device)
            assert len(state_dict) != 0, "Loaded state dict is empty. Needs some fix!."


            model.load_state_dict(
            torch.load(model_checkpoint_path, map_location=device), strict=False)


    # length of state dict
   # state_dict = torch.load(model_checkpoint_path, map_location=device)
   # print(f"Length of state dict: {len(state_dict)}")
  

    model.to(device)
    model.eval()

    

    # Iterate over training data and collect descriptors for StructCore
    with torch.no_grad():
        for batch in tqdm(iter(train_dataloader)):
            
            # Convert list batch to tuple if needed (this is messy but I dont understand why it returns a list.. should directly be a tuple)
            if isinstance(batch, list):
                batch = tuple(elem.to(device) for elem in batch)

            if mode == 'patchcore':
                anomaly_maps, pred_scores, _, _, _ = model(batch)
            else:
                anomaly_maps, pred_scores = model(batch)

            struct_core.accumulate(anomaly_maps, pred_scores)

    struct_core.fit()


    print(f"StructCore mean: {struct_core.mean}")
    print(f"StructCore std: {struct_core.std}")
    print(f"StructCore auto_lambda: {struct_core.auto_lambda}")

  #  torch.save({
   #     'mean':        struct_core.mean,
   #     'std':         struct_core.std,
   #     'auto_lambda': struct_core.auto_lambda,
   # }, 'structcore_stats.pt')


    return struct_core

def test_model(dataset_path : str, backbone : str, ad_layers : list, model_checkpoint_path : str, device : torch.device, max_dataset_size : int = None, visual_test_path: str = None, mode = 'patchcore', scoring_mode = 'MAXMEAN_1', filter_post = 'NONE', target_path = 'full_no_filters', mask_border_filter_thickness = 1, filter_test_bool = False, pass_og_bool = False, custom_weights_path = None):
    

    
    if scoring_mode == 'STRUCTCORE':
        print("StructCore collection ...")
        struct_core = struct_core_collection(dataset_path, backbone, ad_layers, model_checkpoint_path, device, max_dataset_size, mode, target_path)
        print("StructCore collection done")

        
    # initialize the feature extractor
    if mode != 'stfpm':
        feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, custom_weights_path = custom_weights_path)
    else:
        teacher = CustomFeatureExtractor(backbone, ad_layers, device, frozen = True, custom_weights_path = custom_weights_path)
        student = CustomFeatureExtractor(backbone, ad_layers, device, frozen = False)


    # Only anomalous samples for testing
    test_set_path = dataset_path  / Path(f'{target_path}/processed')
    test_dataset = SingleRaspberryDataset(test_set_path, split = 'test', synthetic_augmentation = False, AD_model = mode, backbone_model = backbone, pass_og_bool = pass_og_bool) # filter_test_bool = filter_test_bool

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    ## mask_border_filter_thickness refers to : 
        # We want to only look at the mask area of the raspberry when we create the anomaly map
        # Therefore, we draw the contour of the raspberry and only look at the area within this contour.
        # Since often at the border there can be "errors" (i.e segmentation not perfect and anomaly scores there high since model thinks border of a 
        # occluded raspberry is an anomaly), we set some thickness to the contour and then remove this area. Obviously we lose some raspberry by this
        # but the anomalies seem not be directly at the border area, so we lose little.

    # load the model
    if mode == 'patchcore':
        model = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor, k = 70000, num_neighbors = 500, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, mask_border_filter_thickness = mask_border_filter_thickness, filter_post = filter_post)
    elif mode == 'cfa':
        model = CFA(feature_extractor, backbone, device, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness)
    elif mode == 'stfpm':
        model = STFPM(teacher, student, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness)
    elif mode == 'rd4ad':
        model = RD4AD(backbone, device, input_size = (224,224), struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness)
    elif mode == 'fastflow':
        model = create_fastflow((224,224), backbone, device, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness)
    elif mode == 'padim':
        diagonal_convergence = False
        model = Padim(backbone, class_name = 'raspberry', device = device, diag_cov = diagonal_convergence, layers_idxs = ad_layers, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness)
    elif mode == 'ganomaly':
        model = Ganomaly(input_size = (256,256), num_input_channels = 3, n_features = 64, latent_vec_size = 100, extra_layers = 0, add_final_conv_layer = True)
    elif mode == 'supersimplenet':
        model = SuperSimpleNet(feature_extractor)
    elif mode == 'sinbad':
        model = SINBAD(device=device, input_size=(224, 224),feature_extractor=feature_extractor, n_projections=200,n_quantiles=5,shrinkage=0.1,scoring_mode='knn')

    

    if mode == 'patchcore' or mode == 'cfa':
        model.load_model(model_checkpoint_path)
    else:
        if mode == 'stfpm':
            model.student.model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        elif mode == 'sinbad':
            checkpoint_pkl = model_checkpoint_path.replace('.pt', '.pkl').replace('.pth', '.pkl')
            if not checkpoint_pkl.endswith('.pkl'):
                checkpoint_pkl = checkpoint_pkl + '.pkl'
            
            with open(checkpoint_pkl, 'rb') as f:
                state = pickle.load(f)
            assert len(state) != 0, "Loaded state dict is empty."
            model.load_sinbad(state)
        else:
            state_dict = torch.load(model_checkpoint_path, map_location=device)
            assert len(state_dict) != 0, "Loaded state dict is empty. Needs some fix!."


            model.load_state_dict(
            torch.load(model_checkpoint_path, map_location=device), strict=False)


    # length of state dict
   # state_dict = torch.load(model_checkpoint_path, map_location=device)
   # print(f"Length of state dict: {len(state_dict)}")
  

    model.to(device)
    model.eval()




    evaluator = Evaluator(test_dataloader, device)
    metrics = evaluator.evaluate(model)

    print("Evaluation performances:")
    print(f"""
    img_roc: {metrics['img_roc_auc']}
    pxl_roc: {metrics['pxl_roc_auc']}
    f1_img: {metrics['img_f1']}
    f1_pxl: {metrics['pxl_f1']}
    img_pr: {metrics['img_pr_auc']}
    pxl_pr: {metrics['pxl_pr_auc']}
    pxl_pro: {metrics['pxl_au_pro']}
    """)


    opt_threshold = 1571021.4


    # chek for the visual test
    if visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)
        all_pred_scores_non_anomalous = []
        all_pred_scores_anomalous = []
        pred_scores_per_grade = [[] for _ in range(5)]
        mask_size_wrong_predictions = []
        mask_size_correct_predictions = []
        for images, labels, masks, paths, full_mask, actual_grade, og_img, og_mask, og_depth in tqdm(iter(test_dataloader)):
            if mode == 'patchcore':
                anomaly_maps, pred_scores, _ , _, cls_tokens = model((images.to(device), full_mask.to(device), og_img.to(device), og_mask.to(device), og_depth.to(device)))
            elif mode == 'stfpm' or mode == 'cfa' or mode == 'rd4ad':
                anomaly_maps, pred_scores = model((images.to(device), full_mask.to(device), og_img.to(device), og_mask.to(device), og_depth.to(device)))
            elif mode == 'sinbad':
                output = model((images.to(device), full_mask.to(device), og_img.to(device), og_mask.to(device), og_depth.to(device)))
                anomaly_maps = None
                pred_scores = output[0]
            else:
                anomaly_maps, pred_scores = model(images.to(device))

            # Check if still requires grad 
            if isinstance(pred_scores, torch.Tensor) and pred_scores.requires_grad:
                pred_scores = pred_scores.detach()
            if isinstance(anomaly_maps, torch.Tensor) and anomaly_maps.requires_grad:
                anomaly_maps = anomaly_maps.detach()



            if anomaly_maps is not None:
                anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            

            for i in range(len(pred_scores)): # anomaly_maps.shape[0
                if pred_scores[i] > opt_threshold:
                    curr_label = str(labels[i].item()) + "_PRED_1"
                    if labels[i].item() == 1:
                        mask_size_correct_predictions.append(full_mask[i].sum().item())
                    else:
                        mask_size_wrong_predictions.append(full_mask[i].sum().item())
                else:
                    curr_label = str(labels[i].item()) + "_PRED_0"
                    if labels[i].item() == 0:
                        mask_size_correct_predictions.append(full_mask[i].sum().item())
                    else:
                        mask_size_wrong_predictions.append(full_mask[i].sum().item())

                if anomaly_maps is not None:
                    save_anomaly_map(visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
                                            curr_label, masks[i])
                else:
                    save_imgs(visual_test_path, pred_scores[i], paths[i],
                                            curr_label, masks[i])
                # For later evaluation, also save the predicted scores for anomalous and non-anomalous samples separately
                if labels[i].item() == 0:
                    all_pred_scores_non_anomalous.append(pred_scores[i].item())
                else:
                    all_pred_scores_anomalous.append(pred_scores[i].item())
                # Extract the grade from the path and save the predicted scores per grade
                grade = int(paths[i].split('grade')[-1].split('.png')[0])
                pred_scores_per_grade[grade - 1].append(pred_scores[i].item())
                

        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)    
        colors = ['#4C72B0', '#C44E52']
        labels = ['Non-anomalous', 'Anomalous']
        data = [all_pred_scores_non_anomalous, all_pred_scores_anomalous]

        for i, ax in enumerate(axes):
            if len(data[i]) > 0:
                ax.hist(data[i], bins=20, color=colors[i], alpha=0.8, edgecolor='white')
                median = np.median(data[i])
                ax.axvline(x=median, color=colors[i], linestyle='--', linewidth=1.5, label=f'Median ({median:.3f})')
            ax.axvline(x=opt_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({opt_threshold:.3f})')
            ax.legend(fontsize=8)
            ax.set_ylabel('Count')
            n = len(data[i])
            ax.set_title(f'{labels[i]} (n={n})', loc='left', fontsize=10)

        axes[-1].set_xlabel('Predicted Score')
        plt.tight_layout()
        plt.savefig(os.path.join(visual_test_path, 'predicted_scores_histogram.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
        grade_labels = ['Grade 1 (Normal)', 'Grade 2 (Normal)', 'Grade 3 (Normal)', 
                        'Grade 4 (Anomalous)', 'Grade 5 (Anomalous)']

        for i, ax in enumerate(axes):
            if len(pred_scores_per_grade[i]) > 0:
                ax.hist(pred_scores_per_grade[i], bins=20, color=colors[i], alpha=0.8, edgecolor='white')
                median = np.median(pred_scores_per_grade[i])
                ax.axvline(x=median, color=colors[i], linestyle='--', linewidth=1.5, label=f'Median ({median:.3f})')
            ax.axvline(x=opt_threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold ({opt_threshold:.3f})')
            ax.legend(fontsize=8)
            ax.set_ylabel('Count')
            n = len(pred_scores_per_grade[i])
            ax.set_title(f'{grade_labels[i]} (n={n})', loc='left', fontsize=10)

        axes[-1].set_xlabel('Predicted Score')
        plt.tight_layout()
        plt.savefig(os.path.join(visual_test_path, 'predicted_scores_per_grade_subplots.png'), dpi=150, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        colors = ['#55A868', '#C44E52']
        labels = ['Correct Predictions', 'Wrong Predictions']
        data = [mask_size_correct_predictions, mask_size_wrong_predictions]

        for i, ax in enumerate(axes):
            if len(data[i]) > 0:
                ax.hist(data[i], bins=30, color=colors[i], alpha=0.8, edgecolor='white')
                median = np.median(data[i])
                ax.axvline(x=median, color=colors[i], linestyle='--', linewidth=1.5, label=f'Median: {median:.0f} px')
            else:
                median = 0
            ax.set_ylabel('Count')
            n = len(data[i])
            ax.set_title(f'{labels[i]} (n={n}, median={median:.0f} px)', loc='left', fontsize=10)
            ax.legend(fontsize=8)

        axes[-1].set_xlabel('Mask Size (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(visual_test_path, 'mask_size_correct_vs_wrong.png'), dpi=150, bbox_inches='tight')
        plt.close()

def detailed_eval(data_path):
    """
    Evaluate predictions from filenames in data_path.
    Filename format: {actual}_PRED_{predicted}_img{X}_obj{Y}_grade{Z}.png.jpg
    """

    results = []
    for fname in os.listdir(data_path):
        if not fname.endswith('.jpg'):
            continue
        parts = fname.split('_')
        actual = int(parts[0])
        predicted = int(parts[2])
        grade = int(parts[-1].replace('grade', '').replace('.png.jpg', ''))
        results.append({'file': fname, 'actual': actual, 'predicted': predicted, 'grade': grade})

    actuals = [r['actual'] for r in results]
    preds = [r['predicted'] for r in results]

    # Confusion matrix
    cm = confusion_matrix(actuals, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal (0)', 'Anomaly (1)'])
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(data_path, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Per-grade accuracy for non-anomalous (grades 1,2,3)
    print("=== Non-anomalous (actual=0) per grade ===")
    for grade in [1, 2, 3]:
        subset = [r for r in results if r['actual'] == 0 and r['grade'] == grade]
        correct = sum(1 for r in subset if r['predicted'] == 0)
        print(f"  Grade {grade}: {correct}/{len(subset)} correctly predicted as normal")

    # Per-grade accuracy for anomalous (grades 4,5)
    print("\n=== Anomalous (actual=1) per grade ===")
    for grade in [4, 5]:
        subset = [r for r in results if r['actual'] == 1 and r['grade'] == grade]
        correct = sum(1 for r in subset if r['predicted'] == 1)
        print(f"  Grade {grade}: {correct}/{len(subset)} correctly predicted as anomaly")

    return results

def saving_criteria(best_metrics, new_metrics):
    # Since this is needed for the AD models that have training
    if new_metrics["img_roc_auc"] > best_metrics["img_roc_auc"]:
        return True
    else:
        return False   


def main():

    # TODO : try cfa, first do l2 and all that also in cfa code! for dinov2 ; doesnt work trivially ...

    # TODO : use the cls token from ViT (dinov2) as a global anomaly score ? ; dont seem to work as well, see plots..

    # TODO : dinov2 only on the masked area of the object, i.e. we ignore background ; but same issue in MVtec etc. dont think it will change a lot ...

    # TODO : ask about self-supervised pre-training based on dataset : we only have roughly 4800 non-anomalous imgs, arent these too little for good pre-training/transfer learning ? i.e .compared to amount that these models were trained on

    # TODO : try dinomaly in anomalib ; even though model implemented for multi-class, maybe it can help since its ViT based and more global ?



    # TODO : try hole filters 

    MODEL_MODE = 'stfpm' # 'patchcore', 'cfa', 'stfpm', 'rd4ad', 'fastflow', 'padim', 'ganomaly', 'supersimplenet'
    SYN_AUG_BOOL = True # whether to use synthetic occlusions during training
    SYN_AUG_MODE = 'replace' # 'replace' or 'augment'

    # TODO : fix patchcore heatmap visually (i.e somehow dim down that everything red, try to understand why)
    FILTER_PRE = 'FULL_NO_FILTERS_256' # FILTERED_SIZE_k_imgsize, where k refers to the factor for MAD filtering ; FULL_NO_FILTERS_imgsize if no filters

    # Get the last element in filter_pre
    last_element = FILTER_PRE.split('_')[-1]
    if last_element != 'variable':
        pass_og_bool = True
    else:
        pass_og_bool = False

    FILTER_POST = 'NONE' # HOLE_DARKNESS_k_j : filter out holes and dark areas based on depth & darkness of raspberry ; k refers to threshold for depth and j to threshold for darkness ; see utilities/filters for more details
    SCORING = 'MAXMEAN_1' # MAXMEAN_k , where k refers to the factor for the max (i.e. k * max_score + (1-k) * mean_score) ; STRUCTCORE


    dataset_path = Path('../../disk/dataset_single_objects/GT/') 
    target_path = FILTER_PRE.lower()


    # Create a folder 'synthetic' in the current directory and clear it out if it already exists
  #  synthetic_dir = dataset_path / 'synthetic'
  #  if synthetic_dir.exists():
  #      shutil.rmtree(synthetic_dir)
  #  synthetic_dir.mkdir(parents=True, exist_ok=True)


    # TODO : run sinbad with wrn-50-2 just layer 2 and see if dimensions are correct !
    # Train the model
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    backbone = "wide_resnet50_2" 
   # backbone = "dinov2_vitb14"
   # backbone = 'mobilenet_v2'
   # backbone = "dinov3_vitb16"
   # ad_layers = ["features.4", "features.7", "features.10"] 
   # ad_layers = ["layer4"]
  #  ad_layers = ["features.10"] # SINBAD tests
    ad_layers = ["layer2", "layer3"]
   # ad_layers = [3,6]
    end = ".pt" if MODEL_MODE != 'sinbad' else ".pkl"
    save_path = f"../../nvme1/pretrained_models/{MODEL_MODE}_{backbone}_data_{FILTER_PRE}{end}"

    custom_weights_path = None
   # custom_weights_path = f"../../disk/pretrained_models/{backbone}_cutout_{'_'.join(ad_layers)}.pt"
   # if not os.path.exists(custom_weights_path):
   #     raise FileNotFoundError(f"Custom weights not found at {custom_weights_path}. Please run pretrain_backbone_cutout first to generate these weights.")

   # pretrain_backbone_cutout(dataset_path, device, backbone, save_path = custom_weights_path, unfreeze_from=4, epochs=50, lr=1e-4, batch_size=32, n_holes=1, hole_size_range=(32, 64), target_path = target_path)

    train_model(dataset_path, backbone, ad_layers, save_path, device, mode = MODEL_MODE, target_path = target_path, pass_og_bool = pass_og_bool, custom_weights_path = custom_weights_path, synthetic_augmentation_bool = SYN_AUG_BOOL, synthetic_augmentation_mode = SYN_AUG_MODE)

    # Check if visual test path exists and clear it out if it already exists
    visual_test_path = f"../../disk/visual_test/{MODEL_MODE}_{backbone}_data_{FILTER_PRE}_{SCORING}_test_set_{FILTER_POST}/" # NOTE : disk normally
    visual_test_dir = Path(visual_test_path)
    if visual_test_dir.exists():
        shutil.rmtree(visual_test_dir)
    visual_test_dir.mkdir(parents=True, exist_ok=True)


   # test_model(dataset_path, backbone, ad_layers, save_path, device, mode = MODEL_MODE, target_path = target_path, visual_test_path = visual_test_path, scoring_mode = SCORING, filter_post = FILTER_POST, mask_border_filter_thickness = 0, filter_test_bool = FILTER_TEST_BOOL, pass_og_bool = pass_og_bool, custom_weights_path = custom_weights_path)
  #  detailed_eval(f"../../disk/visual_test/{MODEL_MODE}_{backbone}_data_{FILTER_PRE}_{SCORING}_test_set_{FILTER_POST}/")


if __name__ == "__main__":
    main()