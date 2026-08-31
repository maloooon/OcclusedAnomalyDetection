# Main Script for Anomaly Detection

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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

from time import time


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
import math 

import cv2 


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

from create_dataset import _center_object, data_split_non_anomalous

from moviad.utilities.struct_core import StructCore

from helper import apply_random_transform
from image_manipulation import find_holes, find_holes_fix

from ss_cutout import (apply_cutout,
                       train_cutout_reconstruction, create_cutout_model)

from moviad.utilities.seed_utils import SEED, seed_everything, seed_worker, make_generator
seed_everything(SEED)





def _paste_hole(target_img, target_mask, hole_pixels, hole_mask, max_attempts=100, fade_width=8):
    """Paste extracted hole pixels onto target image at a random position inside the target mask.

    The hole bounding box must fit entirely within the target mask (every active hole pixel must
    land on a mask pixel). Up to max_attempts random positions are tried; returns None if none work.
    Edges of the pasted hole are feathered over fade_width pixels using a distance-transform alpha.
    """
    hole_coords = np.argwhere(hole_mask)
    if len(hole_coords) == 0:
        return None
    y_min, x_min = hole_coords.min(axis=0)
    y_max, x_max = hole_coords.max(axis=0)
    hole_crop = hole_pixels[y_min:y_max + 1, x_min:x_max + 1]
    hole_mask_crop = hole_mask[y_min:y_max + 1, x_min:x_max + 1]
    h, w = hole_crop.shape[:2]
    H, W = target_img.shape[:2]

    if h > H or w > W:
        return None

    # Alpha ramps from 0 at the hole boundary to 1 fade_width pixels inward.
    dist = cv2.distanceTransform(hole_mask_crop.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(dist / max(fade_width, 1), 0.0, 1.0)[..., np.newaxis].astype(np.float32)

    hole_crop_f = hole_crop.astype(np.float32)

    for _ in range(max_attempts):
        paste_y = random.randint(0, H - h)
        paste_x = random.randint(0, W - w)
        target_region = (target_mask[paste_y:paste_y + h, paste_x:paste_x + w] > 0)
        if np.all(target_region[hole_mask_crop]):
            result = target_img.copy().astype(np.float32)
            region = result[paste_y:paste_y + h, paste_x:paste_x + w]
            blended = alpha * hole_crop_f + (1.0 - alpha) * region
            region[hole_mask_crop] = blended[hole_mask_crop]
            return result.clip(0, 255).astype(np.uint8)

    return None


class SingleRaspberryDataset(Dataset):
    def __init__(self, dataset_path: str, split=None, synthetic_augmentation=False, synthetic_augmentation_mode = 'replace',
                 randomize_rotation=False,
                 hole_augmentation=False, hole_augmentation_mode='replace',
                 AD_model=None, backbone_model=None, struct_core_collection_bool=False,
                 pass_og_bool=True, include_gt_fill_ins=True):
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
        self.randomize_rotation = randomize_rotation
        self.hole_augmentation = hole_augmentation
        self.hole_augmentation_mode = hole_augmentation_mode
        self.struct_core_collection_bool = struct_core_collection_bool
        self.pass_og_bool = pass_og_bool
 
 
        if AD_model == 'ganomaly': # earlier was only with ganomaly
            self.transform_sizes = 256
        else:
            self.transform_sizes = 224
 
        if self.synthetic_augmentation:
            from synthetic_occlusion import SyntheticOcclusion
            self.synthetic_occlusion = SyntheticOcclusion(
                base_path=Path(self.dataset_path),
                sample_folders=['anomalous', 'normal']
            )
 
        self.transform_img = transforms.Compose([
            transforms.Resize((self.transform_sizes, self.transform_sizes), antialias=True,
                             interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
 
        self.transform_mask = transforms.Compose([
            transforms.Resize((self.transform_sizes, self.transform_sizes), antialias=True,
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
            if not include_gt_fill_ins:
                normal_test    = [r for r in normal_test    if not r.get('filled_from_gt', False)]
                anomalous_data = [r for r in anomalous_data if not r.get('filled_from_gt', False)]
            self.data = anomalous_data + normal_test


        if synthetic_augmentation and self.split == 'train' and synthetic_augmentation_mode == 'augment':
            # This mode is mainly for a model such as PatchCore (meaning we create a fixed augmentation dataset
            # and actually add the synthetic samples to the dataset, instead of creating synthetic samples on the fly 
            #and replacing the original ones during training as in 'replace' mode). Since PatchCore has no epochs
            # that is runs through, this could be a nice variant to research.
            augmented_data = []

            for item in self.data:
                if random.random() < 0.5:
                    # (5,7) (8,10)
                    wanted_size_range = (random.randint(5,7) / 10), (random.randint(8,10) / 10)

                    new_img, new_mask, grade = self.synthetic_occlusion.multi_raspberry_occlusion(
                        wanted_size_range=wanted_size_range,
                        randomize_scale_bool=(False, 0.5, 0.9),
                        randomize_rotation_bool=(self.randomize_rotation, -30, 30),
                        visualize_bool=False,
                        reassign_source_target_bool=False,
                        sampling_mode=('N_largest', 50),
                        k=2,
                        chosen_initial_raspberry=(item['image'], item['mask'], item['grade'])
                    )

                    if new_img is None:
                        continue

                    new_img = np.asarray(new_img, dtype=np.uint8)
                    new_img, new_mask = self.synthetic_occlusion.clean_mask_and_img(new_img, new_mask)
                    new_mask = np.asarray(new_mask, dtype=bool)

                    new_mask_unfiltered = item['mask_unfiltered'] & ~(item['mask'] & ~new_mask)

                    new_mask_pre_center = new_mask.copy()
                    new_img, new_mask = _center_object(new_img, new_mask)

                    coords = np.argwhere(new_mask_pre_center)
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        img_h, img_w = new_mask_pre_center.shape
                        obj_h, obj_w = y_max - y_min + 1, x_max - x_min + 1
                        paste_y = img_h // 2 - obj_h // 2
                        paste_x = img_w // 2 - obj_w // 2
                        src_y_start = max(0, -paste_y)
                        src_x_start = max(0, -paste_x)
                        src_y_end   = min(obj_h, img_h - paste_y)
                        src_x_end   = min(obj_w, img_w - paste_x)
                        dst_y_start, dst_x_start = max(0, paste_y), max(0, paste_x)
                        dst_y_end = dst_y_start + (src_y_end - src_y_start)
                        dst_x_end = dst_x_start + (src_x_end - src_x_start)
                        centered_unfiltered = np.zeros_like(new_mask_unfiltered)
                        centered_unfiltered[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                            new_mask_unfiltered[y_min:y_max+1, x_min:x_max+1][src_y_start:src_y_end, src_x_start:src_x_end]
                        new_mask_unfiltered = centered_unfiltered

                    new_depth = item['depth'].copy()
                    new_depth[~new_mask] = 0

                    # create new sample dict
                    new_item = {
                        'img_path': item['img_path'],  
                        'grade': grade,
                        'mask': new_mask,
                        'mask_unfiltered': new_mask_unfiltered,
                        'depth': new_depth,
                        'image': new_img
                    }

                    augmented_data.append(new_item)

            # extend dataset
            self.data = self.data + augmented_data


        if hole_augmentation and self.split == 'train' and hole_augmentation_mode == 'augment':
            augmented_data = []
            for item in self.data:
                if item.get('has_hole', False) and random.random() < 0.9:
                    img_t, mask_t, depth_t, mask_unf_t = apply_random_transform(
                        item['image'], item['mask'], item['depth'], item['mask_unfiltered']
                    )
                    augmented_data.append({
                        'img_path': item['img_path'],
                        'grade': item['grade'],
                        'mask': mask_t,
                        'mask_unfiltered': mask_unf_t,
                        'depth': depth_t,
                        'image': img_t,
                        'has_hole': True,
                    })
            self.data = self.data + augmented_data

        if hole_augmentation and self.split == 'train' and hole_augmentation_mode == 'paste_hole':
            augmented_data = []
            _paste_save_dir = Path(__file__).parent / 'example_pastes_holes'
            _paste_save_dir.mkdir(parents=True, exist_ok=True)
            _n_saved, _max_save = 0, 20
            non_hole_items = [item for item in self.data if not item.get('has_hole', False)]
            if non_hole_items:
                for item in self.data:
                    if not item.get('has_hole', False):
                        continue
                    _, _, _, hole_pixels = find_holes_fix(
                        item['image'], item['mask'], item['depth'], dilation_radius = 20, return_hole=True
                    )
                    hole_mask = hole_pixels.sum(axis=2) > 0
                    if not hole_mask.any():
                        continue
                    targets = random.choices(non_hole_items, k=10)
                    for target in targets:
                        pasted_img = _paste_hole(target['image'], target['mask'], hole_pixels, hole_mask)
                        if pasted_img is None:
                            continue
                        if _n_saved < _max_save:
                            side_by_side = np.concatenate([target['image'], pasted_img], axis=1)
                            Image.fromarray(side_by_side.astype(np.uint8)).save(
                                _paste_save_dir / f'paste_{_n_saved:04d}.png'
                            )
                            _n_saved += 1
                        augmented_data.append({
                            'img_path': target['img_path'],
                            'grade': target['grade'],
                            'mask': target['mask'].copy(),
                            'mask_unfiltered': target['mask_unfiltered'].copy(),
                            'depth': target['depth'].copy(),
                            'image': pasted_img,
                            'has_hole': True,
                        })
            self.data = self.data + augmented_data


        self.img_paths = [item['img_path'] for item in self.data]
        self.grades = [item['grade'] for item in self.data]
        self.masks = [item['mask'] for item in self.data]
        self.depths = [item['depth'] for item in self.data]
        self.img_arrays = [item['image'] for item in self.data]
        self.masks_unfiltered = [item['mask_unfiltered'] for item in self.data]
        self.hole_booleans = [item.get('has_hole', False) for item in self.data]

 
 
    def __len__(self):
        return len(self.img_paths)

    
    def get_input_size(self):
        return (self.transform_sizes, self.transform_sizes)
 

    def __getitem__(self, idx):
        synthetic_added = False
        hole_added = False

        if self.synthetic_augmentation and self.synthetic_augmentation_mode == 'replace' and random.random() < 0.5:
            
         
            wanted_size_range = (random.randint(5,7) / 10), (random.randint(8,10) / 10)
 
            new_img, new_mask, grade = self.synthetic_occlusion.multi_raspberry_occlusion(
                wanted_size_range=wanted_size_range,
                randomize_scale_bool=(False, 0.5, 0.9),
                randomize_rotation_bool=(self.randomize_rotation, -30, 30),
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

                # Compute unfiltered mask before centering:
                # remove only the pixels that occlusion took from the filtered mask
                new_mask_unfiltered = self.masks_unfiltered[idx] & ~(self.masks[idx] & ~new_mask)

                # Save bbox reference from new_mask before centering, then center img+mask
                new_mask_pre_center = new_mask.copy()
                new_img, new_mask = _center_object(new_img, new_mask)

                # Apply the identical centering transform to new_mask_unfiltered
                coords = np.argwhere(new_mask_pre_center)
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    img_h, img_w = new_mask_pre_center.shape
                    obj_h, obj_w = y_max - y_min + 1, x_max - x_min + 1
                    paste_y = img_h // 2 - obj_h // 2
                    paste_x = img_w // 2 - obj_w // 2
                    src_y_start = max(0, -paste_y)
                    src_x_start = max(0, -paste_x)
                    src_y_end   = min(obj_h, img_h - paste_y)
                    src_x_end   = min(obj_w, img_w - paste_x)
                    dst_y_start, dst_x_start = max(0, paste_y), max(0, paste_x)
                    dst_y_end = dst_y_start + (src_y_end - src_y_start)
                    dst_x_end = dst_x_start + (src_x_end - src_x_start)
                    centered_unfiltered = np.zeros_like(new_mask_unfiltered)
                    centered_unfiltered[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        new_mask_unfiltered[y_min:y_max+1, x_min:x_max+1][src_y_start:src_y_end, src_x_start:src_x_end]
                    new_mask_unfiltered = centered_unfiltered

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
                mask_unfiltered = Image.fromarray(new_mask_unfiltered.astype(np.uint8) * 255)
 
        if (not synthetic_added
                and self.hole_augmentation and self.hole_augmentation_mode == 'replace'
                and self.hole_booleans[idx] and random.random() < 0.5):
            img_arr, mask_arr, depth_arr, mask_unf_arr = apply_random_transform(
                self.img_arrays[idx], self.masks[idx], self.depths[idx], self.masks_unfiltered[idx]
            )
            img = Image.fromarray(img_arr.astype(np.uint8))
            og_img = img_arr.copy()
            mask = Image.fromarray(mask_arr.astype(np.uint8) * 255)
            mask_unfiltered = Image.fromarray(mask_unf_arr.astype(np.uint8) * 255)
            og_mask = mask_arr.copy()
            depth = depth_arr
            og_depth = depth_arr.copy()
            hole_added = True

        if not synthetic_added and not hole_added:
            #img_file = self.img_paths[idx]
           # img = Image.open(img_file).convert("RGB")
            img = Image.fromarray(self.img_arrays[idx])
            og_img = self.img_arrays[idx].copy()
            mask = self.masks[idx]
            mask_unfiltered = self.masks_unfiltered[idx]
            og_mask = mask.copy()
            mask = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_unfiltered = Image.fromarray(mask_unfiltered.astype(np.uint8) * 255)
            depth = self.depths[idx]
            og_depth = depth.copy()
 
        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask_unfiltered = self.transform_mask(mask_unfiltered)

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
                return img, needed_grade, error_mask.int(), img_path, mask, actual_grade, mask_unfiltered, og_img, og_mask, og_depth
            else:
                return img, needed_grade, error_mask.int(), img_path, mask, actual_grade, mask_unfiltered, img, img, img # last 3 do not matter
        else:
            if self.struct_core_collection_bool:
                return img, mask, mask_unfiltered, og_img, og_mask, og_depth # I think here just add og_img, og_mask, og_depth for structcore ; would mess up custom_pretraining as it also uses the flag, but we do not need it anymore ...
            else:
                return img

def pretrain_backbone_cutout(dataset_path, device, backbone, save_path,
                             unfreeze_from=10, epochs=30, lr=1e-4,
                             batch_size=32, n_holes=3, hole_size_range=(32, 64),
                             target_path='full_no_filters',
                             unfreeze_last_n_blocks=2):
    """
    Self-supervised cutout pre-training for backbone fine-tuning.

    target_path : what dataset to train on

    Supported backbones:
      'wide_resnet50_2'       — unfreeze_from controls which of the 4 encoder stages is trained
                                (0=stem, 1=layer1, 2=layer2, 3=layer3; default 3 = only layer3)


    After training, encoder weights are saved to save_path in the torchvision key format
    so that CustomFeatureExtractor can load them via custom_weights_path with strict=False.
    """

    seed_everything(SEED)

    train_set_path = dataset_path / Path(f'{target_path}/processed')

    train_dataset = SingleRaspberryDataset(
        train_set_path, split='train',
        synthetic_augmentation=False,
        AD_model='patchcore', # just so we resize to 224x224
        backbone_model=backbone,
        struct_core_collection_bool=True,  # returns (img, mask) pairs
    )

    n_val = max(1, int(len(train_dataset) * 0.2)) # 20% val split
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=make_generator(SEED),
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        generator=make_generator(SEED), worker_init_fn=seed_worker,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        worker_init_fn=seed_worker,
    )
    print(f"Train: {n_train} samples | Val: {n_val} samples")


    model = create_cutout_model(backbone, device, unfreeze_from=unfreeze_from)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = torch.nn.MSELoss(reduction='none')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Cutout pretraining [{backbone}]: {epochs} epochs")
    print(f"Trainable: {trainable:,} | Frozen: {frozen:,}")

    trained_model = train_cutout_reconstruction(
        model, train_dataloader, device,
        epochs=epochs, lr=lr,
        optimizer=optimizer, criterion=criterion,
        n_holes=n_holes, hole_size_range=hole_size_range,
        val_loader=val_dataloader,
    )

    # Each model class knows its own key-mapping format for the save.
    trained_model.save_encoder_weights(save_path)
    print(f"Saved fine-tuned {backbone} encoder to {save_path}")

    del model, optimizer, train_dataloader, val_dataloader, train_dataset
    torch.cuda.empty_cache()

def train_model(dataset_path : str, backbone : str, ad_layers : list, save_path : str, device : torch.device, max_dataset_size : int = None, mode = 'patchcore', target_path = 'full_no_filters', pass_og_bool = False, custom_weights_path = None, synthetic_augmentation_bool = False, synthetic_augmentation_mode = 'replace', randomize_rotation_bool = False, scoring_mode = 'MAXMEAN_1', filter_post = 'NONE', mask_border_filter_thickness = 0, cls_token_viz_bool = False, hole_augmentation_bool = False, hole_augmentation_mode = 'replace', include_gt_fill_ins = True, batch_size_train = 8, epochs = 50, AD_only_on_mask = True, mask_dilation_radius = 0):

    seed_everything(SEED)

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

    train_dataset = SingleRaspberryDataset(train_set_path, split = 'train', synthetic_augmentation = synthetic_augmentation_bool, synthetic_augmentation_mode = synthetic_augmentation_mode, randomize_rotation = randomize_rotation_bool, AD_model = mode, backbone_model = backbone, hole_augmentation = hole_augmentation_bool, hole_augmentation_mode = hole_augmentation_mode)


    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True,
        generator=make_generator(SEED), worker_init_fn=seed_worker,
    )

    # Only anomalous samples for testing
    test_set_path = dataset_path  / Path(f'{target_path}/processed')
    test_dataset = SingleRaspberryDataset(test_set_path, split = 'test', synthetic_augmentation = False, AD_model = mode, backbone_model = backbone, pass_og_bool = pass_og_bool, include_gt_fill_ins = include_gt_fill_ins)

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, worker_init_fn=seed_worker,
    )

    print(f"Length train dataset: {len(train_dataset)}")
    print(f"Length test dataset: {len(test_dataset)}")

    input_size = train_dataset.get_input_size()
    

    # NOTE : Ganomaly & supersimplenet & STFPM have no device, therefore also run on cuda:0 ...
    if mode == 'patchcore': # k = 70000, num_neighbors = 500 // YOLO +WRN-50-2 layers2,3 has 3506832 total patches, so 1%-patchcore is 35068 , 3 neighbours is basic setting
        model = PatchCore(device, input_size= input_size, feature_extractor=feature_extractor, k = 35068, num_neighbors = 3, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, mask_border_filter_thickness = mask_border_filter_thickness, filter_post = filter_post, cls_token_viz_bool = cls_token_viz_bool, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'cfa':
        model = CFA(feature_extractor, backbone, device, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'stfpm':
        model = STFPM(teacher, student, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, protrusion_damping_radius = 0, protrusion_damping_gamma = 0, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'fastflow':
        model = create_fastflow(input_size, backbone, device, AD_only_on_mask = AD_only_on_mask, mask_border_filter_thickness = mask_border_filter_thickness, custom_weights_path = custom_weights_path, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'rd4ad':
        model = RD4AD(backbone, device, input_size = input_size, AD_only_on_mask = AD_only_on_mask, mask_border_filter_thickness = mask_border_filter_thickness, skip_layer1 = False, custom_weights_path = custom_weights_path, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'padim':
        diagonal_convergence = False
        model = Padim(backbone, class_name = 'raspberry', device = device, diag_cov = diagonal_convergence, layers_idxs = ad_layers, AD_only_on_mask = AD_only_on_mask, mask_border_filter_thickness = mask_border_filter_thickness, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'ganomaly':
        model = Ganomaly(input_size = input_size, num_input_channels = 3, n_features = 64, latent_vec_size = 100, extra_layers = 0, add_final_conv_layer = True)
    elif mode == 'supersimplenet':
        model = SuperSimpleNet(feature_extractor, AD_only_on_mask = AD_only_on_mask, mask_border_filter_thickness = mask_border_filter_thickness, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'sinbad':
        model = SINBAD(
            device=device,
            input_size=input_size,
            feature_extractor=feature_extractor,
            n_projections=1000,   # paper default
            n_quantiles=5,        # paper default
            shrinkage=0.1,        # paper default
            scoring_mode='knn',   # paper default
            use_raw_pixels = False,
            AD_only_on_mask = AD_only_on_mask,
            mask_dilation_radius = mask_dilation_radius,
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
        trainer.train(epochs = epochs, evaluation_epoch_interval=1)
    else:
        trainer.train()

    # save the model
    if save_path:

        # Can save at the very end since we do not have typical epoch training (i.e. do not need to save best results during training, just the one result at the end)
        if mode == 'patchcore':
            print("Saving model ...")
            model.save_model(save_path)
        if mode == 'padim':
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

def struct_core_collection(dataset_path : str, backbone : str, ad_layers : list, model_checkpoint_path : str, device : torch.device, max_dataset_size : int = None, mode = 'patchcore', target_path = 'full_no_filters', top_k_ratio = 0.01, filter_post = 'NONE', mask_border_filter_thickness = 0, protrusion_damping_radius = 0, protrusion_damping_gamma = 0, AD_only_on_mask = True, mask_dilation_radius = 0):
    """
    After creating the memory bank/ training a model, collect descriptors for StructCore based on training data
    """

    seed_everything(SEED)

    struct_core = StructCore(top_k_ratio = top_k_ratio)

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
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        generator=make_generator(SEED), worker_init_fn=seed_worker,
    )

    input_size = train_dataset.get_input_size()
    

    # load the model
    if mode == 'patchcore':
        model = PatchCore(device, input_size=input_size, feature_extractor=feature_extractor,k = 35068, num_neighbors = 3, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'stfpm':
        model = STFPM(teacher, student,  filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'cfa':
        model = CFA(feature_extractor, backbone, device, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma)
    elif mode == 'rd4ad':
        model = RD4AD(backbone, device, input_size = input_size, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'fastflow':
        model = create_fastflow(input_size, backbone, device, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'padim':
        diagonal_convergence = False
        model = Padim(backbone, class_name = 'raspberry', device = device, diag_cov = diagonal_convergence, layers_idxs = ad_layers, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'ganomaly':
        model = Ganomaly(input_size = input_size, num_input_channels = 3, n_features = 64, latent_vec_size = 100, extra_layers = 0, add_final_conv_layer = True, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'supersimplenet':
        model = SuperSimpleNet(feature_extractor, AD_only_on_mask = AD_only_on_mask, mask_border_filter_thickness = mask_border_filter_thickness, mask_dilation_radius = mask_dilation_radius)

    
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
    _param_dtypes = {p.dtype for p in model.parameters()}
    _buf_dtypes = {b.dtype for b in model.buffers() if b.numel() > 0}
    print(f"[{mode}] parameter dtypes: {_param_dtypes | _buf_dtypes}")

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

            # Use the effective_mask the model computed internally (includes dilation,
            # border-filter, and any model-specific resizing).  Models that don't set
            # this attribute (e.g. PatchCore whose scores are non-negative so the
            # background-zero issue doesn't apply) fall back to None.
            mask_for_struct = getattr(model, '_last_effective_mask', None)
            struct_core.accumulate(anomaly_maps, pred_scores, mask=mask_for_struct)

    train_descriptors_for_viz = torch.cat(struct_core._descriptors, dim=0).clone()
    train_scores_for_viz      = torch.cat(struct_core._base_scores,  dim=0).clone()
    struct_core.fit()


    print(f"StructCore train mean: {struct_core.mean}")
    print(f"StructCore train std: {struct_core.std}")
    print(f"StructCore auto_lambda: {struct_core.auto_lambda}")

  #  torch.save({
   #     'mean':        struct_core.mean,
   #     'std':         struct_core.std,
   #     'auto_lambda': struct_core.auto_lambda,
   # }, 'structcore_stats.pt')



    # Visualization purposes for Master Thesis (uncleanly added here and a complete mess since it was close to hand in... sorry for that, I have no time to rewrite this cleanly; 
    # I anyway only used it for some specific visualizations, so not that important)
    '''
    # --- Collect test descriptor statistics ---
    _DIFFUSE_IDS = frozenset([
        "img001_obj15", "img001_obj19", "img006_obj0", "img006_obj10", "img007_obj5",
        "img008_obj16", "img009_obj11", "img011_obj17", "img011_obj27", "img012_obj0",
        "img014_obj24", "img015_obj9", "img016_obj26", "img018_obj12", "img018_obj13",
        "img018_obj16", "img019_obj13", "img021_obj11", "img022_obj3", "img024_obj21",
        "img024_obj4", "img025_obj0", "img025_obj11", "img028_obj13", "img028_obj19",
        "img028_obj20", "img029_obj13", "img029_obj5", "img030_obj10", "img033_obj11",
        "img035_obj0", "img036_obj14", "img036_obj2", "img036_obj6", "img037_obj1",
        "img038_obj15", "img040_obj22", "img040_obj23", "img044_obj11", "img045_obj21",
        "img047_obj22", "img050_obj12", "img051_obj8", "img053_obj15", "img054_obj1",
        "img054_obj10", "img062_obj6", "img062_obj7", "img064_obj2", "img065_obj12",
        "img066_obj20", "img067_obj20", "img068_obj13", "img070_obj18", "img070_obj22",
        "img070_obj24", "img071_obj8", "img073_obj8", "img074_obj17", "img075_obj16",
        "img076_obj8", "img079_obj18", "img082_obj12", "img084_obj20", "img085_obj7",
        "img086_obj10", "img090_obj24", "img099_obj20", "img105_obj4", "img107_obj1",
        "img108_obj13", "img126_obj6", "img129_obj16", "img130_obj0", "img134_obj23",
        "img139_obj14", "img140_obj19", "img149_obj10", "img149_obj7", "img150_obj1",
        "img150_obj10", "img150_obj8", "img150_obj9", "img152_obj0", "img152_obj3",
        "img154_obj11", "img154_obj2", "img158_obj19", "img159_obj6", "img160_obj4",
        "img161_obj21", "img163_obj0", "img163_obj14", "img167_obj23", "img170_obj24",
        "img171_obj10", "img176_obj3", "img177_obj1", "img182_obj11", "img184_obj12",
        "img186_obj15", "img187_obj9", "img188_obj16", "img194_obj19", "img196_obj15",
        "img196_obj16", "img196_obj18", "img196_obj5", "img200_obj8",
    ])

    print("StructCore: collecting test descriptor statistics ...")
    test_set_path = dataset_path / Path(f'{target_path}/processed')
    test_dataset_sc = SingleRaspberryDataset(
        test_set_path, split='test', synthetic_augmentation=False,
        AD_model=mode, backbone_model=backbone, pass_og_bool=True,
    )
    if max_dataset_size is not None:
        test_dataset_sc = torch.utils.data.Subset(test_dataset_sc, range(max_dataset_size))
    test_dataloader_sc = torch.utils.data.DataLoader(
        test_dataset_sc, batch_size=4, shuffle=False, worker_init_fn=seed_worker,
    )

    test_descriptors = []
    test_scores_list  = []    # default image-level scores (pred_scores from model)
    test_labels_list = []    # binary label: 0 = non-anomalous, 1 = anomalous
    test_is_diffuse_list = []  # True if anomalous and sample_id in _DIFFUSE_IDS

    with torch.no_grad():
        for batch in tqdm(iter(test_dataloader_sc), desc="StructCore test"):
            images, labels, _masks, paths, full_mask, _actual_grade, mask_unfiltered, og_img, og_mask, og_depth = batch
            images = images.to(device)
            full_mask = full_mask.to(device)
            mask_unfiltered = mask_unfiltered.to(device)
            og_img = og_img.to(device)
            og_mask = og_mask.to(device)
            og_depth = og_depth.to(device)

            if mode == 'patchcore':
                anomaly_maps, pred_scores, _, _, _ = model((images, mask_unfiltered, full_mask, og_img, og_mask, og_depth))
            elif mode in ('stfpm', 'cfa', 'rd4ad', 'fastflow', 'supersimplenet'):
                anomaly_maps, pred_scores = model((images, full_mask, mask_unfiltered, og_img, og_mask, og_depth))
            else:
                anomaly_maps, pred_scores = model(images)

            test_scores_list.append(pred_scores.detach().cpu())

            mask_for_struct = getattr(model, '_last_effective_mask', None)
            desc = struct_core._compute_descriptor(anomaly_maps, mask=mask_for_struct)
            test_descriptors.append(desc.detach().cpu())

            for b in range(desc.shape[0]):
                lbl = int(labels[b].item())
                # img_path basename: img{X}_obj{Y}_grade{Z}.png → sample_id = img{X}_obj{Y}
                stem = Path(paths[b]).stem
                sample_id = '_'.join(stem.split('_')[:2])
                test_labels_list.append(lbl)
                test_is_diffuse_list.append(lbl == 1 and sample_id in _DIFFUSE_IDS)

    all_test_descriptors = torch.cat(test_descriptors, dim=0)
    test_labels_t    = torch.tensor(test_labels_list, dtype=torch.bool)
    test_is_diffuse_t = torch.tensor(test_is_diffuse_list, dtype=torch.bool)

    def _desc_stats(name, mask_bool):
        sub = all_test_descriptors[mask_bool]
        n = sub.shape[0]
        if n == 0:
            print(f"  {name}: no samples")
            return
        print(f"  {name} (n={n}):")
        print(f"    mean: {sub.mean(dim=0)}")
        print(f"    std:  {sub.std(dim=0)}")

    print("StructCore test descriptor statistics:")
    _desc_stats("non-anomalous",       ~test_labels_t)
    _desc_stats("anomalous",            test_labels_t)
    _desc_stats("anomalous (diffuse)",  test_is_diffuse_t)
    _desc_stats("anomalous (non-diff)", test_labels_t & ~test_is_diffuse_t)

    # --- Distribution visualizations ---
    _DIM_NAMES = ["sigma", "topk_mean", "tv"]
    n_dims = all_test_descriptors.shape[1]
    dim_names = _DIM_NAMES[:n_dims]

    import re as _re
    _seed_m = _re.search(r'seed_(\d+)', target_path)
    _seed_tag = f"_seed_{_seed_m.group(1)}" if _seed_m else ""
    viz_dir = Path(__file__).parent / 'structcore_viz' / f'{mode}_{backbone}{_seed_tag}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    normal_test_mask = ~test_labels_t
    diffuse_mask     = test_is_diffuse_t
    non_diff_mask    = test_labels_t & ~test_is_diffuse_t

    # Global per-dimension x-range across all groups so every figure is comparable.
    _all_groups = [train_descriptors_for_viz, all_test_descriptors]
    _combined = torch.cat(_all_groups, dim=0)
    _xlims = [
        (float(_combined[:, i].min()), float(_combined[:, i].max()))
        for i in range(n_dims)
    ]

    def _save_dist_fig(descriptors: torch.Tensor, title: str, filename: str) -> None:
        fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4), squeeze=False)
        axes = axes[0]
        for i, (ax, dname) in enumerate(zip(axes, dim_names)):
            vals = descriptors[:, i].numpy()
            ax.hist(vals, bins=30, alpha=0.8, edgecolor='white', color='#4C72B0',
                    range=_xlims[i])
            mean_v, std_v = float(vals.mean()), float(vals.std())
            ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5,
                       label=f'mean={mean_v:.4f}')
            ax.axvline(mean_v - std_v, color='orange', linestyle=':', linewidth=1.2)
            ax.axvline(mean_v + std_v, color='orange', linestyle=':', linewidth=1.2,
                       label=f'±std={std_v:.4f}')
            ax.set_xlim(_xlims[i])
            ax.set_title(dname)
            ax.set_xlabel('value')
            ax.set_ylabel('count')
            ax.legend(fontsize=8)
        n = descriptors.shape[0]
        fig.suptitle(f'{title}  (n={n})', fontsize=11)
        plt.tight_layout()
        plt.savefig(viz_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    _save_dist_fig(train_descriptors_for_viz,
                   'StructCore descriptors — train (normal)',
                   'train_normal.png')
    if normal_test_mask.any():
        _save_dist_fig(all_test_descriptors[normal_test_mask],
                       'StructCore descriptors — test (normal)',
                       'test_normal.png')
    if diffuse_mask.any():
        _save_dist_fig(all_test_descriptors[diffuse_mask],
                       'StructCore descriptors — test (diffuse anomalous)',
                       'test_diffuse.png')
    if non_diff_mask.any():
        _save_dist_fig(all_test_descriptors[non_diff_mask],
                       'StructCore descriptors — test (non-diffuse anomalous)',
                       'test_non_diffuse.png')
    if test_labels_t.any():
        _save_dist_fig(all_test_descriptors[test_labels_t],
                       'StructCore descriptors — test (all anomalous)',
                       'test_anomalous.png')

    print(f"StructCore distribution figures saved to {viz_dir}")

    # --- Default-scoring distribution visualizations ---
    all_test_scores = torch.cat(test_scores_list, dim=0)  # (N_test,)
    _score_xlim = (
        float(min(train_scores_for_viz.min(), all_test_scores.min())),
        float(max(train_scores_for_viz.max(), all_test_scores.max())),
    )

    default_viz_dir = Path(__file__).parent / 'default_viz' / f'{mode}_{backbone}{_seed_tag}'
    default_viz_dir.mkdir(parents=True, exist_ok=True)

    def _save_score_fig(scores: torch.Tensor, title: str, filename: str) -> None:
        vals = scores.numpy()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(vals, bins=30, alpha=0.8, edgecolor='white', color='#4C72B0',
                range=_score_xlim)
        mean_v, std_v = float(vals.mean()), float(vals.std())
        ax.axvline(mean_v, color='red', linestyle='--', linewidth=1.5,
                   label=f'mean={mean_v:.4f}')
        ax.axvline(mean_v - std_v, color='orange', linestyle=':', linewidth=1.2)
        ax.axvline(mean_v + std_v, color='orange', linestyle=':', linewidth=1.2,
                   label=f'±std={std_v:.4f}')
        ax.set_xlim(_score_xlim)
        ax.set_xlabel('anomaly score')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        n = len(vals)
        fig.suptitle(f'{title}  (n={n})', fontsize=11)
        plt.tight_layout()
        plt.savefig(default_viz_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    _save_score_fig(train_scores_for_viz, 'Default scores — train (normal)', 'train_normal.png')
    if normal_test_mask.any():
        _save_score_fig(all_test_scores[normal_test_mask], 'Default scores — test (normal)', 'test_normal.png')
    if diffuse_mask.any():
        _save_score_fig(all_test_scores[diffuse_mask], 'Default scores — test (diffuse anomalous)', 'test_diffuse.png')
    if non_diff_mask.any():
        _save_score_fig(all_test_scores[non_diff_mask], 'Default scores — test (non-diffuse anomalous)', 'test_non_diffuse.png')

    print(f"Default scoring distribution figures saved to {default_viz_dir}")

    # --- Comparison: StructCore vs default (raw scores) ---
    from scipy.stats import gaussian_kde as _kde

    _sc_mean  = struct_core.mean.cpu()
    _sc_std   = struct_core.std.cpu()
    _auto_lam = struct_core.auto_lambda.cpu()
    sc_train_d = ((train_descriptors_for_viz - _sc_mean) / _sc_std).norm(dim=1)
    sc_test_d  = ((all_test_descriptors      - _sc_mean) / _sc_std).norm(dim=1)

    sc_hybrid_train = train_scores_for_viz + _auto_lam * sc_train_d
    sc_hybrid_test  = all_test_scores      + _auto_lam * sc_test_d

    # Raw score dicts per group — no standardisation needed (SC hybrid and default share S_base)
    r_sc = {
        'train': sc_hybrid_train.numpy(),
        'norm':  sc_hybrid_test[normal_test_mask].numpy(),
        'diff':  sc_hybrid_test[diffuse_mask].numpy(),
        'ndiff': sc_hybrid_test[non_diff_mask].numpy(),
    }
    r_scd = {
        'train': sc_train_d.numpy(),
        'norm':  sc_test_d[normal_test_mask].numpy(),
        'diff':  sc_test_d[diffuse_mask].numpy(),
        'ndiff': sc_test_d[non_diff_mask].numpy(),
    }
    r_df = {
        'train': train_scores_for_viz.numpy(),
        'norm':  all_test_scores[normal_test_mask].numpy(),
        'diff':  all_test_scores[diffuse_mask].numpy(),
        'ndiff': all_test_scores[non_diff_mask].numpy(),
    }

    _groups = [
        ('train normal', 'train', '#4C72B0'),
        ('test normal',  'norm',  '#55A868'),
        ('test diffuse', 'diff',  '#C44E52'),
        ('test non-diff','ndiff', '#DD8452'),
    ]

    comparison_viz_dir = Path(__file__).parent / 'comparison_viz' / f'{mode}_{backbone}{_seed_tag}'
    comparison_viz_dir.mkdir(parents=True, exist_ok=True)

    # Helpers
    def _mean(arr): return float(arr.mean())    if len(arr) > 0 else float('nan')
    def _med(arr):  return float(np.median(arr)) if len(arr) > 0 else float('nan')
    def _std(arr):  return float(arr.std())     if len(arr) > 1 else float('nan')

    def _auroc(pos, neg):
        """AUROC via Wilcoxon rank-sum — P(score_pos > score_neg)."""
        if len(pos) == 0 or len(neg) == 0:
            return float('nan')
        all_s  = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        ranks  = np.argsort(np.argsort(all_s)) + 1
        return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))

    # Overlay: 2-panel KDE (SC hybrid vs Default) on shared x and y scale
    fig, (ax_sc, ax_df) = plt.subplots(1, 2, figsize=(14, 5))

    # Shared x-limits: union of all values across both methods
    _all_overlay_vals = np.concatenate(list(r_sc.values()) + list(r_df.values()))
    _ov_x_lo = float(_all_overlay_vals.min())
    _ov_x_hi = float(_all_overlay_vals.max())
    _ov_xs   = np.linspace(_ov_x_lo, _ov_x_hi, 500)

    # Compute all KDE densities first so we can find the shared y-limit
    _ov_y_hi = 0.0
    _ov_curves = {ax_sc: [], ax_df: []}
    for ax, r_dict in [(ax_sc, r_sc), (ax_df, r_df)]:
        for label, key, color in _groups:
            vals = r_dict[key]
            if len(vals) < 2:
                continue
            dens = _kde(vals)(_ov_xs)
            _ov_y_hi = max(_ov_y_hi, float(dens.max()))
            _ov_curves[ax].append((label, key, color, vals, dens))

    _ov_y_hi *= 1.05  # 5% headroom

    for ax, r_dict, title in [(ax_sc, r_sc, 'StructCore scoring'),
                               (ax_df, r_df, 'Default scoring')]:
        for label, key, color, vals, dens in _ov_curves[ax]:
            ax.plot(_ov_xs, dens, color=color, linewidth=2, label=f'{label} (n={len(vals)})')
            ax.axvline(float(vals.mean()), color=color, linestyle='--', linewidth=1, alpha=0.6)
        ax.set_title(title)
        ax.set_xlim(_ov_x_lo, _ov_x_hi)
        ax.set_ylim(0, _ov_y_hi)
        ax.set_xlabel('raw score')
        ax.set_ylabel('density')
        ax.set_yticks([])
        ax.legend(fontsize=8)

    fig.suptitle(
        f'{mode} · {backbone}{_seed_tag}  —  score distributions (raw)',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(comparison_viz_dir / 'overlay.png', dpi=150, bbox_inches='tight')
    plt.close()

    # AUROC per subset
    auroc_data = {}
    for _anom_label, _anom_key in [('all anoms vs norm', None),
                                    ('diffuse vs norm',   'diff'),
                                    ('non-diff vs norm',  'ndiff')]:
        for r_dict, tag in [(r_sc, 'sc'), (r_scd, 'scd'), (r_df, 'df')]:
            pos = np.concatenate([r_dict['diff'], r_dict['ndiff']]) \
                  if _anom_key is None else r_dict[_anom_key]
            auroc_data[(_anom_label, tag)] = _auroc(pos, r_dict['norm'])

    # Console output
    print(f"\n  AUROC (anomaly vs test-normal, raw scores):")
    print(f"  {'':22} {'SC hybrid':>10}  {'D_struct':>10}  {'Default':>10}")
    for _al in ['all anoms vs norm', 'diffuse vs norm', 'non-diff vs norm']:
        print(f"  {_al:<22} "
              f"{auroc_data[(_al, 'sc')]:>10.3f}  "
              f"{auroc_data[(_al, 'scd')]:>10.3f}  "
              f"{auroc_data[(_al, 'df')]:>10.3f}")

    print(f"\n  Raw scores — mean (±std) / median (±std):")
    print(f"  {'Group':22}  {'SC hybrid':^48}  {'D_struct':^48}  {'Default':^48}")
    for _lbl, _key, _ in _groups:
        def _fmt(r, k=_key):
            return f"μ={_mean(r[k]):.3f}(±{_std(r[k]):.3f})  med={_med(r[k]):.3f}(±{_std(r[k]):.3f})"
        print(f"  {_lbl:<22}  {_fmt(r_sc):<48}  {_fmt(r_scd):<48}  {_fmt(r_df):<48}")

    print(f"\n  Effective separation (test group − test normal):")
    print(f"  {'':22}  {'SC hybrid':^28}  {'D_struct':^28}  {'Default':^28}")
    for _lbl, _key in [('test diffuse', 'diff'), ('test non-diff', 'ndiff')]:
        def _dfmt(r, k=_key):
            return f"Δμ={_mean(r[k])-_mean(r['norm']):+.3f}  Δmed={_med(r[k])-_med(r['norm']):+.3f}"
        print(f"  {_lbl:<22}  {_dfmt(r_sc):<28}  {_dfmt(r_scd):<28}  {_dfmt(r_df):<28}")

    # Summary figure: AUROC (top / primary) + raw stats (bottom / secondary)
    fig_s, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 9.5))
    for ax in (ax_top, ax_bot):
        ax.axis('off')

    # Top: AUROC table
    auroc_tbl_rows = [
        [_al,
         f"{auroc_data[(_al, 'sc')]:.3f}",
         f"{auroc_data[(_al, 'scd')]:.3f}",
         f"{auroc_data[(_al, 'df')]:.3f}"]
        for _al in ['all anoms vs norm', 'diffuse vs norm', 'non-diff vs norm']
    ]
    tbl_top = ax_top.table(
        cellText=auroc_tbl_rows,
        colLabels=['AUROC subset', 'SC hybrid', 'D_struct', 'Default'],
        loc='center', cellLoc='center',
    )
    tbl_top.auto_set_font_size(False)
    tbl_top.set_fontsize(9)
    tbl_top.scale(1.2, 1.9)
    ax_top.set_title('AUROC — anomaly vs test-normal  (primary metric)', fontsize=10, pad=4)

    # Bottom: mean (±std) / median (±std) per group + effective separation rows
    def _cell(r, key):
        return f"μ={_mean(r[key]):.3f}(±{_std(r[key]):.3f})  med={_med(r[key]):.3f}(±{_std(r[key]):.3f})"

    def _diff_cell(r, key):
        return f"Δμ={_mean(r[key]) - _mean(r['norm']):+.3f}  Δmed={_med(r[key]) - _med(r['norm']):+.3f}"

    stat_tbl_rows = [
        [_lbl, _cell(r_sc, _key), _cell(r_scd, _key), _cell(r_df, _key)]
        for _lbl, _key, _ in _groups
    ]
    stat_tbl_rows.append(['diff − test_norm',
                          _diff_cell(r_sc, 'diff'), _diff_cell(r_scd, 'diff'), _diff_cell(r_df, 'diff')])
    stat_tbl_rows.append(['ndiff − test_norm',
                          _diff_cell(r_sc, 'ndiff'), _diff_cell(r_scd, 'ndiff'), _diff_cell(r_df, 'ndiff')])
    tbl_bot = ax_bot.table(
        cellText=stat_tbl_rows,
        colLabels=['Group', 'SC hybrid  (μ±σ / med±σ)', 'D_struct  (μ±σ / med±σ)', 'Default  (μ±σ / med±σ)'],
        loc='center', cellLoc='center',
    )
    tbl_bot.auto_set_font_size(False)
    tbl_bot.set_fontsize(8.5)
    tbl_bot.scale(1.2, 1.7)
    ax_bot.set_title('Raw score statistics — mean±std / median±std per group  +  effective separation vs train-normal',
                     fontsize=10, pad=4)

    fig_s.suptitle(
        f"{mode} · {backbone}{_seed_tag}  —  StructCore vs Default comparison",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(comparison_viz_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Comparison figures saved to {comparison_viz_dir}")
    '''


    return struct_core

def test_model(dataset_path : str, backbone : str, ad_layers : list, model_checkpoint_path : str, device : torch.device, max_dataset_size : int = None, visual_test_path: str = None, mode = 'patchcore', scoring_mode = 'MAXMEAN_1', filter_post = 'NONE', target_path = 'full_no_filters', mask_border_filter_thickness = 1, pass_og_bool = False, custom_weights_path = None, cls_token_viz_bool = False, top_k_ratio = 0.01, protrusion_damping_radius = 0, protrusion_damping_gamma = 0, include_gt_fill_ins = True, AD_only_on_mask = True, mask_dilation_radius = 0):

    seed_everything(SEED)

    if scoring_mode == 'STRUCTCORE':
        print("StructCore collection ...")
        struct_core = struct_core_collection(dataset_path, backbone, ad_layers, model_checkpoint_path, device, max_dataset_size, mode, target_path, top_k_ratio = top_k_ratio, mask_border_filter_thickness = mask_border_filter_thickness, filter_post = filter_post, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
        print("StructCore collection done")

        
    # initialize the feature extractor
    if mode != 'stfpm':
        feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, custom_weights_path = custom_weights_path)
    else:
        teacher = CustomFeatureExtractor(backbone, ad_layers, device, frozen = True, custom_weights_path = custom_weights_path)
        student = CustomFeatureExtractor(backbone, ad_layers, device, frozen = False)


    # Only anomalous samples for testing
    test_set_path = dataset_path  / Path(f'{target_path}/processed')
    test_dataset = SingleRaspberryDataset(test_set_path, split = 'test', synthetic_augmentation = False, AD_model = mode, backbone_model = backbone, pass_og_bool = pass_og_bool, include_gt_fill_ins = include_gt_fill_ins)

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, worker_init_fn=seed_worker,
    )


    input_size = test_dataset.get_input_size()

    model_load_start_time = time()
    # load the model
    if mode == 'patchcore':
        model = PatchCore(device, input_size=input_size, feature_extractor=feature_extractor, k = 35068, num_neighbors = 3, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, mask_border_filter_thickness = mask_border_filter_thickness, filter_post = filter_post, cls_token_viz_bool = cls_token_viz_bool, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'cfa':
        model = CFA(feature_extractor, backbone, device, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma)
    elif mode == 'stfpm':
        model = STFPM(teacher, student, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, protrusion_damping_radius = protrusion_damping_radius, protrusion_damping_gamma = protrusion_damping_gamma, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'rd4ad':
        model = RD4AD(backbone, device, input_size = input_size, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, skip_layer1 = False, custom_weights_path = custom_weights_path, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'fastflow':
        model = create_fastflow(input_size, backbone, device, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, custom_weights_path = custom_weights_path, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'padim':
        diagonal_convergence = False
        model = Padim(backbone, class_name = 'raspberry', device = device, diag_cov = diagonal_convergence, layers_idxs = ad_layers, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, mask_border_filter_thickness = mask_border_filter_thickness, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'ganomaly':
        model = Ganomaly(input_size = input_size, num_input_channels = 3, n_features = 64, latent_vec_size = 100, extra_layers = 0, add_final_conv_layer = True, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, filter_post = filter_post, AD_only_on_mask = AD_only_on_mask, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'supersimplenet':
        model = SuperSimpleNet(feature_extractor, AD_only_on_mask = AD_only_on_mask, mask_border_filter_thickness = mask_border_filter_thickness, struct_core_instance = struct_core if scoring_mode == 'STRUCTCORE' else None, scoring_mode = scoring_mode, mask_dilation_radius = mask_dilation_radius)
    elif mode == 'sinbad':
        model = SINBAD(device=device, input_size=input_size, feature_extractor=feature_extractor, n_projections=1000, n_quantiles=5, shrinkage=0.1, scoring_mode='knn', AD_only_on_mask=AD_only_on_mask, mask_dilation_radius=mask_dilation_radius)

    

    if mode == 'patchcore' or mode == 'cfa':
        print(model_checkpoint_path)
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

    model_load_end_time = time()
    print(f"Model loading time: {model_load_end_time - model_load_start_time:.4f} seconds")



    # length of state dict
   # state_dict = torch.load(model_checkpoint_path, map_location=device)
   # print(f"Length of state dict: {len(state_dict)}")



  

    model.to(device)
    model.eval()
    _param_dtypes = {p.dtype for p in model.parameters()}
    _buf_dtypes = {b.dtype for b in model.buffers() if b.numel() > 0}
    print(f"[{mode}] parameter dtypes: {_param_dtypes | _buf_dtypes}")

    evaluator = Evaluator(test_dataloader, device)
    metrics, opt_thresh = evaluator.evaluate(model) #  opt_thresh 


    # PRINTS FOR SIZES
    
    '''
    if mode == 'cfa' or mode == 'patchcore':
        # Already implemented in MVTEC ad

        sizes, total_size = model.get_model_size_and_macs()

        print(f"SIZES : {sizes}")
        print(f"TOTAL SIZE : {total_size}")

    # Needed to compute sizes for other models here (also close to hand-in, so didn't add it into the model classes themselves...)
    elif mode == 'sinbad':
        def _tensor_mb(t):
            return t.numel() * t.element_size() / 1e6

        def _numpy_mb(a):
            return a.nbytes / 1e6

        def _module_mb(m):
            return (
                sum(p.numel() * p.element_size() for p in m.parameters()) +
                sum(b.numel() * b.element_size() for b in m.buffers())
            ) / 1e6

        backbone_mb = _module_mb(model.feature_extractor.model)

        proj_mb, thresh_mb, cov_mb, desc_mb = 0.0, 0.0, 0.0, 0.0
        sfe = model.set_feature_extractor
        if sfe is not None:
            proj_mb = _tensor_mb(sfe.projections)
            if sfe.min_vals is not None:
                thresh_mb = _tensor_mb(sfe.min_vals) + _tensor_mb(sfe.max_vals)

        sc = model.scorer
        if sc is not None:
            if sc.cov_inv is not None:
                cov_mb = _numpy_mb(sc.cov_inv)
            if sc.train_mean is not None:
                cov_mb += _numpy_mb(sc.train_mean)
            if sc.train_descriptors_whitened is not None:
                desc_mb = _numpy_mb(sc.train_descriptors_whitened)

        total_mb = backbone_mb + proj_mb + thresh_mb + cov_mb + desc_mb

        print(f"backbone (frozen):            {backbone_mb:.2f} MB")
        print(f"random projections:           {proj_mb:.2f} MB")
        print(f"CDF thresholds (min/max):     {thresh_mb:.4f} MB")
        print(f"covariance inverse + mean:    {cov_mb:.2f} MB")
        print(f"training descriptors (knn):   {desc_mb:.2f} MB")
        print(f"total:                        {total_mb:.2f} MB")

    elif mode == 'fastflow':
        def _module_mb(m):
            return (
                sum(p.numel() * p.element_size() for p in m.parameters()) +
                sum(b.numel() * b.element_size() for b in m.buffers())
            ) / 1e6

      
        backbone_mb = _module_mb(model.feature_extractor)
        flow_mb = _module_mb(model.fast_flow_module)
        norms_mb = _module_mb(model.norms) if hasattr(model, 'norms') else 0.0
        total_mb = backbone_mb + flow_mb + norms_mb

        print(f"backbone (frozen):            {backbone_mb:.2f} MB")
        print(f"normalizing flow blocks:      {flow_mb:.2f} MB")
        print(f"layer norms:                  {norms_mb:.4f} MB")
        print(f"total:                        {total_mb:.2f} MB")

    elif mode == 'supersimplenet':
        def _module_mb(m):
            return (
                sum(p.numel() * p.element_size() for p in m.parameters()) +
                sum(b.numel() * b.element_size() for b in m.buffers())
            ) / 1e6

    
        backbone_mb = _module_mb(model.feature_extractor.feature_extractor.model)
        adaptor_mb = _module_mb(model.adaptor)
        segdec_mb = _module_mb(model.segdec)
        total_mb = backbone_mb + adaptor_mb + segdec_mb

        print(f"backbone (frozen):            {backbone_mb:.2f} MB")
        print(f"adaptor conv:                 {adaptor_mb:.4f} MB")
        print(f"discriminator head:           {segdec_mb:.2f} MB")
        print(f"total:                        {total_mb:.2f} MB")

    elif mode == 'ganomaly':
        def _module_mb(m):
            return (
                sum(p.numel() * p.element_size() for p in m.parameters()) +
                sum(b.numel() * b.element_size() for b in m.buffers())
            ) / 1e6

        
        gen_mb = _module_mb(model.generator)
        disc_mb = _module_mb(model.discriminator)
        total_mb = gen_mb + disc_mb

        print(f"generator (enc1+dec+enc2):    {gen_mb:.2f} MB")
        print(f"discriminator:                {disc_mb:.2f} MB")
        print(f"total:                        {total_mb:.2f} MB")

    elif mode == 'stfpm':
        def _module_mb(m):
            return (
                sum(p.numel() * p.element_size() for p in m.parameters()) +
                sum(b.numel() * b.element_size() for b in m.buffers())
            ) / 1e6

        
        teacher_mb = _module_mb(model.teacher.model)
        student_mb = _module_mb(model.student.model)
        total_mb = teacher_mb + student_mb

        print(f"teacher (frozen):             {teacher_mb:.2f} MB")
        print(f"student (learnable):          {student_mb:.2f} MB")
        print(f"total:                        {total_mb:.2f} MB")

    elif mode == 'rd4ad':
        def _module_mb(m):
            return (
                sum(p.numel() * p.element_size() for p in m.parameters()) +
                sum(b.numel() * b.element_size() for b in m.buffers())
            ) / 1e6

        encoder_mb = _module_mb(model.encoder)
        bn_mb = _module_mb(model.bn)
        decoder_mb = _module_mb(model.decoder)
        total_mb = encoder_mb + bn_mb + decoder_mb

        print(f"encoder (frozen):             {encoder_mb:.2f} MB")
        print(f"bottleneck (learnable):       {bn_mb:.2f} MB")
        print(f"decoder (learnable):          {decoder_mb:.2f} MB")
        print(f"total:                        {total_mb:.2f} MB")

    else:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_mb = (param_bytes + buffer_bytes) / 1e6

        print(f"params: {param_bytes / 1e6:.2f} MB")
        print(f"buffers: {buffer_bytes / 1e6:.2f} MB")
        print(f"total: {total_mb:.2f} MB")

        for name, buf in model.named_buffers():
            print(name, buf.shape, f"{buf.numel() * buf.element_size() / 1e6:.2f} MB")
    '''


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


    opt_threshold = opt_thresh


    # chek for the visual test
    if visual_test_path:

        # Get output directory.
        dirpath = pathlib.Path(visual_test_path)
        dirpath.mkdir(parents=True, exist_ok=True)

        import json
        with open(dirpath / "threshold.json", "w") as f:
            json.dump({"optimal_threshold": float(opt_threshold)}, f)

        all_pred_scores_non_anomalous = []
        all_pred_scores_anomalous = []
        pred_scores_per_grade = [[] for _ in range(5)]
        mask_size_wrong_predictions = []
        mask_size_correct_predictions = []
        for images, labels, masks, paths, full_mask, actual_grade, mask_unfiltered, og_img, og_mask, og_depth in tqdm(iter(test_dataloader)):
            if mode == 'patchcore':
                anomaly_maps, pred_scores, _ , _, cls_tokens = model((images.to(device), mask_unfiltered.to(device), full_mask.to(device), og_img.to(device), og_mask.to(device), og_depth.to(device)))
            elif mode == 'stfpm' or mode == 'cfa' or mode == 'rd4ad' or mode == 'fastflow' or mode == 'supersimplenet':
                anomaly_maps, pred_scores = model((images.to(device), full_mask.to(device), mask_unfiltered.to(device), og_img.to(device), og_mask.to(device), og_depth.to(device)))
            elif mode == 'sinbad':
                output = model((images.to(device), full_mask.to(device), mask_unfiltered.to(device), og_img.to(device), og_mask.to(device), og_depth.to(device)))
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
    Filename format: {actual}_PRED_{predicted}_img{X}_obj{Y}_grade{Z}.png
    """

    results = []
    for fname in os.listdir(data_path):
        if not fname.startswith('1') and not fname.startswith('0'):
            continue
        
        parts = fname.split('_')
        actual = int(parts[0])
        predicted = int(parts[2])
        grade = int(parts[-1].replace('grade', '').replace('.png', ''))
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
    # Based on average of img_roc_auc, img_f1 and img_pr_auc ; if this average is better for the new metrics, we save the new model
    # NOTE: In Thesis only AUROC used! So depending on what saving criteria is wanted, change here
    if (new_metrics["img_roc_auc"] + new_metrics["img_f1"] + new_metrics["img_pr_auc"]) / 3 > (best_metrics["img_roc_auc"] + best_metrics["img_f1"] + best_metrics["img_pr_auc"]) / 3:
        return True
     
    return False


def main():

    # Loading datasets here...
   # datasets_og = ['full_no_filters_seed_0_yolo_640_shared_test_set_256','full_no_filters_seed_1_yolo_640_shared_test_set_256', 'full_no_filters_seed_42_yolo_640_shared_test_set_256'] # 
   # datasets_gt = ['full_no_filters_seed_0_gt_256', 'full_no_filters_seed_1_gt_256', 'full_no_filters_seed_42_gt_256']
    # 'full_no_filters_seed_0_gt_256', 'full_no_filters_seed_1_gt_256', 'full_no_filters_seed_42_gt_256'
  #  datasets = ['filtered_darkness_80_0.3_seed_0_yolo_640_shared_test_set_256', 'filtered_darkness_80_0.3_seed_1_yolo_640_shared_test_set_256', 'filtered_darkness_80_0.3_seed_42_yolo_640_shared_test_set_256']
  #  datasets = ['filtered_unblurred_seed_0_yolo_640_shared_test_set_256', 'filtered_unblurred_seed_1_yolo_640_shared_test_set_256', 'filtered_unblurred_seed_42_yolo_640_shared_test_set_256']
    datasets = ['filtered_clean_protrusions_seed_0_yolo_640_shared_test_set_256', 'filtered_clean_protrusions_seed_1_yolo_640_shared_test_set_256', 'filtered_clean_protrusions_seed_42_yolo_640_shared_test_set_256']
# datasets = ['filtered_specular_suppression_seed_0_yolo_640_shared_test_set_256', 'filtered_specular_suppression_seed_1_yolo_640_shared_test_set_256', 'filtered_specular_suppression_seed_42_yolo_640_shared_test_set_256']
  #  datasets = ['filtered_unblurred_and_clean_protrusions_seed_0_yolo_640_shared_test_set_256', 'filtered_unblurred_and_clean_protrusions_seed_1_yolo_640_shared_test_set_256', 'filtered_unblurred_and_clean_protrusions_seed_42_yolo_640_shared_test_set_256'] # , 
  #  datasets_ssn = ['filtered_unblurred_and_specular_suppression_and_clean_protrusions_seed_0_yolo_640_shared_test_set_256', 'filtered_unblurred_and_specular_suppression_and_clean_protrusions_seed_1_yolo_640_shared_test_set_256', 'filtered_unblurred_and_specular_suppression_and_clean_protrusions_seed_42_yolo_640_shared_test_set_256']
   # all_filter_datasets = ['filtered_darkness_80_0.3_seed_0_yolo_640_shared_test_set_256', 'filtered_darkness_80_0.3_seed_1_yolo_640_shared_test_set_256', 'filtered_darkness_80_0.3_seed_42_yolo_640_shared_test_set_256', 'filtered_unblurred_seed_0_yolo_640_shared_test_set_256', 'filtered_unblurred_seed_1_yolo_640_shared_test_set_256', 'filtered_unblurred_seed_42_yolo_640_shared_test_set_256', 'filtered_clean_protrusions_seed_0_yolo_640_shared_test_set_256', 'filtered_clean_protrusions_seed_1_yolo_640_shared_test_set_256', 'filtered_clean_protrusions_seed_42_yolo_640_shared_test_set_256', 'filtered_specular_suppression_seed_0_yolo_640_shared_test_set_256', 'filtered_specular_suppression_seed_1_yolo_640_shared_test_set_256', 'filtered_specular_suppression_seed_42_yolo_640_shared_test_set_256']
  #  datasets = ['filtered_darkness_80_0.3_and_unblurred_and_clean_protrusions_seed_0_yolo_640_shared_test_set_256', 'filtered_darkness_80_0.3_and_unblurred_and_clean_protrusions_seed_1_yolo_640_shared_test_set_256', 'filtered_darkness_80_0.3_and_unblurred_and_clean_protrusions_seed_42_yolo_640_shared_test_set_256']

    for i in range(3):

        MODEL_MODE = 'rd4ad' # 'patchcore', 'cfa', 'stfpm', 'rd4ad', 'fastflow', 'padim', 'ganomaly', 'supersimplenet', 'sinbad'
        SYN_AUG_BOOL = False # whether to use synthetic occlusions during training
        SYN_AUG_MODE = 'augment' # 'replace' or 'augment' ('replace' is 'on-the-fly' in thesis, 'augment' is fixed)
        RANDOMIZE_ROTATION_BOOL = False # whether to randomize rotation in synthetic occlusions
        HOLE_AUG_BOOL = False # Whether to use synthetic holes during training (I wrote about this in the conclusion/model specific analysis, saying that it essentially didn't work)
        HOLE_AUG_MODE = 'paste_hole'

        BATCH_SIZE_TRAIN = 16
        EPOCHS = 50
        AD_ONLY_ON_MASK = True # Whether to only look at the AD heatmap within the raspberry mask
        MASK_DILATION_RADIUS = 10 # pixels to expand the mask outward before applying to the anomaly map # 2 for Patchcore, else 10

        # NOTE : this only works with patchcore + dinov2 
        CLS_TOKEN_VIZ_BOOL = False # Implemented for testing purposes (understanding whether CLS token can be used for distinguishing better between different raspberry grades); reported shortly in Model Specific Analysis
         
        FILTER_PRE = datasets[i]
       

        # True: test set includes GT fill-in samples (full GT test set size).
        # False: test set contains only model-detected samples in the GT test set.
        # Leave this to True, it makes the most sense for comparison with GT masks.
        INCLUDE_GT_FILL_INS = True

        FILTER_PRE = FILTER_PRE.upper()
        # Get the last element in filter_pre
        last_element = FILTER_PRE.split('_')[-1]

        # This was when I tested with variable img input sizes (i.e. different raspberries have different sizes and then I had input images of varying sizes), but this turned out not to work well, so can be forgotten
        if last_element != 'variable':
            pass_og_bool = True
        else:
            pass_og_bool = False


        # Post-Filtering Techniques, i.e. after AD Heatmap is created, remove heatmap where there is a hole in the raspberry/raspberry is too dark
        # Post-Filtering might lead to better results, but it is a too "simple" fix, therefore I didn't report it in the thesis and this can also be just forgotten, leave it as NONE
        FILTER_POST = 'NONE' # HOLE_DARKNESS_40_40 best results,  HOLE_DARKNESS_k_j : filter out holes and dark areas based on depth & darkness of raspberry ; k refers to threshold for depth and j to threshold for darkness ; see utilities/filters for more details ; DARKNESS_k : filter out dark areas based on darkness of raspberry, k refers to threshold for darkness ; see utilities/filters for more details ; DRUPELETS for removing specular highlights ; NONE if no post filtering
        PROTRUSION_DAMPING_GAMMA = 1 if 'HOLE_DARKNESS' in FILTER_POST else 0

        # Which Scoring to use for the final anomaly score
        SCORING = 'STRUCTCORE' # MAXMEAN_k , where k refers to the factor for the max (i.e. k * max_score + (1-k) * mean_score) ; STRUCTCORE

        TOP_K_RATIO_STRUCTCORE = 0.01

        dataset_path = Path('../../nvme1/thesis/dataset_single_objects/') 
        target_path = FILTER_PRE.lower()


        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print(device)
       
       
       # Set backbone+ its layers'
        backbone = "wide_resnet50_2" 
      #  backbone = "resnet18"
      #  backbone = "dinov2_vitb14"
      #  backbone = 'mobilenet_v2'
    # backbone = "dinov3_vitb16"
       # ad_layers = ["features.13", "features.17"] 
      #  ad_layers = ["features.6", "features.13"] 
      #  ad_layers = ["features.3", "features.6", "features.13"]
    # ad_layers = ["layer4"]
    #  ad_layers = ["features.10"] # SINBAD tests
     #   ad_layers = ["layer2", "layer3"]
       # ad_layers = ["layer3", "layer4"]
      #  ad_layers = ["layer2", "layer3"]
      #  ad_layers = ["layer2", "layer3", "layer4"]
        ad_layers = ["layer1", "layer2", "layer3"]
        
      #  ad_layers = [5] 

    ##  if CLS_TOKEN_VIZ_BOOL:
    #      ad_layers.append(11) # extracting also CLS token for visualizations
        end = ".pt" if MODEL_MODE != 'sinbad' else ".pkl"
        aug_parts = []
        if SYN_AUG_BOOL:
            aug_parts.append(SYN_AUG_MODE)
        if RANDOMIZE_ROTATION_BOOL:
            aug_parts.append("rand_rot")
        if HOLE_AUG_BOOL:
            aug_parts.append(f"hole_{HOLE_AUG_MODE}")
        aug_str = "_".join(aug_parts) if aug_parts else "no_aug"
        


        custom_weights_path = None

        # This part can be uncommented for pre-training a backbone based on self-supervised tuning (see thesis) 
     #   ad_layers_pretrained = ["layer2","layer3"]
      #  custom_weights_path = f"../../nvme1/thesis/pretrained_models/dataset_{FILTER_PRE}_{backbone}_cutout_{'_'.join([str(layer) for layer in ad_layers_pretrained])}.pt"

        if custom_weights_path is not None:
            save_path = f"../../nvme1/thesis/pretrained_models/{MODEL_MODE}_{backbone}_{'_'.join([str(layer) for layer in ad_layers])}_data_{FILTER_PRE}__custom_pretrained_{'_'.join([str(layer) for layer in ad_layers_pretrained])}{end}_{aug_str}" # _FULL
        else:
            save_path = f"../../nvme1/thesis/pretrained_models/{MODEL_MODE}_{backbone}_{'_'.join([str(layer) for layer in ad_layers])}_data_{FILTER_PRE}{end}_{aug_str}_" # _FULL 

 

        # unfreeze_from : mobilenet/wrn-50-2, unfreeze_last_n_blocks : vit, for vit we unfreeze the last n blocks, for mobilenet/wrn we unfreeze from the layer specified by unfreeze_from (e.g. 4 means unfreeze from layer4 and then also layer4 itself)
       # pretrain_backbone_cutout(dataset_path, device, backbone, save_path=custom_weights_path,
       #                         unfreeze_from=2, epochs=100, lr=1e-3, batch_size=32,
       #                         n_holes=1, hole_size_range=(32, 64), target_path=target_path, unfreeze_last_n_blocks=1, lora_rank = 4, lora_alpha = 4)


    
       # train_model(dataset_path, backbone, ad_layers, save_path, device, mode = MODEL_MODE, target_path = target_path, pass_og_bool = pass_og_bool, scoring_mode = SCORING, filter_post = FILTER_POST, mask_border_filter_thickness = 0, custom_weights_path = custom_weights_path, synthetic_augmentation_bool = SYN_AUG_BOOL, synthetic_augmentation_mode = SYN_AUG_MODE, randomize_rotation_bool = RANDOMIZE_ROTATION_BOOL, cls_token_viz_bool = CLS_TOKEN_VIZ_BOOL, hole_augmentation_bool = HOLE_AUG_BOOL, hole_augmentation_mode = HOLE_AUG_MODE, include_gt_fill_ins = INCLUDE_GT_FILL_INS, epochs = EPOCHS, batch_size_train = BATCH_SIZE_TRAIN, AD_only_on_mask = AD_ONLY_ON_MASK, mask_dilation_radius = MASK_DILATION_RADIUS)

        # Check if visual test path exists and clear it out if it already exists
        if custom_weights_path is not None:
            visual_test_path = f"../../nvme1/thesis/visual_test/{MODEL_MODE}_{backbone}_{'_'.join([str(layer) for layer in ad_layers])}_pretrained_data_{FILTER_PRE}_{SCORING}_test_set_{FILTER_POST}_{aug_str}/"
        else:
            visual_test_path = f"../../nvme1/thesis/visual_test/{MODEL_MODE}_{backbone}_{'_'.join([str(layer) for layer in ad_layers])}_data_{FILTER_PRE}_{SCORING}_test_set_{FILTER_POST}_{aug_str}/" # NOTE : disk normally
       # visual_test_path = f"../../nvme1/thesis/visual_test/{MODEL_MODE}_{backbone}_data_{FILTER_PRE}_{SCORING}_test_set_{FILTER_POST}_{aug_str}/" # NOTE : disk normally
        visual_test_dir = Path(visual_test_path)
        if visual_test_dir.exists():
            shutil.rmtree(visual_test_dir)
        visual_test_dir.mkdir(parents=True, exist_ok=True)


        
        test_model(dataset_path, backbone, ad_layers, save_path, device, mode = MODEL_MODE,
        target_path = target_path, visual_test_path = visual_test_path, scoring_mode = SCORING,
        filter_post = FILTER_POST, mask_border_filter_thickness = 0, pass_og_bool = pass_og_bool,
        custom_weights_path = custom_weights_path, cls_token_viz_bool = CLS_TOKEN_VIZ_BOOL, top_k_ratio = TOP_K_RATIO_STRUCTCORE,
        protrusion_damping_radius = 0, protrusion_damping_gamma = PROTRUSION_DAMPING_GAMMA, include_gt_fill_ins = INCLUDE_GT_FILL_INS, AD_only_on_mask = AD_ONLY_ON_MASK, mask_dilation_radius = MASK_DILATION_RADIUS)
        detailed_eval(visual_test_path)



if __name__ == "__main__":
    main()