
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np

from moviad.common.common_utils import obsolete
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.evaluator import Evaluator

from torchvision.transforms.functional import InterpolationMode

from synthetic_occlusion import SyntheticOcclusion

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


# TODO : in the Dataset for Raspberries, add as possible transformation the synthetic occlusion ? Such that with each epoch during training, we add different occlusion patterns
# TODO : based on the modes we need to make it fair, i.e. easy would be to just on the currently selected raspberries, but maybe we can select a batch of raspberries and then do on them
# TODO : the multi-raspberry modes. Need to figure out how to do it also with the fact that we load in anomalous/normal first 

# TODO : need to set the whole mask of anomaly raspberries to the mask size of the anomaly raspberry (i.e. we have no exact mask where anomaly is)
class SingleRaspberryDataset(Dataset):
    def __init__(self, dataset_path: str, split = None, synthetic_augmentation = False):
        self.dataset_path = dataset_path
        self.split = split
        self.synthetic_augmentation = synthetic_augmentation
        if self.synthetic_augmentation:
            self.synthetic_occlusion = SyntheticOcclusion(base_path= Path(self.dataset_path), sample_folders = ['anomalous','normal'])

        self.transform_img = transforms.Compose([
        transforms.Resize((224,224), antialias = True, interpolation=InterpolationMode.NEAREST), # PatchCore specific (i.e. taken from paper)
       # transforms.CenterCrop((224,224)), # PatchCore specific (i.e. taken from paper)
        transforms.ToTensor(),  # Converts to [C, H, W] tensor in [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Since also used in MVTec implementation
                            std=[0.229, 0.224, 0.225]),
    ])

        self.transform_mask = transforms.Compose([
        transforms.Resize((224,224), antialias = True, interpolation=InterpolationMode.NEAREST), # PatchCore specific (i.e. taken from paper)
      #  transforms.CenterCrop((224,224)), # PatchCore specific (i.e. taken from paper)
        transforms.ToTensor(),  # Converts to [C, H, W] tensor in [0, 1]
    ])

    
        normal_path = Path(self.dataset_path) / 'normal' / 'normal_samples.pkl' 
        anomalous_path = Path(self.dataset_path) / 'anomalous' / 'anomalous_samples.pkl'

        self.train_indices_non_anom = pickle.load(open(Path(self.dataset_path) / 'splits' / 'train_normal_indices.pkl', 'rb'))
        self.test_indices_non_anom = pickle.load(open(Path(self.dataset_path) / 'splits' / 'test_normal_indices.pkl', 'rb'))

        with open(normal_path, 'rb') as f:
            normal_data = pickle.load(f)
        with open(anomalous_path, 'rb') as f:
            anomalous_data = pickle.load(f)

        n_anomalous = len(anomalous_data)


        normal_test = [normal_data[i] for i in self.test_indices_non_anom]
        normal_train = [normal_data[i] for i in self.train_indices_non_anom]

        if self.split == 'train':
            # Training: all normal samples except the ones reserved for testing
            self.data = normal_train
        else:
            # Testing: all anomalous + equal number of random normal samples
            self.data = anomalous_data + normal_test

        self.img_paths = [item['img_path'] for item in self.data]
        self.grades = [item['grade'] for item in self.data]
        self.masks = [item['mask'] for item in self.data]
        self.img_arrays = [item['image'] for item in self.data]
        

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
    
        
        if self.synthetic_augmentation and random.random() < 0.5:  # Apply synthetic occlusion with 50% chance

            wanted_size_range= (random.randint(1, 5) / 10), (random.randint(6, 9) / 10) # Random range with possible values 0,0.1,0.2...1

            # We can use both anomalous and normal for the current occlusion ideas as long as we choose the initial
            # raspberry (target raspberry) of the class (anomalous or normal) that we want to create
            new_img , new_mask, grade = self.synthetic_occlusion.multi_raspberry_occlusion(
                                                                wanted_size_range= wanted_size_range,
                                                                randomize_scale_bool=(False, 0.5, 0.9),
                                                                randomize_rotation_bool=(False, -180,180),
                                                                visualize_bool=False,
                                                                reassign_source_target_bool = False,
                                                                sampling_mode= ('N_largest', 50),
                                                                k = 2,
                                                                chosen_initial_raspberry = (self.img_arrays[idx], self.masks[idx], self.grades[idx]))

            # In case no occlusion pattern was found
            if new_img is None:
                img_file = self.img_paths[idx]
                # Load image as PIL Image
                img = Image.open(img_file).convert("RGB")

            else:
                new_img = np.asarray(new_img, dtype=np.uint8)
                new_mask = np.asarray(new_mask, dtype=bool)
                # Center the occluded img & mask
                new_img, new_mask = _center_object(new_img, new_mask)
                # Save this new image to a synthetic folder, since we do not want to change the original images
                synthetic_img_path = Path(self.dataset_path) / 'synthetic' / f'synthetic_{Path(self.img_paths[idx]).stem}.png'
                # Turn into PIL image
                img = Image.fromarray(new_img, 'RGB')
                img.save(synthetic_img_path)
                img = np.asarray(img, dtype=np.uint8)

                # Replace 
                self.img_arrays[idx] = img
                self.masks[idx] = new_mask
                self.img_paths[idx] = synthetic_img_path 
            

        # Get the image path and load the image
        img_file = self.img_paths[idx]
        # Load image as PIL Image
        img = Image.open(img_file).convert("RGB")
        # Get the mask
        mask = self.masks[idx]
        # Load mask as PIL Image
        mask = Image.fromarray(mask.astype(np.uint8) * 255) 
        

        
        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        


        if self.split == 'test':
            img_path = self.img_paths[idx] 
            grade = self.grades[idx]
            if grade > 3:
                error_mask = mask
               # mask = img > 0 # NOTE : We do not have exact masks of the anomalous regions of the raspberries, therefore we set the whole mask of anomaly raspberries to the mask size of the anomaly raspberry (i.e. we have no exact mask where anomaly is)
                needed_grade = 1 # in evaluation grades need to be 0,1 or -1,1 for the two classes
            else:
                error_mask = torch.zeros(img.shape[1], img.shape[2]) # Create a mask of the same size as the image with all 0s (i.e. no anomaly)
                needed_grade = 0 # in evaluation grades need to be 0,1 or -1,1 for the two classes
            # Add channel dimension to mask (i.e shape [H, W] -> [1, H, W]) if needed
            if len(error_mask.shape) == 2:
                error_mask = error_mask.unsqueeze(0)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            return img, needed_grade, error_mask.int(), img_path, mask

        else:
            return img
            
def train_patchcore(dataset_path : str, backbone : str, ad_layers : list, save_path : str, device : torch.device, max_dataset_size : int = None):

    # initialize the feature extractor
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)

    # Only normal samples for training
    train_dataset = SingleRaspberryDataset(dataset_path, split = 'train', synthetic_augmentation = False)


    if max_dataset_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(max_dataset_size))
    print(f"Length train dataset: {len(train_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Only anomalous samples for testing
    test_dataset = SingleRaspberryDataset(dataset_path, split = 'test', synthetic_augmentation = False)

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)


    # Define the model
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor, k = 10000)
    patchcore.to(device)
    patchcore.train()

    trainer = TrainerPatchCore(patchcore, train_dataloader, test_dataloader, device)
    trainer.train()

    # save the model
    if save_path:
        torch.save(patchcore.state_dict(), save_path)

    # force garbage collector in case
    del patchcore
    del test_dataset
    del train_dataset
    del train_dataloader
    del test_dataloader
    torch.cuda.empty_cache()
    gc.collect()

def test_patchcore(dataset_path : str, backbone : str, ad_layers : list, model_checkpoint_path : str, device : torch.device, max_dataset_size : int = None, visual_test_path: str = None):

    # Only anomalous samples for testing
    test_dataset = SingleRaspberryDataset(dataset_path, split = 'test')

    if max_dataset_size is not None:
        test_dataset = torch.utils.data.Subset(test_dataset, range(max_dataset_size))
    print(f"Length test dataset: {len(test_dataset)}")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    # load the model
    feature_extractor = CustomFeatureExtractor(backbone, ad_layers, device, True, False, None)
    patchcore = PatchCore(device, input_size=(224, 224), feature_extractor=feature_extractor, k = 10000)
    patchcore.load_model(model_checkpoint_path)
    patchcore.to(device)
    patchcore.eval()

    evaluator = Evaluator(test_dataloader, device)
    metrics = evaluator.evaluate(patchcore)

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


    opt_threshold = 2.5206628



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
        for images, labels, masks, paths, full_mask in tqdm(iter(test_dataloader)):
            anomaly_maps, pred_scores, _ , _ = patchcore((images.to(device), full_mask.to(device)))

            anomaly_maps = torch.permute(anomaly_maps, (0, 2, 3, 1))

            

            for i in range(anomaly_maps.shape[0]):
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

                patchcore.save_anomaly_map(visual_test_path, anomaly_maps[i].cpu().numpy(), pred_scores[i], paths[i],
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

    

    
def overlay_mask_on_image(image, mask):
    green_mask = np.zeros_like(image)
    green_mask[mask > 0] = [0, 255, 0]  # Green color

    # Blend the original image with the red mask
    blended = cv2.addWeighted(image, 0.8, green_mask, 0.2, 0)
    cv2.imwrite("original_normal_image.png", image)
    cv2.imwrite("overlayed_image.png", blended)



def main():



    dataset_path = Path('../../disk/dataset_single_objects/GT/processed/')

    # Create a folder 'synthetic' in the current directory and clear it out if it already exists
    synthetic_dir = dataset_path / 'synthetic'
    if synthetic_dir.exists():
        shutil.rmtree(synthetic_dir)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    # TODO: how to achieve top quality for input images ?

     


    # Train the model
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    backbone = "mobilenet_v2" # "wide_resnet50_2" #"mobilenet_v2" #   #
    ad_layers = ["features.4", "features.7", "features.10"] #["layer2", "layer3"] #   # 
    save_path = "../../disk/pretrained_models/patch_core_mobilenet_v2.pt"


    #from torchvision.models import wide_resnet50_2
    #from torchvision.models.feature_extraction import get_graph_node_names

    #model = wide_resnet50_2(pretrained=True)
    #train_nodes, eval_nodes = get_graph_node_names(model)

    #for node in eval_nodes:
    #    print(node)

    # Sanity check on Dataset and visualization of masks
  #  dataset = SingleRaspberryDataset(dataset_path, split = 'test', synthetic_augmentation = False)

  #  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

     #Get an example batch
   # for images, labels, masks, paths, full_mask in dataloader:
        # Visualize the first image and its mask
      #  img = images[0].permute(1, 2, 0).numpy()  # Convert from [C, H, W] to [H, W, C]
      #  img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # back to [0,1]
       # img = (img * 255).clip(0, 255).astype(np.uint8)
       # mask = masks[0].squeeze(0).numpy() > 0  # Convert from [1, H, W] to [H, W] and binarize

        # Overlay the mask on the image
       # overlay_mask_on_image(img, mask)

     #   break






    #train_patchcore(dataset_path, backbone, ad_layers, save_path, device)

    test_patchcore(dataset_path, backbone, ad_layers, save_path, device, visual_test_path = "../../disk/visual_test/mobilenet_v2_3_7_10/")
    detailed_eval("../../disk/visual_test/mobilenet_v2_3_7_10/")








if __name__ == "__main__":
    main()



