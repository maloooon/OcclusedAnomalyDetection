from typing import List, Tuple
import torch
import torch.nn.functional as F
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from ...utilities.filters import filter_holes_batched
from time import time

SEED = 32
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class STFPM(torch.nn.Module):

    DEFAULT_PARAMETERS = {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.2,
        "weight_decay": 1e-4,
        "momentum": 0.9,
    }

    def __init__ (
        self,
        teacher:CustomFeatureExtractor,
        student:CustomFeatureExtractor,
        struct_core_instance = None,
        scoring_mode = 'MAXMEAN_1',
        filter_post = 'NONE',
        mask_border_filter_thickness = 0
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.struct_core_instance = struct_core_instance
        self.scoring_mode = scoring_mode
        self.filter_post = filter_post
        self.mask_border_filter_thickness = mask_border_filter_thickness

    def forward(self, batch: torch.Tensor):



        if (isinstance(batch, tuple)):
            if len(batch) == 5:
                batch, mask, batch_og, mask_og, depth_og = batch
            if len(batch) == 2:
                batch, mask = batch
               

        
    
            

        if self.training:
            teacher_features, student_features = None, None


        
            with torch.no_grad():
                teacher_features = self.teacher(batch)
            student_features = self.student(batch)
           
          

            
            if "dino" in self.teacher.model_name:
                # In case we use dinov2, also returns the cls tokens currently
                teacher_features, teacher_cls_tokens = teacher_features
                student_features, student_cls_tokens = student_features
            return teacher_features, student_features

        else:
            student_features = self.student(batch)
            teacher_features = self.teacher(batch)


            if "dino" in self.teacher.model_name:
                # In case we use dinov2, also returns the cls tokens currently
                teacher_features, teacher_cls_tokens = teacher_features
                student_features, student_cls_tokens = student_features


            return self.post_process(
                teacher_features, student_features, batch.shape[2:], batch, batch_og, mask_og, depth_og, mask, border_thickness = self.mask_border_filter_thickness
            )

    def __call__(self, batch: torch.Tensor):
        return self.forward(batch)

    def train(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.train()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.eval()
        return super().eval(*args, **kwargs)


    def load_model(self, path):

        """
        Load the Model

        Parameters:
        ----------
            path: str
                The path to the saved model checkpoint
        """

        state_dict = torch.load(path)



    def save_anomaly_map(self, dirpath, anomaly_map, pred_score, filepath, x_type, mask):
        """
        Args:
            dirpath     (str)       : Output directory path.
            anomaly_map (np.ndarray): Anomaly map with the same size as the input image.
            filepath    (str)       : Path of the input image.
            x_type      (str)       : Anomaly type (e.g. "good", "crack", etc).
            contour     (float)     : Threshold of contour, or None.
        """
        def min_max_norm(image):
            a_min, a_max = image.min(), image.max()
            return (image - a_min) / (a_max - a_min)

        def cvt2heatmap(gray):
            return cv.applyColorMap(np.uint8(gray), cv.COLORMAP_JET)

        # Get the image file name.
        filename = os.path.basename(filepath)

        # Load the image file and resize.
        original_image = cv.imread(filepath)
        original_image = cv.resize(original_image, anomaly_map.shape[:2])

        # Normalize anomaly map for easier visualization.
        anomaly_map_norm = cvt2heatmap(255 * min_max_norm(anomaly_map))

        # Overlay the anomaly map to the origimal image.
        output_image = anomaly_map_norm.astype(np.uint8)#(anomaly_map_norm / 2 + original_image / 2).astype(np.uint8) # NOTE : changed

        # Create a figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))

        #convert the images to RGB
        original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGB)

        # Display the input image
        axes[0].imshow(original_image)
        axes[0].set_title(f'Original Image {x_type}')
        axes[0].axis('off')

        # Display the mask image
        axes[1].imshow(mask.squeeze(), cmap ='gray')
        axes[1].set_title(f'Mask')
        axes[1].axis('off')

        # Display the final image
        axes[2].imshow(output_image)
        axes[2].set_title(f'Heatmap {pred_score}')
        axes[2].axis('off')

        # Show the plot
        plt.savefig(str(dirpath + f"/{x_type}_{filename}.jpg"))




    def post_process(self, t_feat, s_feat, output_shape, batch, batch_og, mask_og, depth_og, mask, border_thickness = 5) -> torch.Tensor:

        """
        This method actually produces the anomaly maps for evalution purposes

        NOTE :
            added batch and depth for post-processing filtering

        Args:
            - t_feat: teacher features maps
            - s_feat: student features maps
            - batch_og: original batch
            - mask_og: original mask
            - depth_og: original depth

        Returns:
            - anomaly maps

        """

        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=output_shape, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        # TODO : consider that mask, batch_og etc. is not given in trainloader, so StructCore cant work with it as of now ...
        # TODO : possibly need to add

        # --- Apply raspberry mask with contour-based border exclusion ---
        
        if mask is not None:
            # Ensure (B, 1, H, W)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)


            # Build the effective mask per sample: mask minus contour border
            effective_mask = torch.zeros_like(mask, dtype=torch.float32)

            for i in range(mask.shape[0]):
                sample_mask = mask[i, 0].cpu().numpy().astype(np.uint8)  # (H, W)

                if border_thickness > 0:
                    # Find external contours of the raspberry region
                    contours, _ = cv.findContours(
                        sample_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                    )
                    # Draw contour band — thickness draws half inward, half outward,
                    # but since outside is already 0 in the mask, the AND below
                    # ensures only the inward portion matters.
                    contour_band = np.zeros_like(sample_mask)
                    cv.drawContours(
                        contour_band, contours, -1, 255, thickness=border_thickness
                    )
                    # Subtract contour band from the mask
                    inner_mask = sample_mask.copy()
                    inner_mask[contour_band == 255] = 0
                else:
                    inner_mask = sample_mask

                effective_mask[i, 0] = torch.from_numpy(
                    inner_mask.astype(np.float32)
                ).to(device)

            # Binarize (in case mask had values other than 0/1)
            effective_mask = (effective_mask > 0).float()

            # Zero out everything outside the effective mask
            score_maps = score_maps * effective_mask


        # Filter holes before computing anomaly scores
        if 'HOLE_DARKNESS' in self.filter_post:
            thresh_depth, thresh_dark = self.filter_post.split('_')[2:4]
            score_maps = filter_holes_batched(score_maps, batch, batch_og, mask_og, depth_og, depth_threshold_percentile = int(thresh_depth), brightness_threshold_percentile = int(thresh_dark))

        # Compute per-image anomaly score only over valid regions
        flat = score_maps.view(score_maps.size(0), -1)
        anomaly_scores = torch.max(flat, dim=1)[0]


        if self.struct_core_instance is not None and self.scoring_mode == 'STRUCTCORE':
            anomaly_scores = self.struct_core_instance.score(score_maps, anomaly_scores)

        else:
            k = float(self.scoring_mode.split('_')[-1])
            max_scores = anomaly_scores #torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
            mean_scores = torch.mean(flat, dim=1)#torch.mean(score_maps.view(score_maps.size(0), -1), dim=1)
            anomaly_scores = k * max_scores + (1 - k) * mean_scores




            # OG
        # anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]



       


        


        return score_maps, anomaly_scores