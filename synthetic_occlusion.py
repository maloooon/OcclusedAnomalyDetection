import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SyntheticOcclusion:
    def __init__(self, data_path):
        """
        Args : 
        data_path (str) : Path to the dataset of single raspberry images ; right now of a single sample (i.e. of one bonnet)
        
        """
        # Open the .npz file
        data_dict = np.load(data_path, allow_pickle=True)
        self.masks = data_dict["masks"] # Array of segmentation masks of single raspberries of the currently loaded sample
        self.images = data_dict["images"] # Array of single raspberry images of the currently loaded sample



    def single_raspberry_occlusion(self, abs_size_threshold=100, 
                                wanted_size_range=None,
                                realism_bool=False, 
                                randomize_scale_bool=(False, 0.2, 2.0), 
                                randomize_rotation_bool=(False, -30, 30)):
        """
        Occlusion of single raspberries (Idea 2 in presentation).
        Based on the idea of Copy & Paste as in Ghiasi et al., 
        "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation".

        Args :
            abs_size_threshold (int) : Minimum absolute size (in pixels) of the new segmentation mask.
                                    Only used if wanted_size_range is None.
            wanted_size_range (tuple) : If provided, (min_pixels, max_pixels) range for the remaining
                                        visible area. Overrides abs_size_threshold if set.
            realism_bool (bool) : Whether to apply blending to make the occlusion look more natural.
            randomize_scale_bool (tuple) : If first element is True, randomly scale the source 
                                        raspberry by a factor sampled uniformly from the range
                                        defined by the second and third elements of the tuple.
            randomize_rotation_bool (tuple) : If first element is True, randomly rotate the source
                                            raspberry by an angle sampled uniformly from the range
                                            defined by the second and third elements of the tuple.

        Workflow : 
            1. Select two random images of single raspberries from the dataset.
            2. Paste the source raspberry region onto the target image at a random location within
            the locality of the raspberry in the target image (Targeted Pasting).
            3. Optionally apply blending to make the occlusion look more natural.
        """

        # 1. Select two random images and their respective masks
        chosen_idx = np.random.choice(len(self.images), size=2, replace=True)

        target_img = self.images[chosen_idx[0]].copy()
        source_img = self.images[chosen_idx[1]].copy()

        target_mask = self.masks[chosen_idx[0]].copy()
        source_mask = self.masks[chosen_idx[1]].copy()

        # 2. Paste the source raspberry region onto the target image
        new_target_img, new_target_mask = self._paste(
            target_mask, source_mask, 
            abs_size_threshold, 
            wanted_size_range,
            source_img, target_img,
            randomize_scale_bool=randomize_scale_bool,
            randomize_rotation_bool=randomize_rotation_bool,
            visualize_bool=True
        )

        # 3. Optionally apply blending for realism
    # if realism_bool:
    #     new_target_img = self._apply_blending(new_target_img, source_img, new_target_mask)

        return new_target_img, new_target_mask


    def _paste(self, target_mask, source_mask, abs_size_threshold, wanted_size_range,
            source_img, target_img, 
            randomize_scale_bool=(False, 0.2, 2.0), 
            randomize_rotation_bool=(False, -30, 30),
            visualize_bool=False):
        """
        Paste the source raspberry onto the target raspberry at a random location within
        the locality of the target raspberry.

        Args :
            target_mask (np.array) : Binary mask of the target raspberry.
            source_mask (np.array) : Binary mask of the source raspberry.
            abs_size_threshold (int) : Minimum absolute size (in pixels) of the new segmentation mask.
            wanted_size_range (tuple) : If provided, (min_pixels, max_pixels) for remaining visible area.
            source_img (np.array) : Source raspberry image.
            target_img (np.array) : Target raspberry image.
            randomize_scale_bool (tuple) : (bool, min_scale, max_scale)
            randomize_rotation_bool (tuple) : (bool, min_angle, max_angle)
            visualize_bool (bool) : Whether to visualize the result.

        Returns :
            new_target_img (np.array) : Updated target image with source pasted.
            new_target_mask (np.array) : Updated binary mask after occlusion.
        """

        source_img = np.asarray(source_img, dtype=np.uint8)
        target_img = np.asarray(target_img, dtype=np.uint8)
        source_mask = np.asarray(source_mask, dtype=bool)
        target_mask = np.asarray(target_mask, dtype=bool)
        
        # Apply transformations to source if requested
        transformed_source_img = source_img.copy()
        transformed_source_mask = source_mask.copy()
        
        # Scale transformation
        if randomize_scale_bool[0]:
            scale_factor = np.random.uniform(randomize_scale_bool[1], randomize_scale_bool[2])
            h, w = source_mask.shape
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            transformed_source_img = cv2.resize(transformed_source_img, (new_w, new_h), 
                                            interpolation=cv2.INTER_LINEAR)
            transformed_source_mask = cv2.resize(transformed_source_mask.astype(np.uint8), 
                                                (new_w, new_h), 
                                                interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Rotation transformation
        if randomize_rotation_bool[0]:
            angle = np.random.uniform(randomize_rotation_bool[1], randomize_rotation_bool[2])
            h, w = transformed_source_mask.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            transformed_source_img = cv2.warpAffine(transformed_source_img, M, (w, h), 
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0)
            transformed_source_mask = cv2.warpAffine(transformed_source_mask.astype(np.uint8), M, (w, h),
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0).astype(bool)
        
        # Get coordinates of True pixels in both masks
        target_coords = np.argwhere(target_mask)
        source_coords = np.argwhere(transformed_source_mask)
        
    
        # Determine acceptance criteria
        if wanted_size_range is not None:
            min_size, max_size = wanted_size_range
            def is_valid(remaining_size):
                return min_size <= remaining_size <= max_size
        else:
            def is_valid(remaining_size):
                return remaining_size >= abs_size_threshold
        
        # Try up to 50 paste attempts
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # 1. Select random pixel from target mask
            target_anchor_idx = np.random.randint(len(target_coords))
            target_anchor_y, target_anchor_x = target_coords[target_anchor_idx]
            
            # 2. Select random pixel from source mask
            source_anchor_idx = np.random.randint(len(source_coords))
            source_anchor_y, source_anchor_x = source_coords[source_anchor_idx]
            
            # 3. Calculate offset to align these two pixels
            offset_y = target_anchor_y - source_anchor_y
            offset_x = target_anchor_x - source_anchor_x
            
            # 4. Create the pasted mask and image
            new_target_mask = target_mask.copy()
            new_target_img = target_img.copy()
            
            h_target, w_target = target_mask.shape
            h_source, w_source = transformed_source_mask.shape
            
            # Calculate valid paste region
            src_y_start = max(0, -offset_y)
            src_x_start = max(0, -offset_x)
            src_y_end = min(h_source, h_target - offset_y)
            src_x_end = min(w_source, w_target - offset_x)
            
            dst_y_start = max(0, offset_y)
            dst_x_start = max(0, offset_x)
            dst_y_end = dst_y_start + (src_y_end - src_y_start)
            dst_x_end = dst_x_start + (src_x_end - src_x_start)
            
            # Check if there's valid overlap
            if src_y_end <= src_y_start or src_x_end <= src_x_start:
                continue
            
            # Extract the regions
            source_region_mask = transformed_source_mask[src_y_start:src_y_end, src_x_start:src_x_end].astype(bool)
            source_region_img = transformed_source_img[src_y_start:src_y_end, src_x_start:src_x_end]
            
            # Paste source onto target where source mask is True
            new_target_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask] = \
                source_region_img[source_region_mask]
            
            # Update mask: Remove occluded parts (set to False where source overlaps)
            new_target_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask] = False
            
            # Check if remaining target mask meets criteria
            remaining_size = np.sum(new_target_mask)
            
            if is_valid(remaining_size):
                # Valid result found - return immediately (first success)
                if visualize_bool:
                    self._visualize_paste(target_img, target_mask, 
                                        source_img, source_mask,  
                                        transformed_source_img, transformed_source_mask,  
                                        new_target_img, new_target_mask, 
                                        offset_y, offset_x)
                return new_target_img, new_target_mask
        
        # If no valid paste found after max_attempts, return original
        print(f"Warning: No valid paste location found after {max_attempts} attempts. "
            f"Returning original image.")
        return target_img, target_mask

    def _visualize_paste(self, target_img, target_mask, source_img, source_mask,
                        transformed_source_img, transformed_source_mask,
                        result_img, result_mask, offset_y, offset_x):
        """
        Visualize the paste operation showing original images, masks, and results.
        
        Args:
            target_img: Original target raspberry image
            target_mask: Original target raspberry mask
            source_img: Source raspberry image (before transformation)
            source_mask: Source raspberry mask (before transformation)
            transformed_source_img: Source image after scale/rotation
            transformed_source_mask: Source mask after scale/rotation
            result_img: Resulting image after paste
            result_mask: Resulting mask after occlusion
            offset_y: Vertical offset used for pasting
            offset_x: Horizontal offset used for pasting
        """

        target_img = np.asarray(target_img, dtype=np.uint8)
        source_img = np.asarray(source_img, dtype=np.uint8)
        transformed_source_img = np.asarray(transformed_source_img, dtype=np.uint8)
        result_img = np.asarray(result_img, dtype=np.uint8)
        target_mask = np.asarray(target_mask, dtype=bool)
        source_mask = np.asarray(source_mask, dtype=bool)
        transformed_source_mask = np.asarray(transformed_source_mask, dtype=bool)
        result_mask = np.asarray(result_mask, dtype=bool)
        
        
        # Create the final occluded raspberry image (only showing the remaining visible part)
        occluded_raspberry_img = result_img.copy()
        occluded_raspberry_img[~result_mask] = 0  # Black out everything except remaining mask
        
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        
        # Row 1: Images with mask overlays
        axes[0, 0].imshow(target_img)
        axes[0, 0].imshow(target_mask, alpha=0.3, cmap='Reds')
        axes[0, 0].set_title(f'Target (original)\n{np.sum(target_mask)} pixels')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(source_img)
        axes[0, 1].imshow(source_mask, alpha=0.3, cmap='Blues')
        axes[0, 1].set_title(f'Source (original)\n{np.sum(source_mask)} pixels')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(transformed_source_img)
        axes[0, 2].imshow(transformed_source_mask, alpha=0.3, cmap='Purples')
        axes[0, 2].set_title(f'Source (transformed)\n{np.sum(transformed_source_mask)} pixels')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(result_img)
        axes[0, 3].imshow(result_mask, alpha=0.3, cmap='Greens')
        axes[0, 3].set_title(f'Result (composite)\n{np.sum(result_mask)} pixels remaining')
        axes[0, 3].axis('off')
        
        axes[0, 4].imshow(occluded_raspberry_img)
        axes[0, 4].set_title(f'Occluded raspberry (final)\n{np.sum(result_mask)} visible pixels')
        axes[0, 4].axis('off')
        
        # Row 2: Masks and analysis
        axes[1, 0].imshow(target_mask, cmap='gray')
        axes[1, 0].set_title('Target mask')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(source_mask, cmap='gray')
        axes[1, 1].set_title('Source mask (original)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(transformed_source_mask, cmap='gray')
        axes[1, 2].set_title(f'Source mask (transformed)\nOffset: ({offset_y}, {offset_x})')
        axes[1, 2].axis('off')
        
        # Show the occlusion region (what was removed)
        occlusion_region = target_mask & ~result_mask
        axes[1, 3].imshow(result_mask, cmap='Greens', alpha=0.5)
        axes[1, 3].imshow(occlusion_region, cmap='Reds', alpha=0.5)
        occlusion_percentage = (np.sum(occlusion_region) / np.sum(target_mask)) * 100 if np.sum(target_mask) > 0 else 0
        axes[1, 3].set_title(f'Result mask (green) + Occluded area (red)\n{occlusion_percentage:.1f}% occluded')
        axes[1, 3].axis('off')
        
        # Show just the result mask
        axes[1, 4].imshow(result_mask, cmap='gray')
        axes[1, 4].set_title(f'Final mask (ground truth)\n{np.sum(result_mask)} pixels')
        axes[1, 4].axis('off')
        
        plt.tight_layout()
        plt.show()

        
    


def main():
    synthetic_occlusion = SyntheticOcclusion(data_path='dataset_single_objects/GT/processed/img001/processed_img001_data.npz')
    new_img, new_mask = synthetic_occlusion.single_raspberry_occlusion(
        abs_size_threshold=50,
        #wanted_size_range=(300, 800),
        realism_bool=False,
        randomize_scale_bool=(True, 0.2, 2.0),
        randomize_rotation_bool=(False, -180,180)
    )
    

if __name__ == "__main__":
    main()
