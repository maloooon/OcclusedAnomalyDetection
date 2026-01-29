import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

class SyntheticOcclusion:
    def __init__(self, base_path, sample_folders=None):
        """
        Args:
            base_path: Path to dataset_single_objects/GT/processed/
            sample_folders: List of folder names (e.g., ['img001', 'img002']) 
                          If None, loads all folders
        """
        base_path = Path(base_path)
        
        # Get all sample folders if not specified
        if sample_folders is None:
            sample_folders = sorted([d.name for d in base_path.iterdir() if d.is_dir()])
        
        all_masks = []
        all_images = []
        
        for folder in sample_folders:
            npz_path = base_path / folder / f"processed_{folder}_data.npz"
            
            if not npz_path.exists():
                print(f"Warning: {npz_path} not found, skipping")
                continue
                
            data = np.load(npz_path, allow_pickle=True)
            all_masks.append(data['masks'])
            all_images.append(data['images'])
        
        # Combine all samples
        self.masks = np.concatenate(all_masks, axis=0)
        self.images = np.concatenate(all_images, axis=0)
        
        print(f"Loaded {len(sample_folders)} samples")
        print(f"Total masks shape: {self.masks.shape}")
        print(f"Total images shape: {self.images.shape}")

    def k_raspberries_occlusion(self, k=5,
                                max_occlusions_per_raspberry=3,
                                wanted_size_range_per_occlusion=(0.7, 0.9),
                                randomize_scale_bool=(False, 0.8, 1.2),
                                randomize_rotation_bool=(True, -180, 180),
                                nudge = (10,3), 
                                visualize_bool = False,
                                mode_sampling = 'weighted',
                                canvas_size=(1000, 1000)):
        """
        Occlusion of k single raspberries.
        Based on the idea of Copy & Paste as in Ghiasi et al., 
        "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation".

        Args:
            k (int) : Number of raspberries to occlude sequentially.
            max_occlusions_per_raspberry (int) : Maximum number of times a raspberry can be occluded.
            wanted_size_range_per_occlusion (tuple) : Desired size range (min, max) for each occlusion.
            randomize_scale_bool (tuple) : (apply_randomization, min_scale, max_scale) for random scaling.
            randomize_rotation_bool (tuple) : (apply_randomization, min_angle, max_angle) for random rotation.
            nudge (tuple) : (trials, radius) for random nudging.
            visualize_bool (bool) : Whether to visualize the occlusion process.
            mode_sampling (str) : 'weighted', 'uniform' or ('N_largest', N) sampling of raspberries.
            canvas_size (tuple) : (height, width) of the canvas to draw on. The first raspberry will be
                                placed at the center of this canvas.
        """
        
        if k < 2:
            raise ValueError("k must be at least 2")
        
        # Calculate mask sizes for all raspberries in dataset
        mask_sizes = np.array([np.sum(mask) for mask in self.masks])

        # 1. Select k raspberries from dataset
        if mode_sampling == 'weighted':
            weights = np.sqrt(mask_sizes)
            weights = weights / np.sum(weights)
            chosen_indices = np.random.choice(len(self.images), size=k, replace=True, p=weights)
        elif mode_sampling == 'uniform':
            chosen_indices = np.random.choice(len(self.images), size=k, replace=True)
        elif type(mode_sampling) == tuple and mode_sampling[0] == 'N_largest':
            N = mode_sampling[1]
            largest_indices = np.argsort(mask_sizes)[-N:]
            chosen_indices = np.random.choice(largest_indices, size=k, replace=True)
        
        # 2. Create blank canvas
        canvas_height, canvas_width = canvas_size
        composite_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # 3. Place first raspberry at center of canvas
        first_img = np.asarray(self.images[chosen_indices[0]].copy(), dtype=np.uint8)
        first_mask = np.asarray(self.masks[chosen_indices[0]].copy(), dtype=bool)
        
        # Get bounding box of first raspberry
        coords = np.argwhere(first_mask)
        if len(coords) == 0:
            raise ValueError("First raspberry has empty mask")
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1
        
        # Extract bounding box
        bbox_img = first_img[y_min:y_max+1, x_min:x_max+1]
        bbox_mask = first_mask[y_min:y_max+1, x_min:x_max+1]
        
        # Place at center of canvas
        center_x = (canvas_width - bbox_width) // 2
        center_y = (canvas_height - bbox_height) // 2
        
        composite_img[center_y:center_y+bbox_height, center_x:center_x+bbox_width][bbox_mask] = \
            bbox_img[bbox_mask]
        
        # Create composite mask (full canvas size)
        composite_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
        composite_mask[center_y:center_y+bbox_height, center_x:center_x+bbox_width] = bbox_mask

        # Track occlusion count for each raspberry
        occlusion_counts = np.zeros(k, dtype=int)
        original_mask_sizes = [np.sum(bbox_mask)]  # Size of first raspberry's actual mask
        
        # Store metadata
        occlusion_metadata = {
            'raspberry_ids': chosen_indices.tolist(),
            'canvas_size': canvas_size,
            'occlusion_history': [],
            'final_visible_sizes': []
        }
        
        # Keep track of all raspberry masks
        all_raspberry_masks = [composite_mask.copy()]
        all_raspberry_imgs = [first_img.copy()]

        # Keep track of center point of each raspberry mask
        ys, xs = np.where(composite_mask)
        all_raspberry_centers = [(int(xs.mean()), int(ys.mean()))]

        # Store ORIGINAL masks at their pasted locations
        all_original_masks_on_canvas = [composite_mask.copy()]
        
        print(f"Starting multi-raspberry occlusion with {k} raspberries...")
        print(f"Canvas size: {canvas_height}x{canvas_width}")
        print(f"First raspberry placed at center ({center_x}, {center_y})")
        print(f"Base raspberry size: {np.sum(composite_mask)} pixels")
        
        # 4. Add remaining k-1 raspberries sequentially
        for i in range(1, k):
            source_idx = i
            source_img = self.images[chosen_indices[i]].copy()
            source_mask = self.masks[chosen_indices[i]].copy()
            
            print(f"\n--- Adding raspberry {i+1}/{k} ---")
            
            # Calculate original size before transformation
            source_coords_original = np.argwhere(source_mask)
            if len(source_coords_original) > 0:
                sy_min, sx_min = source_coords_original.min(axis=0)
                sy_max, sx_max = source_coords_original.max(axis=0)
                original_source_size = np.sum(source_mask[sy_min:sy_max+1, sx_min:sx_max+1])
            else:
                original_source_size = np.sum(source_mask)
            
            print(f"Source raspberry size: {original_source_size} pixels")
            
            # 5. Apply transformations to source
            transformed_source_img = np.asarray(source_img.copy(), dtype=np.uint8)
            transformed_source_mask = np.asarray(source_mask.copy(), dtype=bool)   
            
            # Scale
            if randomize_scale_bool[0]:
                scale_factor = np.random.uniform(randomize_scale_bool[1], randomize_scale_bool[2])
                h, w = source_mask.shape
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                transformed_source_img = cv2.resize(transformed_source_img, (new_w, new_h), 
                                                interpolation=cv2.INTER_LINEAR)
                transformed_source_mask = cv2.resize(transformed_source_mask.astype(np.uint8), 
                                                    (new_w, new_h), 
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Rotation
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
            
            # 6. Select random anchor points for placement
            all_existing_coords = []
            for j, mask in enumerate(all_raspberry_masks[:i]):
                coords = np.argwhere(mask)
                if len(coords) > 0:
                    all_existing_coords.extend(coords.tolist())
            
            if len(all_existing_coords) == 0:
                print("Warning: No valid coordinates found. Stopping.")
                break
            
            all_existing_coords = np.array(all_existing_coords)

            # Retry placement with nudging
            max_trials = 200
            trial = 0
            failed_trials = 0
            valid_placement = False

            # Precompute source coords once
            source_coords = np.argwhere(transformed_source_mask)

            if len(source_coords) == 0:
                print("Warning: Source mask is empty. Skipping.")
                continue

            while trial < max_trials:
                trial += 1

                # NUDGE: remove center circular region every k failed trials
                if failed_trials > 0 and nudge is not None:
                    k_trials, radius = nudge
                    if failed_trials % k_trials == 0 and len(source_coords) > 0:
                        mask_center = np.mean(source_coords, axis=0)
                        distances = np.linalg.norm(source_coords - mask_center, axis=1)
                        source_coords = source_coords[distances > radius]

                        print(f" Nudging anchors outward — removed inner radius {radius}px")

                        if len(source_coords) == 0:
                            source_coords = np.argwhere(transformed_source_mask)

                # Select random target anchor
                target_anchor_idx = np.random.randint(len(all_existing_coords))
                target_anchor_y, target_anchor_x = all_existing_coords[target_anchor_idx]
                
                # Select random source anchor
                source_anchor_idx = np.random.randint(len(source_coords))
                source_anchor_y, source_anchor_x = source_coords[source_anchor_idx]
                
                # Calculate offset
                offset_y = target_anchor_y - source_anchor_y
                offset_x = target_anchor_x - source_anchor_x
                
                print(f"Placing at offset ({offset_y}, {offset_x})")
                
                # Calculate where source will be pasted
                h_target, w_target = composite_img.shape[:2]  # This is now canvas_height, canvas_width
                h_source, w_source = transformed_source_mask.shape
                
                # Boundary handling
                src_y_start = max(0, -offset_y)
                src_x_start = max(0, -offset_x)
                src_y_end = min(h_source, h_target - offset_y)
                src_x_end = min(w_source, w_target - offset_x)
                
                dst_y_start = max(0, offset_y)
                dst_x_start = max(0, offset_x)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                
                if src_y_end <= src_y_start or src_x_end <= src_x_start:
                    print("Warning: No valid overlap. Skipping.")
                    failed_trials += 1
                    continue

                # Check if destination bbox is within canvas bounds
                # Does not work and also not necessarily needed ... i.e. even if we paste a bit out 
                # of bounds, it doesn't mess up anything
            #    if dst_y_start < 0 or dst_x_start < 0 or dst_y_end > h_target or dst_x_end > w_target:
            #        print("Warning: Destination out of bounds. Skipping.")
            #        failed_trials += 1
            #        continue
                
                # Extract source region
                source_region_mask = transformed_source_mask[src_y_start:src_y_end, src_x_start:src_x_end].astype(bool)
                source_region_img = transformed_source_img[src_y_start:src_y_end, src_x_start:src_x_end]
                
                # Check which raspberries will be affected
                affected_raspberries = []
                for j in range(len(all_raspberry_masks[:i])):
                    existing_mask_region = all_raspberry_masks[j][dst_y_start:dst_y_end, dst_x_start:dst_x_end]
                    
                    overlap = existing_mask_region & source_region_mask
                    overlap_amount = np.sum(overlap)
                    
                    if overlap_amount > 0:
                        current_size = np.sum(all_raspberry_masks[j])
                        
                        affected_raspberries.append({
                            'index': j,
                            'overlap_pixels': overlap_amount,
                            'current_size': current_size,
                            'remaining_size': current_size - overlap_amount,
                            'full_percentage' : (current_size - overlap_amount) / original_mask_sizes[j] * 100
                        })
                
                print(f"This placement will affect {len(affected_raspberries)} raspberries:")
                for aff in affected_raspberries:
                    print(f"  Raspberry #{aff['index']+1}: -{aff['overlap_pixels']} pixels "
                        f"({aff['full_percentage'] :.1f}% remaining)")
                
                # Validate constraints
                valid_placement = True
                for aff in affected_raspberries:
                    j = aff['index']
                    
                    if occlusion_counts[j] >= max_occlusions_per_raspberry:
                        print(f"  Rejected: Raspberry #{j+1} already occluded {occlusion_counts[j]} times")
                        valid_placement = False
                        break
                    
                    min_occlusion_pct = wanted_size_range_per_occlusion[0] * 100
                    max_occlusion_pct = wanted_size_range_per_occlusion[1] * 100
                    
                    if aff['full_percentage'] < min_occlusion_pct:
                        print(f"  Rejected: Occlusion too large for raspberry #{j+1} "
                            f"({aff['full_percentage']:.1f}% < {min_occlusion_pct}%)")
                        valid_placement = False
                        break
                    
                    if aff['full_percentage'] > max_occlusion_pct:
                        print(f"  Rejected: Occlusion too small for raspberry #{j+1} "
                            f"({aff['full_percentage']:.1f}% > {max_occlusion_pct}%)")
                        valid_placement = False
                        break

                    if aff['overlap_pixels'] == np.sum(source_region_mask):
                        print(f"  Rejected: Source entirely contained within raspberry #{j+1}")
                        valid_placement = False
                        break
                
                if valid_placement:
                    print("Placement accepted!")
                    break
            
                failed_trials += 1
            
            if not valid_placement:
                print(f"Failed to find valid placement for raspberry {i+1} after {max_trials} trials. Skipping.")
                continue

            # Paste source onto composite
            composite_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask] = \
                source_region_img[source_region_mask]
            
            # Update affected raspberry masks
            for aff in affected_raspberries:
                j = aff['index']
                all_raspberry_masks[j][dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask] = False
                occlusion_counts[j] += 1
                
                occlusion_metadata['occlusion_history'].append({
                    'step': i,
                    'source_raspberry': source_idx,
                    'target_raspberry': j,
                    'occlusion_amount_pixels': int(aff['overlap_pixels']),
                    'full_percentage': float(aff['full_percentage']),
                    'remaining_size': int(aff['remaining_size'])
                })
            
            # Add new source raspberry to tracking
            new_raspberry_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
            new_raspberry_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = source_region_mask
            all_raspberry_masks.append(new_raspberry_mask)
            all_raspberry_imgs.append(source_img)
            
            # Store original size (actual pixels in pasted region)
            original_mask_sizes.append(np.sum(source_region_mask))
            
            # Calculate center
            ys, xs = np.where(new_raspberry_mask)
            if len(xs) > 0:
                center_x = int(xs.mean())
                center_y = int(ys.mean())
            else:
                center_x = (dst_x_start + dst_x_end) // 2
                center_y = (dst_y_start + dst_y_end) // 2
            all_raspberry_centers.append((center_x, center_y))
            
            # Store original mask at canvas location
            original_mask_on_canvas = np.zeros((canvas_height, canvas_width), dtype=bool)
            original_mask_on_canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = source_region_mask
            all_original_masks_on_canvas.append(original_mask_on_canvas)
            
            print(f"Successfully added raspberry {i+1}")
        
        # Calculate final statistics
        for j in range(min(len(all_raspberry_masks), k)):
            mask = all_raspberry_masks[j]
            visible_size = np.sum(mask)
            original_size = original_mask_sizes[j]
            occlusion_metadata['final_visible_sizes'].append({
                'raspberry_id': j,
                'original_size': int(original_size),
                'final_visible_size': int(visible_size),
                'times_occluded': int(occlusion_counts[j]),
                'full_percentage': float(visible_size / original_size * 100) if original_size > 0 else 0
            })
        
        print("\n=== Final Summary ===")
        for info in occlusion_metadata['final_visible_sizes']:
            print(f"Raspberry #{info['raspberry_id']+1}: {info['final_visible_size']}/{info['original_size']} pixels "
                f"({info['full_percentage']:.1f}% remaining, {info['times_occluded']} occlusion events)")

        if visualize_bool:
            num_masks = min(len(all_raspberry_masks), k)
            total_plots = num_masks + 1
            
            cols = min(4, total_plots)
            rows = int(np.ceil(total_plots / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)
            
            axes[0].imshow(np.asarray(composite_img, dtype=np.uint8))
            axes[0].set_title("Composite Image")
            axes[0].axis("off")
            
            for i in range(num_masks):
                current_mask = all_raspberry_masks[i]
                original_mask = all_original_masks_on_canvas[i]
                
                vis = np.zeros((*current_mask.shape, 3), dtype=np.uint8)
                vis[..., 0] = original_mask.astype(np.uint8) * 255
                vis[..., 0] |= current_mask.astype(np.uint8) * 255
                vis[..., 1] = current_mask.astype(np.uint8) * 255
                
                axes[i + 1].imshow(vis)
                axes[i + 1].set_title(f"Raspberry #{i+1}\nRed=Original, Yellow=Visible")
                axes[i + 1].axis("off")
            
            for ax in axes[total_plots:]:
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()

        # Return only masks that were occluded at least once
        occluded_masks = []
        for j in range(len(all_raspberry_masks)):
            if occlusion_counts[j] > 0:
                occluded_masks.append(all_raspberry_masks[j])

        return composite_img, occluded_masks

    def single_raspberry_occlusion(self, 
                                wanted_size_range=None,
                                randomize_scale_bool=(False, 0.2, 2.0), 
                                randomize_rotation_bool=(False, -30, 30),
                                visualize_bool = True,
                                reassign_source_target_bool = False,
                                sampling_mode = 'uniform'):
        """
        Occlusion of 2 single raspberries.
        Based on the idea of Copy & Paste as in Ghiasi et al., 
        "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation".

        Args:
            wanted_size_range (tuple) : (min_pixels_factor, max_pixels_factor) range for the remaining
                                        visible area of the target raspberry
            randomize_scale_bool (tuple) : If first element is True, randomly scale the source 
                                        raspberry by a factor sampled uniformly from the range
                                        defined by the second and third elements of the tuple.
            randomize_rotation_bool (tuple) : If first element is True, randomly rotate the source
                                            raspberry by an angle sampled uniformly from the range
                                            defined by the second and third elements of the tuple.
            visualize_bool (bool) : Whether to visualize the occlusion process.
            reassign_source_target_bool (bool) : Reassign source and target raspberries such that
                                                source is always larger than target. (i.e. realistic, 
                                                since the larger raspberry is typically more prominent in the image)
            sampling_mode (str) : ('N_largest', N) or 'uniform' sampling of raspberries


        Workflow : 
            1. Select two "random" (depending on sampling strategy) images of single raspberries from the dataset (source and target raspberry).
            2. Apply random transformations (scaling, rotation) to the source raspberry, if wanted.
            3. Paste the source raspberry region onto the target image at a random location within
            the locality of the raspberry in the target image (Targeted Pasting); this is implemented
            by selecting two anchor points : source anchor is a (x,y) coordinate within the source raspberry
             and target anchor is a (x,y) coordinate within the target raspberry. We then calculate the offset
             and paste the source raspberry onto the target image at the calculated location. 
            4. Check that occlusion constraints are satisfied: : 
                a) wanted_size_range is fulfilled, i.e. the remaining visible area of the target raspberry is between wanted_size_range[0] * original_size and wanted_size_range[1] * original_size
                b) source raspberry is not entirely contained within target raspberry.
            If constraints are not satisfied, repeat step 3 until a maximum number of trials is reached (set to 200 currently).
            

        """

        if sampling_mode == 'uniform':
            chosen_idx = np.random.choice(len(self.images), size=2, replace=True)
        elif type(sampling_mode) == tuple and sampling_mode[0] == 'N_largest':
            N = sampling_mode[1]
            mask_sizes = np.array([np.sum(mask) for mask in self.masks])
            largest_indices = np.argsort(mask_sizes)[-N:]
            chosen_idx = np.random.choice(largest_indices, size=2, replace=True)

        target_img = self.images[chosen_idx[0]].copy()
        source_img = self.images[chosen_idx[1]].copy()

        target_mask = self.masks[chosen_idx[0]].copy()
        source_mask = self.masks[chosen_idx[1]].copy()

        # 2. Paste the source raspberry region onto the target image
        new_target_img, new_target_mask = self._paste(
            target_mask, 
            source_mask, 
            source_img, 
            target_img,
            wanted_size_range=wanted_size_range,
            randomize_scale_bool=randomize_scale_bool,
            randomize_rotation_bool=randomize_rotation_bool,
            visualize_bool=visualize_bool,
            reassign_source_target_bool=reassign_source_target_bool,
        )

        return new_target_img, new_target_mask

    def create_artificial_bonnet(self, k=25, 
                                field_size=(961, 644),
                                show_field_boundary=True,
                                randomize_scale_bool=(False, 0.8, 1.2),
                                randomize_rotation_bool=(False, -180, 180),
                                wanted_size_range_per_occlusion=(0.7, 0.9),
                                visualize_bool=True,
                                mode_sampling ='weighted',
                                canvas_size = (1000,1000)):
        """
        Create an artificial bonnet by placing k raspberries randomly within a defined field.
        
        Args:
            k (int): Number of raspberries to place in the field
            field_size (tuple): (width, height) of the bonnet field in pixels
            show_field_boundary (bool): If True, draw green rectangle showing field boundaries
            randomize_scale_bool (tuple): (apply, min_scale, max_scale) for random scaling
            randomize_rotation_bool (tuple): (bool, min_angle, max_angle) for random rotation
            wanted_size_range_per_occlusion (tuple): (min_factor, max_factor) - when a raspberry
                                                    gets occluded, its remaining size must be
                                                    between min_factor and max_factor of original size
            visualize_bool (bool): Whether to visualize results
            mode_sampling (str): 'weighted', 'uniform' or ('N_largest', N) sampling of raspberries
            canvas_size (tuple): (height, width) of the canvas
        
        Returns:
            composite_img: Final bonnet image with all raspberries
            occluded_masks: List of raspberry masks that were occluded at least once
            placement_metadata: Dict with placement information
        """
        
        field_width, field_height = field_size
        
        # Create black canvas
        canvas_height, canvas_width = canvas_size
        composite_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate offset to center the field
        field_offset_x = (canvas_width - field_width) // 2
        field_offset_y = (canvas_height - field_height) // 2
        
        # Draw field boundary if requested (centered)
        if show_field_boundary:
            cv2.rectangle(composite_img, 
                        (field_offset_x, field_offset_y), 
                        (field_offset_x + field_width - 1, field_offset_y + field_height - 1), 
                        (0, 255, 0), thickness=3)
        
        # Calculate mask sizes for all raspberries in dataset
        mask_sizes = np.array([np.sum(mask) for mask in self.masks])
        
        if mode_sampling == 'weighted':
            # Create weighted selection (prefer larger raspberries)
            weights = np.sqrt(mask_sizes)
            weights = weights / np.sum(weights)
            chosen_indices = np.random.choice(len(self.images), size=k, replace=True, p=weights)
        elif mode_sampling == 'uniform':
            chosen_indices = np.random.choice(len(self.images), size=k, replace=True)
        elif type(mode_sampling) == tuple and mode_sampling[0] == 'N_largest':
            N = mode_sampling[1]
            largest_indices = np.argsort(mask_sizes)[-N:]
            chosen_indices = np.random.choice(largest_indices, size=k, replace=True)

        print(f"Creating artificial bonnet with {k} raspberries in {field_width}x{field_height} field")
        print(f"Field centered at offset ({field_offset_x}, {field_offset_y})")
        print(f"Occlusion range: {wanted_size_range_per_occlusion[0]*100:.0f}%-{wanted_size_range_per_occlusion[1]*100:.0f}% of original size must remain")
        
        # Track all raspberry masks and metadata
        all_masks = []
        all_original_sizes = []  # Store original sizes before any occlusion
        all_original_masks_on_canvas = []  # Store TRANSFORMED original masks at canvas positions
        all_raspberry_centers = []
        placement_metadata = {
            'raspberry_ids': chosen_indices.tolist(),
            'field_offset': (field_offset_x, field_offset_y),
            'positions': [],
            'placements': [],
            'occlusions': []
        }
        
        # Place raspberries one by one
        for i, idx in enumerate(chosen_indices):
            source_img = np.asarray(self.images[idx].copy(), dtype=np.uint8)
            source_mask = np.asarray(self.masks[idx].copy(), dtype=bool)
            
            print(f"\n--- Placing raspberry {i+1}/{k} (ID: {idx}, size: {np.sum(source_mask)} pixels) ---")
            
            # Apply transformations
            transformed_source_img = source_img.copy()
            transformed_source_mask = source_mask.copy()
            
            # Scale
            if randomize_scale_bool[0]:
                scale_factor = np.random.uniform(randomize_scale_bool[1], randomize_scale_bool[2])
                h, w = source_mask.shape
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                transformed_source_img = cv2.resize(transformed_source_img, (new_w, new_h), 
                                                interpolation=cv2.INTER_LINEAR)
                transformed_source_mask = cv2.resize(transformed_source_mask.astype(np.uint8), 
                                                    (new_w, new_h), 
                                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                print(f"  Scaled by {scale_factor:.2f}x")
            
            # Rotation
            angle = 0.0
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
                print(f"  Rotated by {angle:.1f}°")
            
            # Calculate bounding box of the TRANSFORMED raspberry
            coords = np.argwhere(transformed_source_mask)
            
            if len(coords) == 0:
                print(f"  Warning: Empty mask for raspberry {i+1}, skipping")
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            bbox_height = y_max - y_min + 1
            bbox_width = x_max - x_min + 1
            
            print(f"  Raspberry bounding box: {bbox_width}x{bbox_height} pixels")
            
            # Check if raspberry fits in field
            if bbox_width > field_width or bbox_height > field_height:
                print(f"  Warning: Raspberry {i+1} too large for field, skipping")
                continue
            
            # Extract bounding box region (used in all trials)
            bbox_img = transformed_source_img[y_min:y_max+1, x_min:x_max+1]
            bbox_mask = transformed_source_mask[y_min:y_max+1, x_min:x_max+1]
            original_raspberry_size = np.sum(bbox_mask)
            
            # Try multiple placements until occlusion constraints are satisfied
            placement_found = False
            trials = 200
            
            for trial in range(trials):
                # Random position within field
                field_x = np.random.randint(0, field_width - bbox_width + 1)
                field_y = np.random.randint(0, field_height - bbox_height + 1)
                
                # Convert to canvas coordinates
                canvas_x = field_offset_x + field_x
                canvas_y = field_offset_y + field_y
                
                # Calculate which existing raspberries will be affected
                affected_raspberries = []
                for j, existing_mask in enumerate(all_masks):
                    existing_region = existing_mask[canvas_y:canvas_y+bbox_height, canvas_x:canvas_x+bbox_width]
                    overlap = existing_region & bbox_mask
                    overlap_amount = np.sum(overlap)
                    
                    if overlap_amount > 0:
                        current_size = np.sum(existing_mask)
                        original_size = all_original_sizes[j]
                        remaining_size = current_size - overlap_amount
                        remaining_percentage = remaining_size / original_size
                        
                        affected_raspberries.append({
                            'index': j,
                            'overlap_pixels': overlap_amount,
                            'current_size': current_size,
                            'original_size': original_size,
                            'remaining_size': remaining_size,
                            'remaining_percentage': remaining_percentage
                        })
                
                # Validate occlusion constraints for ALL affected raspberries
                valid_placement = True
                
                for aff in affected_raspberries:
                    min_factor, max_factor = wanted_size_range_per_occlusion
                    
                    if aff['remaining_percentage'] < min_factor:
                        if trial % 20 == 0:
                            print(f"  Trial {trial+1}: Raspberry #{aff['index']+1} would be {aff['remaining_percentage']*100:.1f}% "
                                f"(< {min_factor*100:.0f}% min)")
                        valid_placement = False
                        break
                    
                    if aff['remaining_percentage'] > max_factor:
                        if trial % 20 == 0:
                            print(f"  Trial {trial+1}: Raspberry #{aff['index']+1} would be {aff['remaining_percentage']*100:.1f}% "
                                f"(> {max_factor*100:.0f}% max)")
                        valid_placement = False
                        break

                    if aff['overlap_pixels'] == np.sum(bbox_mask):
                        if trial % 20 == 0:
                            print(f"  Trial {trial+1}: Raspberry #{aff['index']+1} would entirely contain new raspberry")
                        valid_placement = False
                        break
                
                if valid_placement:
                    placement_found = True
                    print(f"  ✓ Valid placement found on trial {trial+1}")
                    
                    if len(affected_raspberries) > 0:
                        print(f"  Occludes {len(affected_raspberries)} existing raspberries:")
                        for aff in affected_raspberries:
                            print(f"    Raspberry #{aff['index']+1}: {aff['remaining_percentage']*100:.1f}% remaining "
                                f"({aff['overlap_pixels']} pixels occluded)")
                    
                    # Apply the placement
                    composite_img[canvas_y:canvas_y+bbox_height, canvas_x:canvas_x+bbox_width][bbox_mask] = \
                        bbox_img[bbox_mask]
                    
                    # Update all affected raspberry masks
                    for aff in affected_raspberries:
                        j = aff['index']
                        all_masks[j][canvas_y:canvas_y+bbox_height, canvas_x:canvas_x+bbox_width][bbox_mask] = False
                        
                        placement_metadata['occlusions'].append({
                            'new_raspberry': i,
                            'occluded_raspberry': j,
                            'overlap_pixels': int(aff['overlap_pixels']),
                            'remaining_percentage': float(aff['remaining_percentage'])
                        })
                    
                    # Add new raspberry mask to tracking (full canvas size)
                    new_mask = np.zeros(canvas_size, dtype=bool)
                    new_mask[canvas_y:canvas_y+bbox_height, canvas_x:canvas_x+bbox_width] = bbox_mask
                    all_masks.append(new_mask)
                    all_original_sizes.append(original_raspberry_size)
                    
                    # Store TRANSFORMED original mask at canvas location (for visualization)
                    # This is the mask AFTER scaling/rotation but BEFORE occlusions
                    original_mask_on_canvas = np.zeros(canvas_size, dtype=bool)
                    original_mask_on_canvas[canvas_y:canvas_y+bbox_height, canvas_x:canvas_x+bbox_width] = bbox_mask
                    all_original_masks_on_canvas.append(original_mask_on_canvas)
                    
                    # Calculate center
                    ys, xs = np.where(new_mask)
                    if len(xs) > 0:
                        center_x = int(xs.mean())
                        center_y = int(ys.mean())
                    else:
                        center_x = canvas_x + bbox_width // 2
                        center_y = canvas_y + bbox_height // 2
                    all_raspberry_centers.append((center_x, center_y))
                    
                    # Record placement
                    placement_metadata['positions'].append((canvas_x, canvas_y))
                    placement_metadata['placements'].append({
                        'raspberry_id': int(idx),
                        'canvas_position': (int(canvas_x), int(canvas_y)),
                        'field_position': (int(field_x), int(field_y)),
                        'bbox_size': (int(bbox_width), int(bbox_height)),
                        'scale_factor': float(scale_factor) if randomize_scale_bool[0] else 1.0,
                        'rotation': float(angle),
                        'mask_pixels': int(original_raspberry_size),
                        'trials_needed': trial + 1
                    })
                    
                    break  # Exit trial loop
            
            if not placement_found:
                print(f"Failed to find valid placement after {trials} trials, skipping raspberry {i+1}")
        
        print(f"\n=== Bonnet Creation Complete ===")
        print(f"Successfully placed {len(all_masks)} / {k} raspberries")
        print(f"Total occlusion events: {len(placement_metadata['occlusions'])}")
        
        # Calculate final statistics
        final_stats = []
        for i, mask in enumerate(all_masks):
            visible_pixels = np.sum(mask)
            original_pixels = all_original_sizes[i]
            occlusion_pct = (original_pixels - visible_pixels) / original_pixels * 100 if original_pixels > 0 else 0
            
            final_stats.append({
                'raspberry_index': i,
                'original_pixels': original_pixels,
                'visible_pixels': int(visible_pixels),
                'occlusion_percentage': float(occlusion_pct),
                'full_percentage': float(visible_pixels / original_pixels * 100)
            })
        
        placement_metadata['final_stats'] = final_stats

        # Print Final Summary
        print("\n=== Final Summary ===")
        for stats in final_stats:
            print(f"Raspberry #{stats['raspberry_index']+1}: {stats['visible_pixels']}/{stats['original_pixels']} pixels "
                f"({stats['full_percentage']:.1f}% remaining)")
        
        # Visualization
        if visualize_bool:
            num_masks = len(all_masks)
            total_plots = num_masks + 1
            
            cols = min(4, total_plots)
            rows = int(np.ceil(total_plots / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)
            
            axes[0].imshow(np.asarray(composite_img, dtype=np.uint8))
            axes[0].set_title("Composite Bonnet")
            axes[0].axis("off")
            
            for i in range(num_masks):
                current_mask = all_masks[i]
                original_mask = all_original_masks_on_canvas[i]
                
                # Build RGB visualization
                vis = np.zeros((*current_mask.shape, 3), dtype=np.uint8)
                
                # Original mask (transformed, before occlusions) = RED
                vis[..., 0] = original_mask.astype(np.uint8) * 255
                
                # Current visible mask (after occlusions) = YELLOW (red + green)
                vis[..., 0] |= current_mask.astype(np.uint8) * 255
                vis[..., 1] = current_mask.astype(np.uint8) * 255
                
                axes[i + 1].imshow(vis)
                
                visible_pct = 100 - final_stats[i]['occlusion_percentage']
                axes[i + 1].set_title(f"Raspberry #{i+1}\n{visible_pct:.0f}% visible\nRed=Original, Yellow=Visible")
                axes[i + 1].axis("off")
            
            for ax in axes[total_plots:]:
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(8, 8))
            plt.imshow(np.asarray(composite_img, dtype=np.uint8))
            plt.axis('off')
            plt.show()
        
        # Return only masks that were occluded at least once
        occluded_masks = []
        for i, stats in enumerate(final_stats):
            if stats['occlusion_percentage'] > 0:
                occluded_masks.append(all_masks[i])
        
        return composite_img, occluded_masks, placement_metadata

    def _paste(self, target_mask, 
            source_mask, 
            source_img,
            target_img, 
            wanted_size_range = (0.3, 0.7),
            randomize_scale_bool=(False, 0.2, 2.0), 
            randomize_rotation_bool=(False, -30, 30),
            visualize_bool=False,
            reassign_source_target_bool = False):
        """
        Paste the source raspberry onto the target raspberry at a random location within
        the locality of the target raspberry.

        Args :
            target_mask (np.array) : Binary mask of the target raspberry.
            source_mask (np.array) : Binary mask of the source raspberry.
            wanted_size_range (tuple) : If provided, (min_pixels_factor, max_pixels_factor) for remaining visible area. 
                                        min_pixels_factor and max_pixels_factor are fractions of the original target mask size.
                                        i.e. if wanted_size_range=(0.3, 0.7) and the original target mask has 1000 pixels,
                                        the remaining visible area after occlusion should be between 300 and 700 pixels.
            source_img (np.array) : Source raspberry image.
            target_img (np.array) : Target raspberry image.
            randomize_scale_bool (tuple) : (bool, min_scale, max_scale)
            randomize_rotation_bool (tuple) : (bool, min_angle, max_angle)
            visualize_bool (bool) : Whether to visualize the result.
            reassign_source_target_bool (bool) : If True, find whatever mask is larger and assign it as source.
                                                    The rational behind this is that a smaller mask represents
                                                    a raspberry that might be more occluded already, and using an
                                                    occluded example as source for occlusion might lead to less realistic results.


        Notes : 
            For more realism, dont use scaling, i.e. the raspberries should be of similar size.

        """

        if reassign_source_target_bool:
            # Reassign source and target based on mask size
            target_size = np.sum(target_mask)
            source_size = np.sum(source_mask)
            if source_size < target_size:
                # Swap
                target_mask, source_mask = source_mask, target_mask
                target_img, source_img = source_img, target_img

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
            

            # Different interpolation methods : https://blog.roboflow.com/image-resizing/
            transformed_source_img = cv2.resize(transformed_source_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            transformed_source_mask = cv2.resize(transformed_source_mask.astype(np.uint8), 
                                                (new_w, new_h), 
                                                interpolation=cv2.INTER_NEAREST).astype(bool) # inter_nearest for mask to not create intermediate values
        
        # Rotation transformation
        if randomize_rotation_bool[0]:
            angle = np.random.uniform(randomize_rotation_bool[1], randomize_rotation_bool[2])
            h, w = transformed_source_mask.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            transformed_source_img = cv2.warpAffine(transformed_source_img, M, (w, h), 
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, # fill border with black (i.e. we anyway have a black background)
                                                borderValue=0)
            transformed_source_mask = cv2.warpAffine(transformed_source_mask.astype(np.uint8), M, (w, h),
                                                    flags=cv2.INTER_NEAREST,
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0).astype(bool)
        
        # Get coordinates of True pixels in both masks
        target_coords = np.argwhere(target_mask)
        source_coords = np.argwhere(transformed_source_mask)

        # Get number of pixels in original target mask
        original_target_size = np.sum(target_mask)
        
    
        # Try up to 200 paste attempts
        max_attempts = 200
        
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
            
            # Calculate valid paste region ; boundary handling (i.e. that we don't do negative indexing)
            # Decide which parts of source to actually paste 
            src_y_start = max(0, -offset_y) 
            src_x_start = max(0, -offset_x)
            src_y_end = min(h_source, h_target - offset_y)
            src_x_end = min(w_source, w_target - offset_x)
            
            # Corresponding region in target
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
            
            # Count occluded pixels

            occluded_pixels = np.sum(new_target_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask])
            
            # Update mask: Remove occluded parts (set to False where source overlaps)
            new_target_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask] = False
            
            # Check if remaining target mask meets criteria
            remaining_size = np.sum(new_target_mask)

            valid_paste = True
 

            # Check that size is within range
            if not (remaining_size >= wanted_size_range[0] * original_target_size and \
               remaining_size <= wanted_size_range[1] * original_target_size):
               print(f" Attempt {attempt+1}: Remaining size {remaining_size} pixels not in range {wanted_size_range[0] * original_target_size:.0f}-{wanted_size_range[1] * original_target_size:.0f} pixels")
               valid_paste = False


            if occluded_pixels == np.sum(source_region_mask):
                print(f" Attempt {attempt+1}: Source raspberry entirely within target raspberry, invalid paste")
                valid_paste = False

            if valid_paste:
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
    data_path_all_samples = 'dataset_single_objects/GT/processed/'
    synthetic_occlusion = SyntheticOcclusion(base_path= data_path_all_samples, sample_folders = None)

    BONNET_SIZE = (961, 644)  # Width, Height taken from bonnet of img_001
    new_img, new_mask = synthetic_occlusion.single_raspberry_occlusion(
        wanted_size_range=(0.1,0.3),
        randomize_scale_bool=(False, 0.2, 0.8),
        randomize_rotation_bool=(False, -180,180),
        #choose_idx=[3, 6]   
        visualize_bool=True,
        reassign_source_target_bool=False,
        sampling_mode= ('N_largest', 20))
   
 #   new_img, new_masks = synthetic_occlusion.k_raspberries_occlusion(
 #       k=30,
 #       max_occlusions_per_raspberry=4,
 #       wanted_size_range_per_occlusion=(0.7, 1.0),
 #       randomize_scale_bool=(False, 0.5, 0.8),
 #       randomize_rotation_bool=(False, -180, 180),
 #       visualize_bool=True,
 #       mode_sampling = ('N_largest', 20)
 #   )

 #   new_img, new_masks, metadata = synthetic_occlusion.create_artificial_bonnet(
 #       k=20,
 #       field_size=BONNET_SIZE,
 #       show_field_boundary=True,
 #       randomize_scale_bool=(False, 0.7, 1.0),
 #       randomize_rotation_bool=(False, -180, 180),
 #       wanted_size_range_per_occlusion=(0.5, 1.0),
 #       visualize_bool=True,
 #       mode_sampling = ('N_largest', 40),
 #       canvas_size=(1200,1200)
 #   )

 

if __name__ == "__main__":
    main()
