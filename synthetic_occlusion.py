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



    def k_raspberries_occlusion(self, k=5,
                            max_occlusions_per_raspberry=3,
                            min_remaining_size_factor=0.2,
                            wanted_size_range_per_occlusion=(0.05, 0.3),
                            randomize_scale_bool=(False, 0.8, 1.2),
                            randomize_rotation_bool=(True, -180, 180),
                            nudge = (10,3), 
                            visualize_bool = False):
        """
        Create occlusion patterns with k raspberries overlaying each other.

        nudge : (trials, radius)
        """
        
        if k < 2:
            raise ValueError("k must be at least 2")
        
        # 1. Select k raspberries from dataset
        chosen_indices = np.random.choice(len(self.images), size=k, replace=True)
        
        # Initialize with first raspberry as base
        composite_img = self.images[chosen_indices[0]].copy()
        composite_mask = self.masks[chosen_indices[0]].copy()
        
        # Track occlusion count for each raspberry
        occlusion_counts = np.zeros(k, dtype=int)
        original_mask_sizes = [np.sum(self.masks[idx]) for idx in chosen_indices]
        
        # Store metadata
        occlusion_metadata = {
            'raspberry_ids': chosen_indices.tolist(),
            'occlusion_history': [],
            'final_visible_sizes': []
        }
        
        # Keep track of all raspberry masks
        all_raspberry_masks = [composite_mask.copy()]
        all_raspberry_imgs = [composite_img.copy()]

        # Keep track of center point of each raspberry mask after it has been placed on the composite img
        ys, xs = np.where(composite_mask)
        all_raspberry_centers = [(int(xs.mean()), int(ys.mean()))]

    
        
        print(f"Starting multi-raspberry occlusion with {k} raspberries...")
        print(f"Base raspberry size: {np.sum(composite_mask)} pixels")
        
        # 2. Add remaining k-1 raspberries sequentially
        for i in range(1, k):
            source_idx = i
            source_img = self.images[chosen_indices[i]].copy()
            source_mask = self.masks[chosen_indices[i]].copy()
            
            print(f"\n--- Adding raspberry {i+1}/{k} ---")
            print(f"Source raspberry size: {np.sum(source_mask)} pixels")
            
            # 3. Apply transformations to source
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
            
            # 4. Select random anchor points for placement
            # Instead of selecting a target raspberry, just place it randomly within existing raspberries
            all_existing_coords = []
            for j, mask in enumerate(all_raspberry_masks[:i]):  # Only existing raspberries
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

                        # Safety fallback if we removed too much
                        if len(source_coords) == 0:
                            source_coords = np.argwhere(transformed_source_mask)
            

                # Select random target anchor from ALL existing raspberry pixels
                target_anchor_idx = np.random.randint(len(all_existing_coords))
                target_anchor_y, target_anchor_x = all_existing_coords[target_anchor_idx]
                
                # Select random source anchor
                source_coords = np.argwhere(transformed_source_mask)
                if len(source_coords) == 0:
                    print("Warning: Source mask is empty. Skipping.")
                    continue
                
                source_anchor_idx = np.random.randint(len(source_coords))
                source_anchor_y, source_anchor_x = source_coords[source_anchor_idx]
                
                # Calculate offset
                offset_y = target_anchor_y - source_anchor_y
                offset_x = target_anchor_x - source_anchor_x
                
                print(f"Placing at offset ({offset_y}, {offset_x})")
                
                # 5. Calculate where source will be pasted and check ALL affected raspberries
                h_target, w_target = composite_img.shape[:2]
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
                    continue
                
                # Extract source region
                source_region_mask = transformed_source_mask[src_y_start:src_y_end, src_x_start:src_x_end].astype(bool)
                source_region_img = transformed_source_img[src_y_start:src_y_end, src_x_start:src_x_end]
                
                # 6. Check which raspberries will be affected and by how much
                affected_raspberries = []
                for j in range(len(all_raspberry_masks[:i])):  # Only existing raspberries
                    # Extract the region of this raspberry's mask that overlaps with source placement
                    existing_mask_region = all_raspberry_masks[j][dst_y_start:dst_y_end, dst_x_start:dst_x_end]
                    
                    # Calculate overlap
                    overlap = existing_mask_region & source_region_mask
                    overlap_amount = np.sum(overlap)
                    
                    if overlap_amount > 0:
                        current_size = np.sum(all_raspberry_masks[j])
                        occlusion_percentage = overlap_amount / current_size * 100
                        
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
                        f"({aff['full_percentage'] :.1f}% remaining of this raspberry object)")
                
                # 7. Validate constraints for ALL affected raspberries
                valid_placement = True
                for aff in affected_raspberries:
                    j = aff['index']
                    
                    # Check max occlusions
                    if occlusion_counts[j] >= max_occlusions_per_raspberry:
                        print(f"  Rejected: Raspberry #{j+1} already occluded {occlusion_counts[j]} times")
                        valid_placement = False
                        break
                    
                    # Check minimum remaining size
                    if min_remaining_size_factor is not None:
                        min_allowed = int(min_remaining_size_factor * original_mask_sizes[j])
                        if aff['remaining_size'] < min_allowed:
                            print(f"  Rejected: Raspberry #{j+1} would go below minimum size "
                                f"({aff['remaining_size']} < {min_allowed})")
                            valid_placement = False
                            break
                    
                    # Check per-occlusion size range
                    min_occlusion_pct = wanted_size_range_per_occlusion[0] * 100 # print in percentage
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
            

          #  if visualize_bool:
          #      plt.figure(figsize=(8, 8))
          #      plt.imshow(np.asarray(composite_img, dtype=np.uint8))
          #      plt.title(f"After adding raspberry {i+1}")
          #      plt.axis('off')
          #      plt.show()
            
            # Update ALL affected raspberry masks
            for aff in affected_raspberries:
                j = aff['index']
                all_raspberry_masks[j][dst_y_start:dst_y_end, dst_x_start:dst_x_end][source_region_mask] = False
                occlusion_counts[j] += 1
                
                # Record metadata
                occlusion_metadata['occlusion_history'].append({
                    'step': i,
                    'source_raspberry': source_idx,
                    'target_raspberry': j,
                    'occlusion_amount_pixels': int(aff['overlap_pixels']),
                    'full_percentage': float(aff['full_percentage']),
                    'remaining_size': int(aff['remaining_size'])
                })
            
            # Add new source raspberry to tracking
            # Create its mask at the pasted location
            new_raspberry_mask = np.zeros(composite_img.shape[:2], dtype=bool)
            new_raspberry_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = source_region_mask
            all_raspberry_masks.append(new_raspberry_mask)
            all_raspberry_imgs.append(source_img)
            ys, xs = np.where(new_raspberry_mask)
            if len(xs) > 0:
                center_x = int(xs.mean())
                center_y = int(ys.mean())
            else:
                center_x = (dst_x_start + dst_x_end) // 2
                center_y = (dst_y_start + dst_y_end) // 2

            all_raspberry_centers.append((center_x, center_y))
            
            print(f"Successfully added raspberry {i+1}")
        
        # 7. Calculate final statistics
        for j in range(min(len(all_raspberry_masks), k)):
            mask = all_raspberry_masks[j]
            visible_size = np.sum(mask)
            original_size = original_mask_sizes[j]
            occlusion_metadata['final_visible_sizes'].append({
                'raspberry_id': j,
                'original_size': int(original_size),
                'final_visible_size': int(visible_size),
                'times_occluded': int(occlusion_counts[j]),
                'full_percentage': float(visible_size / original_size * 100)
            })
        
        print("\n=== Final Summary ===")
        for info in occlusion_metadata['final_visible_sizes']:
            print(f"Raspberry #{info['raspberry_id']+1}: {info['final_visible_size']}/{info['original_size']} pixels "
                f"({info['full_percentage']:.1f}% remaining, {info['times_occluded']} occlusion events)")


        if visualize_bool:
            import matplotlib.pyplot as plt
            
            num_masks = min(len(all_raspberry_masks), k)
            total_plots = num_masks + 1  # +1 for composite
            
            cols = min(4, total_plots)
            rows = int(np.ceil(total_plots / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)
            
            # Plot composite image
            axes[0].imshow(np.asarray(composite_img, dtype=np.uint8))
            axes[0].set_title("Composite Image")
            axes[0].axis("off")
            
            # Plot each raspberry mask with original overlay
            for i in range(num_masks):
                current_mask = all_raspberry_masks[i]
                current_center = all_raspberry_centers[i]
                original_mask = self.masks[chosen_indices[i]]
                
                # Resize original if needed (in case transforms occurred)
                if original_mask.shape != current_mask.shape:
                    original_mask = cv2.resize(
                        original_mask.astype(np.uint8),
                        (current_mask.shape[1], current_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                # Find the center point of the original mask, calculate offset to center of pasted mask and align it,
                # i.e. move original mask center to the center of the pasted mask
                # Find original mask center
                orig_coords = np.argwhere(original_mask)
                if len(orig_coords) > 0:
                    orig_center_y, orig_center_x = orig_coords.mean(axis=0).astype(int)
                else:
                    orig_center_x, orig_center_y = current_center

                # Target center from pasted mask
                target_center_x, target_center_y = current_center

                # Compute shift
                shift_x = target_center_x - orig_center_x
                shift_y = target_center_y - orig_center_y

                # Create shifted original mask canvas
                shifted_original = np.zeros_like(current_mask)

                # Compute safe paste bounds
                h, w = current_mask.shape
                src_y_start = max(0, -shift_y)
                src_x_start = max(0, -shift_x)
                src_y_end = min(h, h - shift_y)
                src_x_end = min(w, w - shift_x)

                dst_y_start = max(0, shift_y)
                dst_x_start = max(0, shift_x)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)

                # Apply shift
                shifted_original[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    original_mask[src_y_start:src_y_end, src_x_start:src_x_end]

                
                
                # Build RGB visualization
                vis = np.zeros((*current_mask.shape, 3), dtype=np.uint8)
                
                # Original mask = RED
                vis[..., 0] = shifted_original.astype(np.uint8) * 255
                
                # Current visible mask = YELLOW
                vis[..., 1] = current_mask.astype(np.uint8) * 255
                
                axes[i + 1].imshow(vis)
                axes[i + 1].set_title(f"Raspberry #{i+1}\nRed=OG Mask, Yellow= New Mask")
                axes[i + 1].axis("off")
            
            # Hide unused axes
            for ax in axes[total_plots:]:
                ax.axis("off")
            
            plt.tight_layout()
            plt.show()
 
    
        
        return composite_img, all_raspberry_masks



    def single_raspberry_occlusion(self, abs_size_threshold=100, 
                                wanted_size_range=None,
                                realism_bool=False, 
                                randomize_scale_bool=(False, 0.2, 2.0), 
                                randomize_rotation_bool=(False, -30, 30),
                                choose_idx = None):
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
            choose_idx (list) : If provided, list of two indices to select specific images from the dataset.
                                    This is for when we want to take two masks of raspberries that are 
                                    nearly non-occluded such that we can produce more realistic occlusions.

        Workflow : 
            1. Select two random images of single raspberries from the dataset.
            2. Paste the source raspberry region onto the target image at a random location within
            the locality of the raspberry in the target image (Targeted Pasting).
            3. Optionally apply blending to make the occlusion look more natural.
        """

        if choose_idx is None:
            # 1. Select two random images and their respective masks
            chosen_idx = np.random.choice(len(self.images), size=2, replace=True)
        else:
            chosen_idx = choose_idx

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
            visualize_bool=True,
            reassign_source_target_bool=True
        )

        # 3. Optionally apply blending for realism
    # if realism_bool:
    #     new_target_img = self._apply_blending(new_target_img, source_img, new_target_mask)

        return new_target_img, new_target_mask

    def _paste(self, target_mask, 
            source_mask, 
            source_img,
            target_img, 
            abs_size_threshold = 2000,
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
            abs_size_threshold (int) : Minimum absolute size (in pixels) of the new segmentation mask.
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

        Returns :
            new_target_img (np.array) : Updated target image with source pasted.
            new_target_mask (np.array) : Updated binary mask after occlusion.
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
        
    
        # Determine acceptance criteria
        if wanted_size_range is not None:
            min_size_factor, max_size_factor = wanted_size_range
            def is_valid(remaining_size):
                return min_size_factor * original_target_size <= remaining_size <= max_size_factor * original_target_size
        else:
            def is_valid(remaining_size):
                return remaining_size >= abs_size_threshold

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
            
            else:
                # Since we mainly want occluded objects that are still rather large,
                # instead of randomly selecting new anchors, we nudge the target anchor
                # towards a random edge of the target mask to increase chance of valid occlusion
                # , meaning that the remaining size of the target mask stays large enough.
                pass 
        
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


## TOMORROW : implement the spacing idea for k_raspberries_occlusion, i.e.
    # Define a field in which raspberries can be placed (size of a bonnet)
    # Allow for non-occlusions aswell (i.e. random placement within the field)
    # For occlusions, do the typical restraints as before





def main():
    synthetic_occlusion = SyntheticOcclusion(data_path='dataset_single_objects/GT/processed/img001/processed_img001_data.npz')
   # new_img, new_mask = synthetic_occlusion.single_raspberry_occlusion(
   #     abs_size_threshold=50,
   #     wanted_size_range=(0.3, 0.7),
   #     realism_bool=False,
   #     randomize_scale_bool=(False, 1.2, 1.5),
   #     randomize_rotation_bool=(False, -180,180),
   #     #choose_idx=[3, 6]   
   # )
   
    new_img, new_masks = synthetic_occlusion.k_raspberries_occlusion(
        k=5,
        max_occlusions_per_raspberry=4,
        min_remaining_size_factor=0.1,
        wanted_size_range_per_occlusion=(0.7, 0.9),
        randomize_scale_bool=(False, 0.8, 1.2),
        randomize_rotation_bool=(False, -180, 180),
        visualize_bool=True
    )

 

if __name__ == "__main__":
    main()
