
# self-supervised pretraining approach

import math

import torch
import torch.nn as nn
import torchvision
import random
from tqdm import tqdm
import copy
from time import time


class WRN50Decoder(nn.Module):
    """
    Takes feature maps from wide_resnet50_2 layer3 (1024 channels, 14x14 for
    a 224x224 input) and reconstructs a 224x224 RGB image.

    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # 14x14 → 28x28   (1024 ch from layer3)
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 28x28 → 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 56x56 → 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 112x112 → 224x224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)

class WRN50CutoutReconstructionModel(nn.Module):
    def __init__(self, device, unfreeze_from=2):
        """
        Encoder: wide_resnet50_2 up to and including layer3
        Decoder: WRN50Decoder  →  224x224 RGB reconstruction

        WRN-50-2 has four named stages after the stem:
          stage 0 : conv1 + bn1 + relu + maxpool  →  (B, 64,   56, 56)
          stage 1 : layer1  (3 bottleneck blocks)  →  (B, 256,  56, 56)
          stage 2 : layer2  (4 bottleneck blocks)  →  (B, 512,  28, 28)
          stage 3 : layer3  (6 bottleneck blocks)  →  (B, 1024, 14, 14)  ← encoder output

        We stop at layer3 and skip layer4 and the classification head (less params for less risk of overfitting drove this descision mainly)

        unfreeze_from: freeze stages [0, unfreeze_from), leave the rest trainable.
        """
        super().__init__()
        backbone = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")

        # Bundle the four stages into a ModuleList so we can index them the
        # same way as MobileNetV2's encoder (encoder[i] → freeze/unfreeze).
        # We wrap the stem (conv1+bn1+relu+maxpool) in a Sequential because
        # WRN-50-2 exposes them as separate attributes, not a single module.
        self.encoder = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool),  # stage 0
            backbone.layer1,  # stage 1
            backbone.layer2,  # stage 2
            backbone.layer3,  # stage 3
        ])

        # Freeze stages [0, unfreeze_from)
        for i in range(unfreeze_from):
            for param in self.encoder[i].parameters():
                param.requires_grad = False

        self.decoder = WRN50Decoder()
        self.to(device)

    def forward(self, x):
        # Sequential forward through the four encoder stages
        for stage in self.encoder:
            x = stage(x)
        # x is now (B, 1024, 14, 14) — pass to decoder
        return self.decoder(x)

    def save_encoder_weights(self, save_path):
        """
        Save fine-tuned encoder weights in torchvision wide_resnet50_2 key format
        so that custom_feature_extractor_trimmed.py can load them with strict=False.

        encoder[0] is Sequential(conv1, bn1, relu, maxpool), whose state dict
        has integer-prefixed keys ('0.weight' for conv1, '1.*' for bn1). We strip
        the integer prefix and replace it with the torchvision attribute name.

        encoder[1/2/3] are layer1/layer2/layer3 directly, so we just prepend the
        layer name (e.g. 'layer1.0.conv1.weight').
        """
        sd = {}

        # Stem: encoder[0] = Sequential(conv1, bn1, relu, maxpool)
        # Keys '0.*' → conv1, keys '1.*' → bn1 (relu/maxpool have no params)
        for k, v in self.encoder[0].state_dict().items():
            if k.startswith('0.'):
                sd[f'conv1.{k[2:]}'] = v    # e.g. '0.weight' → 'conv1.weight'
            elif k.startswith('1.'):
                sd[f'bn1.{k[2:]}'] = v      # e.g. '1.running_mean' → 'bn1.running_mean'

        # Residual stages: prepend the torchvision attribute name as namespace
        for layer_name, stage_idx in [('layer1', 1), ('layer2', 2), ('layer3', 3)]:
            for k, v in self.encoder[stage_idx].state_dict().items():
                sd[f'{layer_name}.{k}'] = v  # e.g. 'layer1.0.conv1.weight'

        torch.save(sd, save_path)

def create_cutout_model(backbone_name: str, device, **kwargs):
    """
    Instantiate CutoutReconstructionModel for backbone model.

    backbone_name options: "wide_resnet50_2"       

    """
    if backbone_name == "wide_resnet50_2":
        return WRN50CutoutReconstructionModel(device, **kwargs)
    else:
        raise ValueError(
            f"Backbone '{backbone_name}' is not supported for cutout fine-tuning. "
            f"Supported: 'mobilenet_v2', 'wide_resnet50_2', or any key in DINOV2_EMBED_DIMS."
        )

def apply_cutout(images, mask, n_holes=3, hole_size_range=(32, 64)):
    """
    Apply random rectangular cutouts ONLY within the raspberry mask region.
    Returns the masked images and a binary tensor indicating which pixels
    were cut out (for computing loss only on the masked region).

    Args:
        images: (B, 3, H, W) tensor
        mask: (B, 1, H, W) binary mask of raspberry region
        n_holes: number of cutout rectangles
        hole_size_range: (min_size, max_size) for each rectangle edge
    """
    B, C, H, W = images.shape
    cutout_mask = torch.ones(B, 1, H, W, device=images.device)

    for b in range(B):
        for _ in range(n_holes):
            h = random.randint(hole_size_range[0], hole_size_range[1])
            w = random.randint(hole_size_range[0], hole_size_range[1])
            y = random.randint(0, H - h)
            x = random.randint(0, W - w)
            cutout_mask[b, :, y:y+h, x:x+w] = 0.0

    # Only cut out within the raspberry region
    cutout_mask = cutout_mask + (1.0 - mask)  # preserve background (i.e. make sure we only cut out where we have the raspberry mask)
    cutout_mask = cutout_mask.clamp(0, 1)

    masked_images = images * cutout_mask
    # The "holes" are where cutout_mask is 0 AND mask is 1
    holes = (1.0 - cutout_mask) * mask

    return masked_images, holes

def train_cutout_reconstruction(model, train_loader, device, epochs=30, lr=1e-4,
                                optimizer=None, criterion=None,
                                n_holes=3, hole_size_range=(32, 64),
                                val_loader=None, early_stopping_patience=10):
    """
    Loss computed only on the cutout regions within
    the raspberry mask to force the model to learn berry texture,
    not background reconstruction.

    """

    best_val_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks, _ in pbar:
            # images: (B,3,H,W)
            # masks: (B,1,H,W) binary raspberry mask
            images, masks = images.to(device), masks.to(device)

            masked_images, holes = apply_cutout(images, masks, n_holes=n_holes, hole_size_range=hole_size_range)

            reconstruction = model(masked_images)

            # Loss only on the cutout holes
            pixel_loss = criterion(reconstruction, images)  # (B, 3, H, W)
            # Average over channels, mask to holes only
            pixel_loss = pixel_loss.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            hole_pixels = holes.sum()

            if hole_pixels > 0:
                # Division by hole_pixels normalizes the loss to be independent of the number of cutout pixels
                loss = (pixel_loss * holes).sum() / hole_pixels
            else:
                loss = pixel_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks, _ in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    masked_images, holes = apply_cutout(images, masks, n_holes=n_holes, hole_size_range=hole_size_range)
                    reconstruction = model(masked_images)
                    pixel_loss = criterion(reconstruction, images).mean(dim=1, keepdim=True)
                    hole_pixels = holes.sum()
                    if hole_pixels > 0:
                        val_loss += (pixel_loss * holes).sum().item() / hole_pixels.item()
                    else:
                        val_loss += pixel_loss.mean().item()
            val_loss /= len(val_loader)

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
                  + (" *" if improved else f" (no improvement {epochs_without_improvement}/{early_stopping_patience})"))

            model.train()

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (patience {early_stopping_patience} reached)")
                break
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}")

    if val_loader is not None:
        print(f"Loading best checkpoint (val loss: {best_val_loss:.6f})")
        model.load_state_dict(best_state)

    return model
