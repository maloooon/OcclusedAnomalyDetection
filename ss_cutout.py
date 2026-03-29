"""
Cutout self-supervised fine-tuning of one of the PyTorch CNN backbones.
Currently implemented for MobileNetV2 due to its small param size
(i.e. we only have little amount of training data, so we want to minimize overfitting risk).

"""


import torch
import torch.nn as nn
import torchvision
import random
from tqdm import tqdm
from time import time


class LightweightDecoder(nn.Module):
    """
    Takes feature maps from features.10 (96 channels, 14x14)
    and reconstructs 224x224 RGB image.
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # 14x14 -> 28x28
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 28x28 -> 56x56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 56x56 -> 112x112
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 112x112 -> 224x224
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
          #  nn.Sigmoid(),  # use if ensured that pixel values in [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)


class CutoutReconstructionModel(nn.Module):
    def __init__(self, device, unfreeze_from=7):
        """
        Encoder : some pre-trained torch backbone (currently MobileNetV2)
        Decoder : lightweight decoder to reconstruct cutout parts of the image
        unfreeze_from: which encoder layer to start unfreezing from (0-10)

        """
        super().__init__()
        backbone = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1")
        # Encoder: features.0 through features.10
        # We do not need the later layers (i.e. layers after layer 10)
        self.encoder = backbone.features[:11]

        # Freeze early layers
        for i in range(unfreeze_from):
            for param in self.encoder[i].parameters():
                param.requires_grad = False

        self.decoder = LightweightDecoder()
        self.to(device)

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction


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


def train_cutout_reconstruction(model, train_loader, device, epochs=30, lr=1e-4, optimizer = None, criterion = None):
    """
    Train loop. Loss computed only on the cutout regions within
    the raspberry mask — forces the model to learn berry texture,
    not background reconstruction.
    """

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, masks in pbar:
            # images: (B,3,H,W) 
            # masks: (B,1,H,W) binary raspberry mask
            images, masks = images.to(device), masks.to(device)
    
            masked_images, holes = apply_cutout(images, masks)

            reconstruction = model(masked_images)



      
            # Loss only on the cutout holes
            pixel_loss = criterion(reconstruction, images)  # (B, 3, H, W)
            # Average over channels, mask to holes only
            pixel_loss = pixel_loss.mean(dim=1, keepdim=True)  # (B, 1, H, W)
            hole_pixels = holes.sum()

            if hole_pixels > 0:
                loss = (pixel_loss * holes).sum() / hole_pixels
            else:
                loss = pixel_loss.mean()



       
            optimizer.zero_grad()
            loss.backward()
      
            optimizer.step()


            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.6f}")

    return model