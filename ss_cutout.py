"""
Cutout self-supervised fine-tuning of PyTorch CNN/ViT backbones.

Three modes are supported:
  - "mobilenet_v2"       : original implementation (small, low overfitting risk)
  - "wide_resnet50_2"    : WRN-50-2 — same CNN inductive bias as MobileNetV2, trivially extended
  - "dinov2_vit{s,b,l,g}14[_reg]" : DINOv2 ViT — see IMPORTANT caveats below

----------------------------------------------------------------------
WHY WRN-50-2 IS STRAIGHTFORWARD:
  WRN-50-2 is a CNN with local receptive fields. Cutout forces the encoder
  to reconstruct zeroed-out pixels from their local neighborhood, creating
  pressure to learn discriminative local textures. The architecture is
  identical in spirit to MobileNetV2 — we just extract from `layer3`
  (1024 ch, 14x14 for 224x224 input) instead of `features[:11]`
  (64 ch, 14x14). The decoder design follows directly.

WHY DINOv2 FINE-TUNING IS QUESTIONABLE:
  1. ViT attention is GLOBAL: every patch token attends to every other
     patch. When cutout zeroes out a region, the remaining tokens can
     reconstruct it using global context — removing the local-texture
     learning pressure that makes cutout effective for CNNs.
  2. DINOv2 is already self-supervised with DINO training on a very large
     dataset. Its features are well-generalised and work out-of-the-box
     for anomaly detection. Fine-tuning can cause catastrophic forgetting.
  3. DINOv2 (~86M params) is much larger than MobileNetV2 (~3M). With
     limited training data this creates serious overfitting risk, even
     when unfreezing only 1-2 transformer blocks.

  USE DINOv2 FROZEN (as already done in custom_feature_extractor_trimmed.py)
  unless you have strong evidence of a domain gap AND enough data.
  If you do fine-tune: use a very low LR (<1e-5), unfreeze ≤2 blocks,
  and monitor AUROC closely on a held-out set.
----------------------------------------------------------------------
"""

import math

import torch
import torch.nn as nn
import torchvision
import random
from tqdm import tqdm
from time import time


# ---------------------------------------------------------------------------
# LoRA (Low-Rank Adaptation) — used by DINOv2CutoutReconstructionModel
# ---------------------------------------------------------------------------

# Maps the short target name (as passed by the user) to the parent submodule
# attribute on a DINOv2 transformer block. Only these four are injectable.
_LORA_TARGET_PARENT = {
    'qkv':  'attn',   # fused Q+K+V projection  (dim → 3*dim)
    'proj': 'attn',   # attention output projection (dim → dim)
    'fc1':  'mlp',    # MLP first linear          (dim → 4*dim)
    'fc2':  'mlp',    # MLP second linear         (4*dim → dim)
}


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a Low-Rank Adaptation (LoRA).

    The forward pass computes:
        out = base_layer(x) + (x @ A.T @ B.T) * (alpha / rank)

    Where:
        A  ∈  R^{rank × in_features}   — Kaiming-uniform initialised
        B  ∈  R^{out_features × rank}  — zero-initialised

    B is zero at init so the adapter starts as a no-op and training
    begins from the pre-trained base output. Only A and B are trainable.

    Why rank matters:
        rank=4  →  ~37 K new params per DINOv2-ViT-B block (vs ~14 M for
                   unfreezing the full block) — ~380x more parameter-efficient.
        rank=8  →  ~74 K; rank=16 → ~147 K.
        Use higher rank if you see underfitting, lower if overfitting.

    scaling = alpha / rank (default alpha = rank → scaling = 1).
        Increasing alpha makes the adapter update larger relative to the
        frozen weights. alpha = 2*rank is a common choice for larger rank.
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = None):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        # alpha controls the effective learning rate of the adapter.
        # Defaulting to rank gives scaling = 1 (standard LoRA paper setting).
        self.alpha = float(rank) if alpha is None else float(alpha)
        self.scaling = self.alpha / rank

        in_f  = base_layer.in_features
        out_f = base_layer.out_features

        # A: Kaiming uniform (same as nn.Linear default weight init)
        # B: zeros so the initial delta = B @ A = 0 (no perturbation at step 0)
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Defensively freeze base weights in case this class is used standalone.
        # (In DINOv2CutoutReconstructionModel they are already frozen globally.)
        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base output (frozen) + low-rank delta (trainable)
        return self.base_layer(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

    def merged_weight(self) -> torch.Tensor:
        """W_merged = W_base + scaling * (B @ A).  Used at save time."""
        with torch.no_grad():
            return self.base_layer.weight + self.scaling * (self.lora_B @ self.lora_A)


# ---------------------------------------------------------------------------
# MobileNetV2 — original implementation, unchanged
# ---------------------------------------------------------------------------

class LightweightDecoder(nn.Module):
    """
    Takes feature maps from MobileNetV2 features.10 (64 channels, 14x14)
    and reconstructs a 224x224 RGB image via four 2x transposed convolutions.

    NOTE: the original docstring said "96 channels" — the actual output of
    features[10] is 64 channels; the comment was incorrect.
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
        Encoder : MobileNetV2 features[0..10] (14x14 feature map, 64 ch)
        Decoder : LightweightDecoder  →  224x224 RGB reconstruction
        unfreeze_from: freeze layers [0, unfreeze_from), unfreeze the rest (0-10)
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

    def save_encoder_weights(self, save_path):
        """
        Save fine-tuned encoder weights in torchvision mobilenet_v2 key format
        so that custom_feature_extractor_trimmed.py can load them with strict=False.

        self.encoder is backbone.features[:11], a Sequential whose state dict has
        keys like '0.conv.0.0.weight'. Adding the 'features.' prefix gives
        'features.0.conv.0.0.weight', which matches the full mobilenet_v2 state dict.
        When loaded with strict=False, only the matching keys (features.0–10) are
        updated; the rest of the model keeps its ImageNet weights.
        """
        encoder_state = {f'features.{k}': v for k, v in self.encoder.state_dict().items()}
        torch.save(encoder_state, save_path)


# ---------------------------------------------------------------------------
# WRN-50-2  (wide_resnet50_2) — new, analogous to MobileNetV2
# ---------------------------------------------------------------------------

class WRN50Decoder(nn.Module):
    """
    Takes feature maps from wide_resnet50_2 layer3 (1024 channels, 14x14 for
    a 224x224 input) and reconstructs a 224x224 RGB image.

    Same four-step 2x transposed-conv design as LightweightDecoder, but the
    first layer must consume 1024 channels instead of 64.
    Channel schedule mirrors a typical encoder-decoder funnel:
      1024 → 256 → 128 → 64 → 3 (RGB)
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
    def __init__(self, device, unfreeze_from=3):
        """
        Encoder: wide_resnet50_2 up to and including layer3
        Decoder: WRN50Decoder  →  224x224 RGB reconstruction

        WRN-50-2 has four named stages after the stem:
          stage 0 : conv1 + bn1 + relu + maxpool  →  (B, 64,   56, 56)
          stage 1 : layer1  (3 bottleneck blocks)  →  (B, 256,  56, 56)
          stage 2 : layer2  (4 bottleneck blocks)  →  (B, 512,  28, 28)
          stage 3 : layer3  (6 bottleneck blocks)  →  (B, 1024, 14, 14)  ← encoder output

        We stop at layer3 (same 14x14 spatial resolution as MobileNetV2 features[:11])
        and skip layer4 (7x7, too coarse) and the classification head.

        unfreeze_from: freeze stages [0, unfreeze_from), leave the rest trainable.
          Default = 3: only layer3 (stage 3) is fine-tuned — the stage that
          produces the features actually used by the anomaly detector.
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

        # Freeze stages [0, unfreeze_from) — identical logic to MobileNetV2
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


# ---------------------------------------------------------------------------
# DINOv2 — new, with important caveats (see module docstring)
# ---------------------------------------------------------------------------

# Embedding dimension (patch token dimension D) for each DINOv2 variant.
# Used to size the decoder's first layer correctly.
DINOV2_EMBED_DIMS = {
    "dinov2_vits14":     384,
    "dinov2_vitb14":     768,
    "dinov2_vitl14":    1024,
    "dinov2_vitg14":    1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg":1024,
    "dinov2_vitg14_reg":1536,
}


class DINOv2Decoder(nn.Module):
    """
    Takes DINOv2 patch tokens reshaped to a spatial grid (B, D, h, w) where
      h = w = img_size // 14  (e.g. 224 // 14 = 16 for standard 224x224 input)
    and reconstructs the original 224x224 RGB image.

    The spatial resolution 16 → 224 is NOT a clean power of 2, so we use
    three 2x ConvTranspose2d steps (16→32→64→128) and then a bilinear
    upsample to 224, followed by a Conv2d to produce RGB.

    Channel schedule: D → 256 → 128 → 64 → (bilinear upsample) → 3 (RGB)
    The first layer collapses the potentially large D (e.g. 768 for vitb14)
    down to 256, keeping decoder parameter count manageable.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_decoder = nn.Sequential(
            # 16x16 → 32x32   (in_channels = D, e.g. 384 / 768 / 1024 / 1536)
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 128x128 → 224x224: not a power-of-2 step, so use bilinear upsample
        # followed by a 3x3 conv to blend the interpolated values into RGB.
        self.final_upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_decoder(x)
        x = self.final_upsample(x)
        return self.final_conv(x)


class DINOv2CutoutReconstructionModel(nn.Module):
    def __init__(self, device, model_name: str = "dinov2_vitb14",
                 unfreeze_last_n_blocks: int = 2,
                 lora_rank: int = 0,
                 lora_alpha: float = None,
                 lora_targets: tuple = ('qkv', 'proj')):
        """
        Encoder: DINOv2 ViT backbone (most blocks frozen, last N adapted)
        Decoder: DINOv2Decoder  →  224x224 RGB reconstruction

        Two fine-tuning modes, selected by lora_rank:

        lora_rank == 0  (default, block-unfreeze mode):
            The last `unfreeze_last_n_blocks` transformer blocks are fully
            unfrozen. High parameter count (~7 M/block for ViT-B), high
            overfitting risk with limited data.

        lora_rank > 0  (LoRA mode, recommended):
            ALL DINOv2 weights stay frozen. Low-rank adapter matrices
            (A ∈ R^{rank×in}, B ∈ R^{out×rank}) are injected into the
            attention linears of the last `unfreeze_last_n_blocks` blocks.
            For ViT-B with rank=4, lora_targets=('qkv','proj'), 2 blocks:
              ~37 K trainable params  vs  ~14 M for full block unfreeze.

        lora_rank (int):
            Rank r of the low-rank matrices. Typical values: 4, 8, 16.
            Higher rank → more capacity, more overfitting risk.

        lora_alpha (float | None):
            LoRA scaling: delta = (B @ A) * (alpha / rank).
            None defaults to lora_rank → scaling = 1 (standard setting).
            Use alpha = 2*rank for a stronger adapter signal.

        lora_targets (tuple of str):
            Which linear layers inside each block to adapt.
            Supported: 'qkv' (attn), 'proj' (attn), 'fc1' (mlp), 'fc2' (mlp).
            Default ('qkv', 'proj') — attention-only, standard LoRA-for-ViT.

        READ THE MODULE DOCSTRING before using this mode.
        """
        super().__init__()

        if model_name not in DINOV2_EMBED_DIMS:
            raise ValueError(
                f"Unknown DINOv2 model: {model_name}. "
                f"Supported: {list(DINOV2_EMBED_DIMS.keys())}"
            )

        unknown = set(lora_targets) - set(_LORA_TARGET_PARENT)
        if unknown:
            raise ValueError(f"Unknown lora_targets: {unknown}. "
                             f"Supported: {list(_LORA_TARGET_PARENT)}")

        self.patch_size = 14  # DINOv2 always uses 14x14 pixel patches
        self.model_name = model_name
        self.use_lora = lora_rank > 0

        # Load pretrained DINOv2 from torch.hub (same as CustomFeatureExtractor)
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True)

        # Read embed_dim from the model itself to avoid dict/model mismatches
        self.embed_dim = self.dino.embed_dim  # e.g. 384 (vits14) or 768 (vitb14)

        # Freeze all DINOv2 weights first — both modes start from a frozen base
        for param in self.dino.parameters():
            param.requires_grad = False

        if self.use_lora:
            # LoRA mode: inject low-rank adapters into the last N blocks.
            # Everything else stays frozen. The adapters' lora_A / lora_B are
            # nn.Parameter objects created inside LoRALinear with requires_grad=True.
            for block in self.dino.blocks[-unfreeze_last_n_blocks:]:
                for target in lora_targets:
                    parent_name = _LORA_TARGET_PARENT[target]  # 'attn' or 'mlp'
                    parent_module = getattr(block, parent_name)
                    base_linear = getattr(parent_module, target)  # the frozen nn.Linear
                    # Replace the frozen nn.Linear with a LoRALinear wrapper.
                    # Only lora_A and lora_B inside the wrapper are trainable.
                    setattr(parent_module, target,
                            LoRALinear(base_linear, rank=lora_rank, alpha=lora_alpha))
        else:
            # Full-block unfreeze mode (original behaviour).
            # Unfreeze from the tail so low-level features are preserved.
            if unfreeze_last_n_blocks > 0:
                for block in self.dino.blocks[-unfreeze_last_n_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
                # Unfreeze the final LayerNorm — it normalises the unfrozen
                # blocks' outputs and must be trained with them.
                for param in self.dino.norm.parameters():
                    param.requires_grad = True

        self.decoder = DINOv2Decoder(self.embed_dim)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Spatial grid dimensions after patch tokenisation
        h = H // self.patch_size  # e.g. 224 // 14 = 16
        w = W // self.patch_size

        # forward_features runs the full ViT forward pass and returns a dict.
        # 'x_norm_patchtokens' contains the patch tokens AFTER the final LayerNorm,
        # with CLS token and register tokens (if any) already removed.
        # Shape: (B, h*w, D) = (B, 256, embed_dim) for 224x224 input.
        output = self.dino.forward_features(x)
        patch_tokens = output['x_norm_patchtokens']  # (B, N_patches, D)

        # Reshape from token sequence to 2D spatial grid — identical to the
        # reshape in CustomFeatureExtractor.__call__ for DINOv2:
        #   feat.permute(0,2,1).reshape(B, D, h, w)
        features = patch_tokens.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)

        return self.decoder(features)

    def save_encoder_weights(self, save_path):
        """
        Save fine-tuned DINOv2 backbone weights.

        Block-unfreeze mode (use_lora=False):
            Saves self.dino.state_dict() directly — a standard DINOv2 state dict.

        LoRA mode (use_lora=True):
            Merges each adapter's delta into the base weight before saving:
                W* = W_base + scaling * (B @ A)
            The result is a standard DINOv2 state dict (no LoRALinear wrappers),
            loadable into a plain DINOv2 model with strict=False. The merge makes
            the LoRA adaptation zero-overhead at inference time.

        To load these weights in custom_feature_extractor_trimmed.py, add after
        the torch.hub.load call for DINOv2 backbones:

            if custom_weights_path is not None:
                self.model.load_state_dict(
                    torch.load(custom_weights_path, map_location='cpu'), strict=False
                )
        """
        if not self.use_lora:
            torch.save(self.dino.state_dict(), save_path)
            return

        # Build a map of all LoRALinear module paths so we can recognise their keys
        lora_modules = {
            name: mod for name, mod in self.dino.named_modules()
            if isinstance(mod, LoRALinear)
        }

        merged_sd = {}
        for key, value in self.dino.state_dict().items():
            handled = False
            for lora_path, lora_mod in lora_modules.items():
                if key == f'{lora_path}.lora_A' or key == f'{lora_path}.lora_B':
                    # Adapter matrices — absorbed into the merged weight, skip here
                    handled = True
                    break
                elif key == f'{lora_path}.base_layer.weight':
                    # Compute W* = W_base + scaling * (B @ A) and remap the key
                    # from 'blocks.N.attn.qkv.base_layer.weight'
                    #   to 'blocks.N.attn.qkv.weight' (standard DINOv2 naming)
                    merged_sd[f'{lora_path}.weight'] = lora_mod.merged_weight()
                    handled = True
                    break
                elif key == f'{lora_path}.base_layer.bias':
                    # Bias is unchanged; remap key from '.base_layer.bias' → '.bias'
                    merged_sd[f'{lora_path}.bias'] = value
                    handled = True
                    break
            if not handled:
                # Regular (non-LoRA) parameter — copy as-is
                merged_sd[key] = value

        torch.save(merged_sd, save_path)


# ---------------------------------------------------------------------------
# Factory function — selects the right model class from a backbone name string
# ---------------------------------------------------------------------------

def create_cutout_model(backbone_name: str, device, **kwargs):
    """
    Instantiate the correct CutoutReconstructionModel for `backbone_name`.

    backbone_name options:
      "mobilenet_v2"            → CutoutReconstructionModel(device, unfreeze_from=7)
      "wide_resnet50_2"         → WRN50CutoutReconstructionModel(device, unfreeze_from=3)
      "dinov2_vit{s,b,l,g}14"  → DINOv2CutoutReconstructionModel(device, model_name=...,
                                      unfreeze_last_n_blocks=2)

    Extra **kwargs are forwarded to the chosen constructor, allowing you to
    override defaults (e.g. unfreeze_from, unfreeze_last_n_blocks).

    Examples:
        model = create_cutout_model("mobilenet_v2", device)
        model = create_cutout_model("wide_resnet50_2", device, unfreeze_from=2)
        model = create_cutout_model("dinov2_vitb14", device, unfreeze_last_n_blocks=2)
        model = create_cutout_model("dinov2_vitb14", device, lora_rank=4)
        model = create_cutout_model("dinov2_vitb14", device, lora_rank=8, lora_alpha=16,
                                    lora_targets=('qkv', 'proj', 'fc1', 'fc2'))
    """
    if backbone_name == "mobilenet_v2":
        return CutoutReconstructionModel(device, **kwargs)
    elif backbone_name == "wide_resnet50_2":
        return WRN50CutoutReconstructionModel(device, **kwargs)
    elif backbone_name in DINOV2_EMBED_DIMS:
        # Pass model_name explicitly since DINOv2CutoutReconstructionModel
        # needs to know which variant to load from torch.hub
        return DINOv2CutoutReconstructionModel(device, model_name=backbone_name, **kwargs)
    else:
        raise ValueError(
            f"Backbone '{backbone_name}' is not supported for cutout fine-tuning. "
            f"Supported: 'mobilenet_v2', 'wide_resnet50_2', or any key in DINOV2_EMBED_DIMS."
        )


# ---------------------------------------------------------------------------
# apply_cutout — unchanged, backbone-agnostic (pixel-level operation)
# ---------------------------------------------------------------------------

def apply_cutout(images, mask, n_holes=3, hole_size_range=(32, 64)):
    """
    Apply random rectangular cutouts ONLY within the raspberry mask region.
    Returns the masked images and a binary tensor indicating which pixels
    were cut out (for computing loss only on the masked region).

    This function operates at the pixel level and is backbone-agnostic:
    it works identically for CNN (MobileNetV2 / WRN-50-2) and ViT (DINOv2).
    For DINOv2, the zeroed-out pixel regions naturally correspond to zeroed
    patch embeddings, since the patch projection is a linear operation.

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


# ---------------------------------------------------------------------------
# train_cutout_reconstruction — unchanged, works for all three model types
# since all of them implement forward(x) -> (B, 3, H, W)
# ---------------------------------------------------------------------------

def train_cutout_reconstruction(model, train_loader, device, epochs=30, lr=1e-4,
                                optimizer=None, criterion=None,
                                n_holes=3, hole_size_range=(32, 64)):
    """
    Train loop. Loss computed only on the cutout regions within
    the raspberry mask — forces the model to learn berry texture,
    not background reconstruction.

    This function is backbone-agnostic: all three model classes
    (CutoutReconstructionModel, WRN50CutoutReconstructionModel,
    DINOv2CutoutReconstructionModel) expose the same forward(x) → (B,3,H,W)
    interface, so no changes are needed here.

    n_holes, hole_size_range: forwarded to apply_cutout, same semantics.
    """

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
                loss = (pixel_loss * holes).sum() / hole_pixels
            else:
                loss = pixel_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.6f}")

    return model
