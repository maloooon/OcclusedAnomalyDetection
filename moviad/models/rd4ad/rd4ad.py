import copy

import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import numpy as np
import cv2 as cv



from moviad.models.components.rd4ad import resnet as _enc
from moviad.models.components.rd4ad import deresnet as _dec

BACKBONE_MAP = {
    "resnet18":         (_enc.resnet18,         _dec.de_resnet18),
    "resnet34":         (_enc.resnet34,         _dec.de_resnet34),
    "resnet50":         (_enc.resnet50,         _dec.de_resnet50),
    "resnet101":        (_enc.resnet101,        _dec.resnet101),
    "resnet152":        (_enc.resnet152,        _dec.resnet152),
    "wide_resnet50_2":  (_enc.wide_resnet50_2,  _dec.de_wide_resnet50_2),
    "wide_resnet101_2": (_enc.wide_resnet101_2, _dec.de_wide_resnet101_2),
}

class RD4AD(torch.nn.Module):

    DEFAULT_PARAMETERS = {
        "epochs": 200, 
        "batch_size": 16,
        "learning_rate": 0.001,
        "betas": (0.5,0.999),
    }

    def __init__(self, backbone_name, device, input_size = (224, 224), struct_core_instance = None, scoring_mode = 'MAXMEAN_1', filter_post = 'NONE', mask_border_filter_thickness = 0, AD_only_on_mask = False, skip_layer1: bool = False, custom_weights_path: str = None, mask_dilation_radius: int = 0):
        super().__init__()

        self.backbone_name = backbone_name
        self.device = device
        self.input_size = input_size
        self.struct_core_instance = struct_core_instance
        self.scoring_mode = scoring_mode
        self.filter_post = filter_post
        self.mask_border_filter_thickness = mask_border_filter_thickness
        self.AD_only_on_mask = AD_only_on_mask
        self.skip_layer1 = skip_layer1
        self.mask_dilation_radius = mask_dilation_radius

        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}' for RD4AD. "
                f"Supported backbones: {list(BACKBONE_MAP)}"
            )
        encoder_fn, decoder_fn = BACKBONE_MAP[backbone_name]
        print("USING BACKBONE:", backbone_name)
        self.encoder, self.bn = encoder_fn(
            pretrained=custom_weights_path is None,
           # custom_weights_path=custom_weights_path,
            skip_layer1=skip_layer1,
        )
        self.decoder = decoder_fn(pretrained=False, skip_layer1=skip_layer1)



    def to(self, device: torch.device):
        self.encoder.to(device)
        self.bn.to(device)
        self.decoder.to(device)

    def train(self, *args, **kwargs):
        self.encoder.eval()
        self.bn.train()
        self.decoder.train()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()
        return super().eval(*args, **kwargs)

    def forward(self, batch: torch.Tensor):
        """
        Output tensors
        List[torch.Tensor] of len (n_layers)
        every tensor shape is (B C H W)
        """

        if (isinstance(batch, tuple)):
            if len(batch) == 6:
                batch, mask, mask_unfiltered, batch_og, mask_og, depth_og = batch
            elif len(batch) == 2:
                batch, mask = batch
                batch_og, mask_og, depth_og, mask_unfiltered = None, None, None, None
            elif len(batch) == 3:
                batch, mask, mask_unfiltered = batch
                batch_og, mask_og, depth_og = None, None, None

        # Ensure batch has batch dimension
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)

        enc_batch = self.encoder(batch)
        # For DinoV2 also return cls token
       # if "dino" in self.backbone_name:
       #     enc_batch, cls_token = enc_batch
        bn_batch = self.bn(enc_batch)
        dec_batch = self.decoder(bn_batch)

        if self.training:
            return enc_batch, bn_batch, dec_batch
        else:
            return self.post_process(enc_batch, dec_batch, batch, batch_og, mask_og, depth_og, mask, border_thickness = self.mask_border_filter_thickness)
        
    '''
    def post_process(self, enc_batch, dec_batch) -> torch.Tensor:
        anomaly_map = None
        blur = GaussianBlur(1, sigma = 4)

        #iterate over the feature extraction layers batches
        for i in range(len(enc_batch)):
            fs = dec_batch[i]
            ft = enc_batch[i]

            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=self.input_size, mode='bilinear', align_corners=True)

            if anomaly_map is None:
                anomaly_map = a_map
            else:
                anomaly_map += a_map

        anomaly_map = blur(anomaly_map)

        flat = anomaly_map.view(anomaly_map.size(0), -1)
        max_score = flat.max(dim=1).values
        mean_score = flat.mean(dim=1)
        alpha = 0.8
        image_score = alpha * max_score + (1 - alpha) * mean_score
        return anomaly_map, image_score
      #  return anomaly_map, torch.max(anomaly_map.view(anomaly_map.size(0), -1), dim = 1)[0]
      '''


    
    def post_process(self, enc_batch, dec_batch, batch, batch_og, mask_og, depth_og, mask, border_thickness=5) -> torch.Tensor:
        anomaly_map = None
        blur = GaussianBlur(1, sigma = 4)

        #iterate over the feature extraction layers batches
        for i in range(len(enc_batch)):
            fs = dec_batch[i]
            ft = enc_batch[i]

            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=self.input_size, mode='bilinear', align_corners=True)

            if anomaly_map is None:
                anomaly_map = a_map
            else:
                anomaly_map += a_map

        anomaly_map = blur(anomaly_map)

        # --- Apply raspberry mask with contour-based border exclusion ---
        device = anomaly_map.device
        effective_mask = None
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            effective_mask = torch.zeros_like(mask, dtype=torch.float32)

            for i in range(mask.shape[0]):
                sample_mask = mask[i, 0].cpu().numpy().astype(np.uint8)

                if self.mask_dilation_radius > 0:
                    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * self.mask_dilation_radius + 1, 2 * self.mask_dilation_radius + 1))
                    sample_mask = cv.dilate(sample_mask, kernel)

                if border_thickness > 0:
                    contours, _ = cv.findContours(
                        sample_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                    )
                    contour_band = np.zeros_like(sample_mask)
                    cv.drawContours(
                        contour_band, contours, -1, 255, thickness=border_thickness
                    )
                    inner_mask = sample_mask.copy()
                    inner_mask[contour_band == 255] = 0
                else:
                    inner_mask = sample_mask

                effective_mask[i, 0] = torch.from_numpy(
                    inner_mask.astype(np.float32)
                ).to(device)

            effective_mask = (effective_mask > 0).float()
            if self.AD_only_on_mask:
                anomaly_map = anomaly_map * effective_mask

        # Filter holes before computing anomaly scores
        if 'HOLE_DARKNESS' in self.filter_post:
            thresh_depth, thresh_dark = self.filter_post.split('_')[2:4]
            anomaly_map = filter_holes_batched(
                anomaly_map, batch, batch_og, mask_og, depth_og,
                depth_threshold_percentile=int(thresh_depth),
                brightness_threshold_percentile=int(thresh_dark)
            )

        # Compute per-image anomaly score
        flat = anomaly_map.view(anomaly_map.size(0), -1)
        anomaly_scores = torch.max(flat, dim=1)[0]

        self._last_effective_mask = effective_mask if self.AD_only_on_mask else None
        if self.struct_core_instance is not None and self.scoring_mode == 'STRUCTCORE':
            anomaly_scores = self.struct_core_instance.score(anomaly_map, anomaly_scores, mask=self._last_effective_mask)
        else:
            k = float(self.scoring_mode.split('_')[-1])
            max_scores = anomaly_scores
            mean_scores = torch.mean(flat, dim=1)
            anomaly_scores = k * max_scores + (1 - k) * mean_scores

        return anomaly_map, anomaly_scores

    def __call__(self, batch):
        return self.forward(batch)


    def load_model(self, path):

        """
        Load the Model

        Parameters:
        ----------
            path: str
                The path to the saved model checkpoint
        """

        state_dict = torch.load(path)
