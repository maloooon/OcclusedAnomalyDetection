import copy

import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import numpy as np

from moviad.models.components.rd4ad.resnet import resnet18
from moviad.models.components.rd4ad.deresnet import de_resnet18

class RD4AD(torch.nn.Module):

    DEFAULT_PARAMETERS = {
        "epochs": 200, 
        "batch_size": 16,
        "learning_rate": 0.001,
        "betas": (0.5,0.999),
    }

    def __init__(self, backbone_name, device, input_size = (224, 224)):
        super().__init__()

        self.backbone_name = backbone_name
        self.device = device
        self.input_size = input_size

        self.encoder, self.bn = resnet18(pretrained=True)
        self.decoder = de_resnet18(pretrained=False)

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
            batch, mask, _ , _ , _ = batch

        enc_batch = self.encoder(batch)
        bn_batch = self.bn(enc_batch)
        dec_batch = self.decoder(bn_batch)

        if self.training:
            return enc_batch, bn_batch, dec_batch
        else:
            return self.post_process(enc_batch, dec_batch)
        
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
