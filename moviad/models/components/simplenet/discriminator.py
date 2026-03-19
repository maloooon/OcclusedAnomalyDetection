import torch.nn as nn
import torch
from torch.nn import Parameter

def init_weights(module: nn.Module) -> None:
    """Init weight of the model.

    Args:
        module (nn.Module): torch module.
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module,nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)

class Discriminator(nn.Module):
    """SegmentationDetection module responsible for prediction of anomaly map and score.

    Args:
        channel_dim (int): channel dimension of features.
        stop_grad (bool): whether to stop gradient from class. head to seg. head.
    """

    def __init__(
        self,
        channel_dim: int,
        stop_grad: bool = False,
    ) -> None:
        super().__init__()
        self.stop_grad = stop_grad

        # 1x1 conv - linear layer equivalent
        self.seg_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=1024,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        # pooling for cls. conv out and map
        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # cls. head conv block
        self.cls_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_dim + 1,
                out_channels=128,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # cls. head fc block: 128 from dec and 2 from map, * 2 due to max and avg pool
        self.cls_fc = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self) -> tuple[list[Parameter], list[Parameter]]:
        """Get segmentation and classification head parameters.

        Returns:
            seg. head parameters and class. head parameters.
        """
        seg_params = list(self.seg_head.parameters())
        dec_params = list(self.cls_conv.parameters()) + list(self.cls_fc.parameters())
        return seg_params, dec_params

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly map and anomaly score.

        Args:
            features: adapted features.

        Returns:
            predicted anomaly map and score.
        """
        # get anomaly map from seg head
        ano_map = self.seg_head(features)

        map_dec_copy = ano_map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((features, map_dec_copy), dim=1)
        dec_out = self.cls_conv(mask_cat)

        # conv block result pooling
        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        # predicted map pooling (and stop grad)
        map_max = self.map_max_pool(ano_map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(ano_map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze()
        ano_score = self.cls_fc(dec_cat).squeeze()

        return ano_map, ano_score