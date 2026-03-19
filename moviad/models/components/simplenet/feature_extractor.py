import torch.nn as nn
import torch
import torch.nn.functional as F

class UpscalingFeatureExtractor(nn.Module):
    """Feature extractor module.

    Args:
        backbone (str): backbone name.
        layers (list[str]): list of layers used for extraction.
    """

    def __init__(self, featurre_extractor, patch_size: int = 3) -> None:
        super().__init__()

        self.feature_extractor = featurre_extractor

        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor.

        Args:
            input_tensor: input tensor (images)

        Returns:
            (torch.Tensor): extracted feature map.
        """
        features = self.feature_extractor(input_tensor)

        _, _, h, w = features[0].shape
        feature_map = []
        for layer in features:
            # upscale all to 2x the size of the first (largest)
            resized = F.interpolate(
                layer,
                size=(h * 2, w * 2),
                mode="bilinear",
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        return self.pooler(feature_map)

    def get_channels_dim(self) -> int:
        return self.feature_extractor.get_channels_dim()