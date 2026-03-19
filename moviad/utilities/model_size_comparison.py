import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import torch
from torchinfo import summary
from torchvision.models import get_model
from torchvision.models.feature_extraction import create_feature_extractor

from utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor


stats_to_get = ["total_param_bytes", "total_output_bytes", "total_mult_adds"]
model_layers = {
    "wide_resnet50_2": [f"layer{i}" for i in range(1, 5)],
    "mobilenet_v2": [f"features.{i}" for i in range(1, 19)],
    "micronet-m1" : range(1,8),
    "mcunet-in3" : range(1,18),
    "phinet_1.2_0.5_6_downsampling" : range(1,10)
}
input_size = (1, 3, 224, 224)


def get_stat(stats, name):
    return getattr(stats, name)


def compare_to_original(original_stats, new_stats, percentage:bool):
    """percentage: if True, return the percentage difference, otherwise return the ratio"""
    stats_comparison = {}
    p = 100 if percentage else 1
    for stat in stats_to_get:
        orig_val = get_stat(original_stats, stat)
        new_val = get_stat(new_stats, stat)
        stats_comparison[stat] = p * new_val / orig_val
    total_size_bytes_new = new_stats.total_param_bytes + new_stats.total_output_bytes
    total_size_bytes_orig = (
        original_stats.total_param_bytes + original_stats.total_output_bytes
    )
    stats_comparison["total_size_bytes"] = p * total_size_bytes_new / total_size_bytes_orig
    return stats_comparison


def feat_extractor_size_comparison(model_name):
    layers = model_layers[model_name]

    #model = get_model(model_name)
    #feat_ex_orig = create_feature_extractor(model, return_nodes=[layers[-1]])

    model = CustomFeatureExtractor(model_name, layers, device = "cpu").model
    stats_orig = summary(model, input_size, device="cpu", verbose=0)

    comparison_list = []
    for layer in tqdm(layers, desc=f"Model {model_name}"):
        feat_ex_trimmed = CustomFeatureExtractor(model_name, [layer], device = "cpu")
        stats_new = summary(feat_ex_trimmed.model, input_size, device="cpu", verbose=0)
        stats_comparison = compare_to_original(stats_orig, stats_new, percentage=True)
        stats_comparison["layer"] = layer
        stats_comparison["output_shape"] = feat_ex_trimmed(torch.rand(1,3,224,224))[-1].shape
        comparison_list.append(stats_comparison)

    df = pd.DataFrame(comparison_list)
    return df.set_index("layer")


def main():
    pd.set_option("display.precision", 2)

    """

    comp = feat_extractor_size_comparison("wide_resnet50_2")
    print(comp, "\n")

    comp = feat_extractor_size_comparison("micronet-m1")
    print(comp)

    print()
    print("---------------------------------------------")
    print("---------------------------------------------")
    print()

    comp = feat_extractor_size_comparison("wide_resnet50_2")
    print(comp, "\n")

    comp = feat_extractor_size_comparison("mcunet-in3")
    print(comp)
    """

    comp = feat_extractor_size_comparison("mobilenet_v2")
    print(comp, "\n")

    comp = feat_extractor_size_comparison("mcunet-in3")
    print(comp)

    print()
    print("---------------------------------------------")
    print("---------------------------------------------")
    print()

    comp = feat_extractor_size_comparison("mobilenet_v2")
    print(comp, "\n")

    comp = feat_extractor_size_comparison("phinet_1.2_0.5_6_downsampling")
    print(comp)



if __name__ == "__main__":
    main()
