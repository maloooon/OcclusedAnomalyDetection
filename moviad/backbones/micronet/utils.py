import numpy as np
import torch
from typing import Dict, Union
import torch.nn as nn
from PIL.Image import Image


#from icecream import ic

def compute_mask_contamination(mask: Union[torch.Tensor, Image]) -> float:
    if isinstance(mask, Image):
        mask = torch.tensor(np.array(mask))
    contaminated_pixels = torch.nonzero(mask).shape[0]
    return contaminated_pixels / torch.flatten(mask).shape[0]

def prepare_dictionary(filename:str, hparams):
    if hparams.ckpt_pretrained != "":
        filename = hparams.ckpt_pretrained
        #print(f"Loading from checkpoint: {filename}")
    state_dict = torch.load(filename, map_location=torch.device('cpu'))
    #print(filename)
    if len(state_dict.keys()) == 1:
        key = list(state_dict.keys())[0]
        #print(f"One key found in loaded weights: {key}")
        state_dict = state_dict[key]
    else:
        num_keys = len(state_dict.keys())
        #print(f"Dictionary was ready to be uploaded with {num_keys if num_keys > 5 else state_dict.keys()} keys")

        
    return state_dict


def eventually_load(self, state_dict: Dict, keyword: str):
    assert isinstance(self, nn.Module)
    if state_dict:
        if keyword in state_dict:
            #print("Utilizing " + keyword)
            ic(self.load_state_dict(state_dict=state_dict[keyword], strict=False))
            return True
    return False

setattr(nn.Module, "eventually_load", eventually_load)


class Projection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 2048, latent_size:int = 128, avg = False):
        super(Projection, self).__init__()
        layers = []
        if avg:
            layers.extend([
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            ])
        layers.extend([
            nn.Linear(in_features=in_features, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=latent_size)
        ])
        self.core = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.core(x)



