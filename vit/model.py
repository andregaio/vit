import torch
import torch.nn as nn
from vit.models.vit_base import ViTBase
from vit.models.vit_large import ViTLarge
from vit.models.vit_huge import ViTHuge
from utils import model_info, print_model_info


networks = {
    'vit_base' : ViTBase,
    'vit_large' : ViTLarge,
    'vit_huge' : ViTHuge,
}


def _xavier_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class Model(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, network = 'vit_base'):
        super(Model, self).__init__()
        
        self.height, self.width = image_size

        model = networks[network]
        self.network = model(image_size=image_size,
                             patch_size=patch_size,
                             num_classes=num_classes)
        _xavier_init(self.network)
        
        self.params, self.flops = model_info(self.network, torch.rand(1, 3, self.height, self.width))

    def forward(self, x):
        return self.network(x)

    def info(self):
        print_model_info(self.params, self.flops)