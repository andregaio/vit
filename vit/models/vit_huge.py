import torch
from torch import nn
from models.modules.vit import ViT
from utils import model_info, print_model_info


class ViTHuge(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes):
        super().__init__()

        self.height, self.width = image_size

        self.model = ViT(image_size=image_size,
              patch_size=patch_size,
              num_classes=num_classes,
              dim=1280,
              depth=32,
              heads=16,
              mlp_dim=5120)

        self.params, self.flops = model_info(self.model, torch.rand(1, 3, self.height, self.width))

    def forward(self, img):
        return self.model(img)
    
    def info(self):
        print_model_info(self.params, self.flops)