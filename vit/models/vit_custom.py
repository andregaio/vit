import torch
from torch import nn
from models.modules.vit import ViT
from einops.layers.torch import Rearrange
import math


class ViTCustom(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes):
        super().__init__()

        self.height, self.width = image_size

        self.model = ViT(image_size=image_size,
              patch_size=patch_size,
              num_classes=num_classes,
              dim=768,
              depth=12,
              heads=12,
              mlp_dim=3072)

        self.model.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w c) (p1 p2)', p1 = patch_size[0], p2 = patch_size[1]),
            nn.LayerNorm(math.prod(patch_size)),
            nn.Linear(math.prod(patch_size), 768),
            nn.LayerNorm(768),
        )

        self.model.pos_embedding = nn.Parameter(torch.randn(1, int(math.prod(image_size) / math.prod(patch_size) * 3) + 1, 768))

    def forward(self, img):
        return self.model(img)
    
if __name__ == "__main__":
    model = ViTCustom(image_size=(32, 32), patch_size=(4, 4), num_classes=10)
    image = torch.rand(1, 3, 32, 32)
    out = model(image)
    print(out.shape)