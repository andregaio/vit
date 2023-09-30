from torch import nn
from models.modules.vit import ViT

class ViTHuge(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes):
        super().__init__()

        self.model = ViT(image_size=image_size,
              patch_size=patch_size,
              num_classes=num_classes,
              dim=1280,
              depth=32,
              heads=16,
              mlp_dim=5120)

    def forward(self, img):
        return self.model(img)