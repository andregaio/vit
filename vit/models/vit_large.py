from torch import nn
from models.modules.vit import ViT


class ViTLarge(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes):
        super().__init__()

        self.height, self.width = image_size

        self.model = ViT(image_size=image_size,
              patch_size=patch_size,
              num_classes=num_classes,
              dim=1024,
              depth=24,
              heads=16,
              mlp_dim=4096)

    def forward(self, img):
        return self.model(img)