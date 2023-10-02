import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from data import VAL_TRANSFORMS, CLASSES
from model import Model


def predict(args):

    model = Model(image_size=(32, 32), patch_size=(4, 4), num_classes=10, network = args.model)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    img = Image.open(args.image)
    img = VAL_TRANSFORMS(img)
    max_value, max_index = torch.max(F.softmax(model(img.unsqueeze(dim=0)), dim = 1), dim=1)
    label = CLASSES[max_index.item()]
    score = max_value.item()
    print(f'Label: {label}; Score: {score:.2}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--model", type = str, default = 'vit_base')
    parser.add_argument("--weights", type = str, default = 'weights/checkpoint_00070.pt')
    parser.add_argument("--image", type = str, default = 'assets/cat.png')
    args = parser.parse_args()

    predict(args)