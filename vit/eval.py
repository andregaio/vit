import torch
import argparse
from tqdm import tqdm

from data import load_dataset, compute_accuracy, CLASSES
from vit import VGG


def eval(args):

    _, valloader = load_dataset(batch_size = args.batch)
    model = VGG(classes = CLASSES,  network = args.model)
    model.load_state_dict(torch.load(args.weights))
    model.cuda()

    with torch.no_grad():
        val_rolling_accuracy = 0
        for i, data in tqdm(enumerate(valloader, 0), total=len(valloader)):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            val_accuracy = compute_accuracy(outputs, labels)
            val_rolling_accuracy += val_accuracy
        val_average_accuracy = val_rolling_accuracy / len(valloader)
        print(f'Avg. Accuracy: {val_average_accuracy:.2f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--model", type = str, default = 'vgg_A')
    parser.add_argument("--weights", type = str, default = 'weights/checkpoint_00070.pt')
    parser.add_argument("--dataset", type = str, default = 'cifar10')
    parser.add_argument("--batch", type = int, default = 64)
    args = parser.parse_args()

    eval(args)