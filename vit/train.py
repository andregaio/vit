import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
import wandb
import argparse
from data import load_dataset, compute_accuracy, CLASSES


def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, valloader = load_dataset(batch_size = args.batch)
    
    model = Model(image_size=tuple(args.image_size),
                  patch_size=tuple(args.patch_size),
                  num_classes=10,
                  network = args.model)
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    scaler = torch.cuda.amp.GradScaler()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.wandb:
        run = wandb.init(
            project='vit',            
            config={
                'network': args.model,
                'params': model.params,
                'flops': model.flops,
                'img_size': 'x'.join(args.img_size),
                'patch_size': 'x'.join(args.patch_size),
                'dataset': args.dataset,
                'epochs': args.epochs,
                'batch' : args.batch,
                'learning_rate' : args.learning_rate,
                'loss' : 'adam',
                'weight_decay' : args.weight_decay,
            })


    for epoch in range(args.epochs):
        train_rolling_accuracy = train_rolling_loss = 0.
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_accuracy = compute_accuracy(outputs, labels)
            train_rolling_accuracy += train_accuracy
            train_rolling_loss += loss.item()
            print(f'Epoch: {epoch + 1} / {args.epochs}, Iter: {(i + 1)} / {len(trainloader)} Loss: {loss.item():.3f} Accuracy: {train_accuracy:.2f}')
        train_average_accuracy = train_rolling_accuracy / len(trainloader)
        train_average_loss = train_rolling_loss / len(trainloader)
        print(f'Epoch: {epoch + 1} Avg. Loss: {train_average_loss:.2f} Avg. Accuracy: {train_average_accuracy:.2f}')

        scheduler.step()

        with torch.no_grad():
            val_rolling_accuracy = val_rolling_loss = max_val_accuracy = 0.
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device=device), labels.to(device=device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_accuracy = compute_accuracy(outputs, labels)
                val_rolling_accuracy += val_accuracy
                val_rolling_loss += loss.item()
                print(f'Epoch: {epoch + 1} / {args.epochs}, Iter: {(i + 1)} / {len(valloader)} Loss: {loss.item():.3f} Accuracy: {val_accuracy:.2f}')
            val_average_accuracy = val_rolling_accuracy / len(valloader)
            val_average_loss = val_rolling_loss / len(valloader)
            print(f'Epoch: {epoch + 1} Avg. Loss: {val_average_loss:.2f} Avg. Accuracy: {val_average_accuracy:.2f}')

        if args.wandb:

            if val_average_accuracy > max_val_accuracy:
                weights_filepath = f'weights/{run.name}_checkpoint_{args.dataset}_{args.model}_best.pt'
                torch.save(model.state_dict(), weights_filepath)
                max_val_accuracy = val_average_accuracy

            weights_filepath = f'weights/{run.name}_checkpoint_{args.dataset}_{args.model}_last.pt'
            torch.save(model.state_dict(), weights_filepath)

            wandb.log({'train_average_accuracy': train_average_accuracy,
                        'val_average_accuracy': val_average_accuracy,
                        'train_average_loss': train_average_loss,
                        'val_average_loss': val_average_loss,
                        'max_val_accuracy': max_val_accuracy,
                    })


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument("--model", type = str, default = 'vit_base')
    parser.add_argument("--dataset", type = str, default = 'cifar10')
    parser.add_argument("--batch", type = int, default = 128)
    parser.add_argument("--epochs", type = int, default = 78)
    parser.add_argument("--image_size", nargs="+", type = int, default = [32, 32])
    parser.add_argument("--patch_size", nargs="+", type = int, default = [4, 4])
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    parser.add_argument("--weight_decay", type = float, default = 0.1)
    parser.add_argument("--wandb", action="store_true", default = False)
    args = parser.parse_args()

    train(args)