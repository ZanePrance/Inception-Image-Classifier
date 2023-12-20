import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from vanilla import Encoder, ImageClassifier


def calc_accuracy(outputs, targets, topk=(1, 5)):
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    topk_accs = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        topk_acc = correct_k.mul_(100.0 / batch_size)
        topk_accs.append(topk_acc.item())

    return topk_accs


def train(model, iterator, optimizer, loss_function, device, num_epochs, val_iterator):
    print("Training...")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False)

    top1_errors = []
    top5_errors = []
    losses = []
    val_losses = []

    model.to(device)
    loss_function = loss_function.to(device)

    best_top5 = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        top1_error = 0
        top5_error = 0
        total_imgs = 0

        for batch_idx, (x, y) in enumerate(tqdm(iterator, desc=f"Epoch {epoch + 1}", leave=False)):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_prediction, _ = model(x)

            loss = loss_function(y_prediction, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(iterator)
        losses.append(avg_epoch_loss)

        model.eval()
        total_val_imgs = 0
        with torch.no_grad():
            epoch_val_loss = 0
            for x, y in val_iterator:
                x = x.to(device)
                y = y.to(device)
                y_pred, _ = model(x)
                loss = loss_function(y_pred, y)

                top1_acc, top5_acc = calc_accuracy(y_pred, y)
                top1_error += 100 - top1_acc
                top5_error += 100 - top5_acc

                epoch_val_loss += loss.item()
                total_val_imgs += x.size(0)

            avg_val_loss = epoch_val_loss / len(val_iterator)
            val_losses.append(avg_val_loss)

        top1_errors.append(top1_error / len(val_iterator))
        top5_errors.append(top5_error / len(val_iterator))
        scheduler.step(avg_val_loss)

        if top5_error < best_top5:
            best_top5 = top5_error
            epochs_no_improve = 0
            torch.save(model.state_dict(), '/content/drive/My Drive/Lab3/best_curr_weight_mod.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == 8:
            print(f'Early stopping triggered after {epoch + 1}')
            break

        print(
            f'Epoch: [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}, Top-1 Error: {top1_errors[-1]:.2f}%, Top-5 Error: {top5_errors[-1]:.2f}%')

    torch.save(model.state_dict(), '/content/drive/My Drive/Lab3/final_weight_mod.pth')

    actual_epochs = len(losses)

    # Plot the training curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs + 1), losses, label='Training Loss')
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), top1_errors, label='Top-1 Error')
    plt.plot(range(1, actual_epochs + 1), top5_errors, label='Top-5 Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate (%)')
    plt.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train a neural network on CIFAR100.')

    # Add arguments
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=5e-4)
    parser.add_argument('-b', '--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('-e', '--num_epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('-nc', '--num_classes', type=int, help='Number of classes', default=100)
    parser.add_argument('-ew', '--encoder_weights_path', type=str, help='Path to encoder weights')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    encoder_weights_path = args.encoder_weights_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder.encoder
    encoder.load_state_dict(torch.load(encoder_weights_path))

    model = ImageClassifier(encoder=encoder, num_classes=num_classes)

    pretrained_size = 128
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize(pretrained_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(pretrained_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transforms)
    train_iterator = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transforms)
    val_iterator = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_iterator, optimizer, loss_function, device, num_epochs, val_iterator)


if __name__ == "__main__":
    main()
