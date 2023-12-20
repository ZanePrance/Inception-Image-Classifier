import  torch
import torch.utils.data
import torchvision
import  torchvision.transforms as transforms
import  matplotlib.pyplot as plt
import numpy as np
import argparse

# change the from call to match with the model you are currently working with
from modified import Encoder, ImageClassifier

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_pred(model, test_loader, classes):
    model.eval()
    images, labels = next(iter(test_loader))

    image, label = images[0], labels[0]

    imshow(torchvision.utils.make_grid(image))

    print('GroundTruth:', classes[label.item()])


    output, _ = model(image.unsqueeze(0))
    _, pred = torch.max(output, 1)
    print('Predictied:', classes[pred.item()])

    top5 = torch.topk(output, 5).indices.squeeze(0).tolist()
    print('Top 5 Predictions: ', ' '.join('%5s' % classes[j] for j in (top5)))

def main():
    parser = argparse.ArgumentParser(description='Load a trained model.')


    parser.add_argument('-ew', '--encoder_weights_path', type=str, help='Path to the encoder weights file.')
    parser.add_argument('-mp', '--model_path', type=str, help='Path to the model weights file.')

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    encoder_weights_path = args.encoder_weights_path
    model_path = args.model_path
    encoder = Encoder.encoder

    encoder.load_state_dict(torch.load(encoder_weights_path))

    num_classes = 100
    model = ImageClassifier(encoder=encoder, num_classes=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    device = 'cpu'
    model.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testdata = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testdata, batch_size=1,shuffle=True)
    classes = {i: class_name for i, class_name in enumerate(testdata.classes)}

    test_pred(model, test_loader, classes)

if __name__ == '__main__':
    main()
