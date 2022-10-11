import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNN
import sys

def main(args):
    CHANNELS_D = 3
    LEARNING_RATE = 1e-4
    try:
        img_size = int(args[args.index('--img') + 1]) if '--img' in args else 400
        batch = int(args[args.index('--batch') + 1]) if '--batch' in args else 32
        epochs = int(args[args.index('--epochs') + 1]) if '--epochs' in args else 64
    except ValueError:
        print("Usage: train.py --img <img size> --batch <batch size> --epochs <# epochs>")
        exit()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_D)], [0.5 for _ in range(CHANNELS_D)]
            ),
        ]
    )
    dataset = datasets.ImageFolder(
        root="~/Documents/datasets/archive/caltech101_classification/",
        transform=transform)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = CNN(img_size, 6).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(loader):
            inputs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'epoch: {epoch + 1} loss: {running_loss / batch:.3f}')
        running_loss = 0.0
    
    torch.save(cnn.state_dict(), "model.pth")

if __name__ == "__main__":
    main(sys.argv)
