import torch
import torch.nn as nn
import torch.optim as optim
from model import LeNet
from utils import (
    load_checkpoint,
    save_checkpoint,
)
import torchvision.datasets as datasets  # Standard datasets
from torch.utils.data import (
    DataLoader,
)  
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4 # karpathy's constant
batch_size = 16
num_epochs = 3

# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.Compose([transforms.ToTensor(),
     transforms.Resize((32,32), interpolation=2),]), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.Compose([transforms.ToTensor(),
     transforms.Resize((32,32), interpolation=2),]), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = LeNet(in_channel=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []
    for batch_idx, (data, targets) in enumerate((train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())


        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


