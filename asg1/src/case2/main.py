import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

seed = 42
torch.manual_seed(seed)
#random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Download the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

print("number of training samples: " + str(len(train_dataset)) + "\n" +
      "number of testing samples: " + str(len(test_dataset)))

print("datatype of the 1st training sample: ", train_dataset[0][0].type())
print("size of the 1st training sample: ", train_dataset[0][0].size())

batch_size = 64

# Create data loaders.
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Verify size of batches
for X, y in test_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Hardware
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define the model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.tanh(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 256)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

model = LeNet5().to(device)

count_parameters(model)
fc12_params = [p for name, p in model.named_parameters() if name in ['fc1.weight', 'fc2.weight']]
print(fc12_params[0].numel())
print(fc12_params[1].numel())

# raise SystemExit

# Training loop
optimizer_name = "ADAMWlr0,002wd0,006BatchSize64"
criterion_name = "CrEntLoss" 

# def closure():
#     optimizer.zero_grad()
#     closure_output = model(images)
#     closure_loss = criterion(closure_output, labels)
#     closure_loss.backward()
#     return closure_loss

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.002,weight_decay=0.006)
n_epochs = 10
train_losses = []
test_losses = []
test_accuracies = []

#### SCHEDULER??
#### Either ADAMlr0,002 or SGDlr0,02NestMom0,9 is best
#### ADAMWlr0,002wd0,01 Performing alike maybe a bit better
#### ADAMWlr0,002wd0,006 SCORED 90 in accuracy as the first YEAH!

step = 0
for epoch in range(n_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        backward_loss = loss.backward()
        optimizer.step()
        step += 1
        train_losses.append((step, loss.item()))

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            test_loss += criterion(output, labels).item()
            pred = torch.argmax(output, dim=1)
            correct += pred.eq(labels).sum()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Epoch {epoch} Loss {test_loss}')

    test_losses.append((step, test_loss))
    test_accuracies.append(test_accuracy.item())

    print(f'Test set: Average loss: {test_loss}, \
        Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy}%)')


# plot train and test losses to file loss.png
train_steps, train_loss = zip(*train_losses)
test_steps, test_loss = zip(*test_losses)
#test_steps, test_accuracy = zip(*test_accuracies)

fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)  # sharex aligns x-axes

# Plot the first
ax[0].plot(train_steps, train_loss, label="Train Loss", color="blue")
ax[0].set_ylabel("Loss")
ax[0].legend()
ax[0].grid(True)

# Plot the second
ax[1].plot(test_steps, test_loss, label="Test Loss, Average", color="red")
ax[1].set_ylabel("Loss")
ax[1].legend()
ax[1].grid(True)

# Plot the Third
ax[2].plot(test_steps, test_accuracies, label="Test Accuracy", color="green")
ax[2].set_ylabel("Accuracy (%)")
ax[2].set_xlabel("Epoch")
ax[2].legend()
ax[2].grid(True)

steps_per_epoch = len(train_loader)
tick_positions = [i * steps_per_epoch for i in range(n_epochs + 1)]
tick_labels = [str(i) for i in range(n_epochs + 1)]

plt.xticks(tick_positions, tick_labels)

# Adjust layout and show the plot
plt.tight_layout()
# plt.show()
plt.savefig(f"{optimizer_name}+{criterion_name}+epoc{n_epochs}")