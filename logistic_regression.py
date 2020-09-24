import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Download dataset
dataset = MNIST(root='~/Desktop/data', transform=transforms.ToTensor())
test_dataset = MNIST(root='~/Desktop/data', train=False, transform=transforms.ToTensor())

# Create random train and validation set
train_ds, val_ds = random_split(dataset, [50000, 10000])

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds)

input_size = 28*28
num_classes = 10

# TRAINING THE MODEL
# for epoch in range(num_epochs):
#     # TRAINING PHASE
#     for batch in train_loader:
#         # Generate predictions
#         # Calculate loss
#         # Compute gradients
#         # Update weights
#         # Reset gradients
    
#     # VALIDATION PHASE
#     for batch in val_loader:
#         # Generate predictions
#         # Calculate loss
#         # Calculate metrics (accuracy etc.)
#     # CALCULATE AVERAGE VALIDATION LOSS & METRICS
    
#     # LOG EPOCH, LOSS & METRICS FOR INSPECTION

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                          # Generate predictions
        loss = F.cross_entropy(out, labels)         # Calculate loss
        acc = accuracy(out, labels)                 # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]        # Combine accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel()

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

history1 = fit(5, 0.001, model, train_loader, val_loader)
history2 = fit(5, 0.001, model, train_loader, val_loader)

history = history1 + history2
accuracies = [result['val_acc'] for result in history]

plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')
plt.show()