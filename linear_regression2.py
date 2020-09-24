import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


# Seed setting for reproducible results
torch.manual_seed(2)

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

from torch.utils.data import TensorDataset

# Define dataset
train_ds = TensorDataset(inputs, targets)

from torch.utils.data import DataLoader

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

batch_size = 1
train_dl2 = DataLoader(train_ds, batch_size, shuffle=True)
# Define model
model = nn.Linear(3, 2)
model2 = nn.Linear(3,2)

# Import nn.functional
import torch.nn.functional as F

loss_fn = F.mse_loss

# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)
opt2 = torch.optim.SGD(model2.parameters(), lr=1e-5)

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    all_losses = []
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:
            # 1. Generate predictions
            preds = model(xb)

            # 2. Calculate loss
            loss = loss_fn(preds, yb)
            all_losses.append(loss)
            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    return all_losses


def show_loss(all_losses):
    plt.plot(range(len(all_losses)), all_losses, color="red")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.show()

all_losses = fit(300, model, loss_fn, opt, train_dl)
all_losses2 = fit(100, model2, loss_fn, opt2, train_dl2)
show_loss(all_losses)
show_loss(all_losses2)