## A quick-start tutorial for the Fira optimizer

This quick-start tutorial will guide you through setting up a simple neural network training and testing routine using PyTorch and the memory-efficient Fira optimizer.

We'll use it to train a simple fully connected network on the MNIST dataset, which consists of 28x28 pixel images of handwritten digits.

### Step 1: Set Up Your Environment
First, ensure you have the necessary packages installed. You need `torch`, `torchvision`, and `fira`. Install them using pip if you haven't already:

```bash
pip install torch torchvision fira
```

### Step 2: Define the Model

We'll use a simple neural network with two fully connected layers. The network definition is straightforward:

```python
import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # First layer
        self.fc2 = nn.Linear(128, 10)     # Output layer

    def forward(self, x):
        x = x.view(-1, 28*28)            # Flatten the image
        x = F.relu(self.fc1(x))          # First layer activation
        x = self.fc2(x)                  # Output layer
        return x
```

### Step 3: Prepare the Dataset

We use the MNIST dataset, which is conveniently available through `torchvision`.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
```

### Step 4: Initialize the Model, Loss Function, and Optimizer

```python
import torch.optim as optim
from fira import FiraAdamW, divide_params

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
param_groups = divide_params(model, target_modules_list=["Linear"])  # Group parameters
optimizer = FiraAdamW(param_groups, lr=0.01)  # Use FiraAdamW optimizer
```

### Step 5: Train and Test the Model

Define the training and testing functions. The training function processes data in batches, computes loss, and updates the model using the optimizer. The testing function evaluates the model's performance.

```python
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# Run the training and testing loop
for epoch in range(1, 6):  # 5 epochs
    train(model, train_loader, criterion, optimizer, epoch)
    test(model, test_loader, criterion)
```

### Result
In Fira, Adam is used by default with `weight_decay=0`. We compare low-rank Fira with the full-rank Adam and AdamW optimizer at the same learning rate. 

#### Low-rank Fira ($rank/d_{model}=8/128$)
```python
from fira import FiraAdamW, divide_params
param_groups = divide_params(model, target_modules_list = ["fc"], rank=8)
optimizer = FiraAdamW(param_groups, lr=learning_rate)
```
The modules `self.fc1` and `self.fc2` will enable Fira because their names contain the substring `fc`, as specified in the `target_modules_list`.
```bash
python quick_start.py --optimizer fira_adamw  
```
Final Test set result: Average loss: 0.0036, Accuracy: 9518/10000 (95.04%)
#### Full-rank Adam
```python
optimizer = optim.Adam(model.parameters(), lr=0.01)
```
```bash
python quick_start.py --optimizer adam  
```
Final Test set result: Average loss: 0.0040, Accuracy: 9434/10000 (94.34%)
#### Full-rank AdamW
```python
optimizer = optim.AdamW(model.parameters(), lr=0.01)
```
```bash
python quick_start.py --optimizer adamw  
```
Final Test set result: Average loss: 0.0040, Accuracy: 9368/10000 (93.68%)

## Documentation for Fira

### 1. `divide_params` Function

The `divide_params` function is used to mark certain modules within a model to utilize Fira memory-efficient techniques.

- **Parameters**:
  - `model` (`nn.Module`, optional): The model instance to configure. Defaults to `None`.
  - `target_modules_list` (`list[str]`, optional): Modules whose names contain any substring in `target_modules_list` will enable Fira. Adapt as necessary for specific module names in the model. Defaults to an empty list.
  - `rank` (`int`, optional): The dimension to which model parameters are compressed. Defaults to `8`.
  - `update_proj_gap` (`int`, optional): The interval (in training steps) for updating the gradient projections. Defaults to `200`.
  - `alpha` (`float`, optional): Adjusts the learning rate influence, analogous to LoRA. Defaults to `1.0`.
  - `proj_type` (`str`, optional): Specifies the type of projection used. Defaults to `'std'`.

- **Returns**:
  - A modified version of the parameters following Fira's configuration.

- **Description**:
  - This function iterates over the specified modules in `target_modules_list`, applying Fira's memory-efficient strategies. 

### 2. `FiraAdamW` Optimizer

The `FiraAdamW` optimizer enhances the traditional `AdamW` optimizer, supporting memory-efficient training of LLMs.

- **Parameters**:
  - `params` (`Iterable[nn.parameter.Parameter]`): An iterable of parameters to optimize, or dictionaries defining parameter groups.
  - `lr` (`float`, optional): Specifies the learning rate. Defaults to `0.001`.
  - `betas` (`Tuple[float, float]`, optional): Beta coefficients for the Adam optimizer (b1, b2). Defaults to `(0.9, 0.999)`.
  - `eps` (`float`, optional): Epsilon value for numerical stability in Adam's calculations. Defaults to `1e-06`.
  - `weight_decay` (`float`, optional): Applies decoupled weight decay. Defaults to `0.0`.
  - `correct_bias` (`bool`, optional): Corrects bias in Adam's update. Defaults to `True`, although some configurations like Bert TF use `False`.
  - `no_deprecation_warning` (`bool`, optional): Suppresses deprecation warnings if set to `True`. 