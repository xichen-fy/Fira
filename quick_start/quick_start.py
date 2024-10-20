import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
import random

# Set random seed for reproducibility
def seed_torch(seed=1029):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='fira_adamw', help='Optimizer to use')
    return parser.parse_args()

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

def main():
    seed_torch() # Set random seed for reproducibility
    args = parse_args()

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

    import torch.optim as optim
    from fira import FiraAdamW, divide_params

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    param_groups = divide_params(model, target_modules_list = ["fc"], rank=8)  # Group parameters
    if args.optimizer == "fira_adamw":
        optimizer = FiraAdamW(param_groups, lr=0.01)  # Use FiraAdamW optimizer
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)  # Use Adam optimizer
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=0.01) # Use AdamW optimizer

    # Run the training and testing loop
    for epoch in range(1, 6):  # 5 epochs
        train(model, train_loader, criterion, optimizer, epoch)
        test(model, test_loader, criterion)


if __name__ == "__main__":
    main()