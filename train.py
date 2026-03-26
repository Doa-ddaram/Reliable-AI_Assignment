from cnn_models import VGG16, MNISTCNN
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_mnist_dataloader, get_cifar10_dataloader
from tqdm import tqdm
import argparse


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    print(f'Loading {args.dataset} dataset...')
    
    # Load data
    if args.dataset == 'MNIST':
        train_loader = get_mnist_dataloader(root='./data', batch_size=64, train=True)
        test_loader = get_mnist_dataloader(root='./data', batch_size=64, train=False)
        model = MNISTCNN(input_channels=1, num_classes=10).to(device)
    else:
        train_loader = get_cifar10_dataloader(root='./data', batch_size=64, train=True)
        test_loader = get_cifar10_dataloader(root='./data', batch_size=64, train=False)
        model = VGG16(input_channels=3, num_classes=10).to(device)

    # Initialize model, criterion, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10'])
    args = parser.parse_args()
    
    main(args)