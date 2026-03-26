import argparse
from datasets import get_mnist_dataloader, get_cifar10_dataloader
from cnn_models import ResNet18, MNISTCNN
import torch
import torch.nn as nn
import torch.optim as optim
from train import train, evaluate

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    use_attack = args.use_attack
    # Load data
    if args.dataset == 'MNIST':
        train_loader = get_mnist_dataloader(root='./data', batch_size=64, train=True)
        test_loader = get_mnist_dataloader(root='./data', batch_size=64, train=False)
        model = MNISTCNN(input_channels=1, num_classes=10).to(device)
        class_names = [str(i) for i in range(10)]
    else:
        train_loader = get_cifar10_dataloader(root='./data', batch_size=64, train=True)
        test_loader = get_cifar10_dataloader(root='./data', batch_size=64, train=False)
        model = ResNet18(num_classes=10).to(device)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Initialize model, criterion, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device, use_attack=use_attack, attack_eps=args.attack_eps, class_names=class_names)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Train a CNN on MNIST or CIFAR-10')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10'], default='MNIST', help='Dataset to use (default: MNIST)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--use_attack', action='store_true', help='Whether to use adversarial attacks during training (default: True)')
    parser.add_argument('--attack_eps', type=float, default=0.3, help='Epsilon value for adversarial attacks (default: 0.3)')
    args = parser.parse_args()
    
    if args.use_attack:
        print(f'Training on {args.dataset} dataset with adversarial attacks...')
    
    main(args)