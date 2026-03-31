import argparse
from datasets import get_mnist_dataloader, get_cifar10_dataloader
from cnn_models import ResNet18, MNISTCNN
import torch
import torch.nn as nn
import torch.optim as optim
from train import train, evaluate
import os, csv

def init_csv(filename):
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Epsilon', 'Clean_Acc', 'FGSM_Untargeted', 'FGSM_Targeted', 'PGD_Untargeted', 'PGD_Targeted'])
    return filepath

def append_results_to_csv(filepath, epoch, results):
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        for res in results:
            eps = res['eps']
            acc = res['clean_accuracy']
            asr = res['asr']
            writer.writerow([
                epoch,
                f"{eps:.3f}",
                f"{acc:.4f}",
                f"{asr['fgsm_untargeted']:.4f}",
                f"{asr['fgsm_targeted']:.4f}",
                f"{asr['pgd_untargeted']:.4f}",
                f"{asr['pgd_targeted']:.4f}"
            ])

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
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
    
    if args.use_attack:
        csv_filename = f"results/attack_{args.dataset.lower()}_eps_{args.attack_eps}.csv"
        csv_filepath = init_csv(csv_filename)
    
    # Train the model
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
    
        is_attack_epoch = args.use_attack and ((epoch + 1) % args.eval_interval == 0 or (epoch + 1) == num_epochs)
        
        if is_attack_epoch:
            test_loss, test_accuracy, metrics = evaluate(
                model,
                test_loader,
                criterion,
                device,
                use_attack=args.use_attack,
                attack_eps=args.attack_eps,
                class_names=class_names,
                return_attack_metrics=True,
                save_visuals=(epoch == num_epochs - 1),
            )
            append_results_to_csv(csv_filepath, epoch + 1, [metrics])
            print(f'Epoch {epoch+1}/{num_epochs} [ATTACK EVAL] Train Loss: {train_loss:.4f}, Test Acc: {test_accuracy:.4f}')
        else:
            test_loss, test_accuracy = evaluate(
                model, test_loader, criterion, device,
                use_attack=False,
                class_names=class_names,
                return_attack_metrics=False,
                save_visuals=False,
            )
            print(f'Epoch {epoch+1}/{num_epochs} [CLEAN EVAL] Train Loss: {train_loss:.4f}, Test Acc: {test_accuracy:.4f}')
            
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Train a CNN on MNIST or CIFAR-10')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10'], default='MNIST', help='Dataset to use (default: MNIST)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--use_attack', action='store_true', help='Whether to use adversarial attacks during training (default: False)')
    parser.add_argument('--attack_eps', type=float, default=0.3, help='Epsilon value for adversarial attacks (default: 0.3)')
    parser.add_argument('--pgd_steps', type=int, default=30, help='Number of PGD iterations (default: 30)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--eval_interval', type=int, default=3, help='Epoch interval to run attack evaluation (default: 3)')
    args = parser.parse_args()
    
    if args.use_attack:
        print(f'Training on {args.dataset} dataset with adversarial attacks...')


    main(args)