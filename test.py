import train
import argparse

if '__name__' == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN on MNIST or CIFAR-10')
    parser.add_argument('--dataset', type=str, choices=['MNIST', 'CIFAR10'], default='MNIST', help='Dataset to use (default: MNIST)')
    parser.add_argument('--attack_type', type=str, choices=['FGSM', 'PGD'], default='FGSM', help='Type of adversarial attack to use')
    args = parser.parse_args()
    
    train.main(args)