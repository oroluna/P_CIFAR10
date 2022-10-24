'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np

import os
import argparse 

from engine.model import *
from engine.utils.utils import progress_bar

#Hydra import

from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base="1.1", config_path="config", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    print(OmegaConf.to_yaml(cfg))


    #parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    #parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    #parser.add_argument('--resume', '-r', action='store_true',
    #                    help='resume from checkpoint')
    #args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Trainset
    trainset = torchvision.datasets.CIFAR10(
        root=cfg.paths.data, train=True, download=True, transform=transform_train)
    # Sample train subset
    
    num_train_samples =  cfg.data.train
    sample_train = torch.utils.data.Subset(trainset, np.arange(num_train_samples))

    trainloader = torch.utils.data.DataLoader(
        sample_train, batch_size=cfg.params.train.batch_size, shuffle=True, num_workers=2)

    # Testset
    testset = torchvision.datasets.CIFAR10(
        root=cfg.paths.data, train=False, download=True, transform=transform_test)
    # Sample test subset
    num_test_samples = cfg.data.test
    sample_test = torch.utils.data.Subset(testset, np.arange(num_test_samples))

    testloader = torch.utils.data.DataLoader(
        sample_test, batch_size=cfg.params.test.batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG11')
    net = hydra.utils.instantiate(cfg.model)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # If True resume The last training and restart from the last epoch

    if cfg.args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    print(cfg.args.lr)
    optimizer = optim.SGD(net.parameters(), lr=cfg.args.lr,
                        momentum=0.9, weight_decay=cfg.params.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch, best_acc=best_acc):
        best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
          
            # Save metrics in outputs/year-month-day/hour-minut-second
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            with open(f"{hydra_cfg['runtime']['output_dir']}/main.log", "w") as file:
                file.write('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

          

    for epoch in range(start_epoch, start_epoch+cfg.params.epoch_count):
        train(epoch)
        test(epoch)
        scheduler.step()

if __name__ == "__main__":
    my_app()
