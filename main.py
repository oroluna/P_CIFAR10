'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import csv
import pandas as pd
import numpy as np
from engine.utils.utils import progress_bar
from plotting import *

from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt



from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path="config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print("Working directory : {}".format(os.getcwd()))
    print(OmegaConf.to_yaml(cfg))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    num_train_samples = cfg.data.train
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

    net = hydra.utils.instantiate(cfg.model)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if cfg.args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = hydra.utils.call(cfg.optimizer, params=net.parameters(), **cfg.optimizer.params)
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
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            loss = train_loss / (batch_idx + 1)
            accuracy = 100. * correct / total



            train_iterations_results["loss"].append(loss)
            train_iterations_results["accuracy"].append(accuracy)
            train_iterations_results["epoch"].append(epoch)

        train_loss = train_loss / len(trainloader)
        train_acc = correct / len(trainloader)

        return train_loss, train_acc

    def test(epoch):
        global best_acc
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
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                loss = test_loss / (batch_idx + 1)
                accuracy = 100. * correct / total

                test_iterations_results["loss"].append(loss)
                test_iterations_results["accuracy"].append(accuracy)
                test_iterations_results["epoch"].append(epoch)

        # Save checkpoint.
        acc = 100. * correct / total
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

        test_loss = test_loss / len(testloader)
        test_acc = correct / len(testloader)

        return test_loss, test_acc


    ############ TRAINING ############


    epoch_results = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    train_iterations_results = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
    }

    test_iterations_results = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
    }

    for epoch in range(start_epoch, start_epoch + cfg.params.epoch_count):

        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        # Update results dictionary
        epoch_results["epoch"].append(epoch)
        epoch_results["train_loss"].append(train_loss)
        epoch_results["train_acc"].append(train_acc)
        epoch_results["test_loss"].append(test_loss)
        epoch_results["test_acc"].append(test_acc)

        scheduler.step()

    # get output route
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    OUTPUT_ROUTE = hydra_cfg['runtime']['output_dir']

    # CONFUSION MATRIX
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth


    #### SAVE IN TO CSV ####

    if not os.path.exists(f"{OUTPUT_ROUTE}/metrics"):
        os.mkdir(f"{OUTPUT_ROUTE}/metrics")

    # create train iterations metrics dataframe
    train_iterations_metrics_df = pd.DataFrame(train_iterations_results)

    # add moving average accuracy to train iterations
    train_iterations_metrics_df['moving_acc_average'] = train_iterations_metrics_df['accuracy'].expanding().mean()

    # save train iterations metrics
    train_iterations_metrics_df.to_csv(f"{OUTPUT_ROUTE}/metrics/train_iteration_metrics.csv", index=False)

    # create test iterations metrics dataframe
    test_iterations_metrics_df = pd.DataFrame(test_iterations_results)

    # save test iteration metrics
    test_iterations_metrics_df.to_csv(f"{OUTPUT_ROUTE}/metrics/test_iteration_metrics.csv", index=False)

    # create epoch metrics
    epoch_metrics_df = pd.DataFrame(epoch_results)

    # add train_max_accuracy_iteration per epoch
    epoch_metrics_df['train_max_accuracy_iteration'] = train_iterations_metrics_df.groupby("epoch")["accuracy"].max()

    # add train_min_accuracy_iteration per epoch
    epoch_metrics_df['train_min_accuracy_iteration'] = train_iterations_metrics_df.groupby("epoch")["accuracy"].min()

    # add train_max_loss_iteration per epoch
    epoch_metrics_df['train_max_loss_iteration'] = train_iterations_metrics_df.groupby("epoch")["loss"].max()

    # add train_min_loss_iteration per epoch
    epoch_metrics_df['train_min_loss_iteration'] = train_iterations_metrics_df.groupby("epoch")["loss"].min()

    # add test_max_accuracy_iteration per epoch
    epoch_metrics_df['test_max_accuracy_iteration'] = test_iterations_metrics_df.groupby("epoch")["accuracy"].max()

    # add test_max_accuracy_iteration per epoch
    epoch_metrics_df['test_min_accuracy_iteration'] = test_iterations_metrics_df.groupby("epoch")["accuracy"].min()

    # add min_loss_iteration per epoch
    epoch_metrics_df['test_max_loss_iteration'] = test_iterations_metrics_df.groupby("epoch")["loss"].max()

    # add min_loss_iteration per epoch
    epoch_metrics_df['test_min_loss_iteration'] = test_iterations_metrics_df.groupby("epoch")["loss"].min()

    print(epoch_metrics_df)

    # save epoch metrics
    epoch_metrics_df.to_csv(f"{OUTPUT_ROUTE}/metrics/epoch_metrics.csv", index=False)

    # save train max accuracy iteration
    train_max_accuracy_iteration = train_iterations_metrics_df[
        train_iterations_metrics_df["accuracy"] == train_iterations_metrics_df["accuracy"].max()]
    train_max_accuracy_iteration.to_csv('train_max_accuracy_iteration.csv', mode='a', header=False)

    # save test max accuracy iteration
    test_max_accuracy_iteration = test_iterations_metrics_df[
        test_iterations_metrics_df["accuracy"] == test_iterations_metrics_df["accuracy"].max()]
    test_max_accuracy_iteration.to_csv('test_max_accuracy_iteration.csv', mode='a', header=False)


    #### PLOTTING ####
    make_confusion_matrix(OUTPUT_ROUTE, y_true, y_pred, classes)

    plot_moving_average(OUTPUT_ROUTE)
    plot_train_log(OUTPUT_ROUTE)
    plot_loss(OUTPUT_ROUTE)
    plot_accuracy(OUTPUT_ROUTE)




if __name__ == "__main__":
    # train_iteration = 1
    # test_iteration = 1
    best_acc = 0  # best test accuracy
    my_app()

