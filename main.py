'''Train CIFAR10 with PyTorch.'''


# Locals
from plotting import *
from confusion_matrix import ConfusionMatrix


import os
from omegaconf import DictConfig, ListConfig, OmegaConf

import hydra
from hydra import utils
import mlflow
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from engine.utils.utils import progress_bar



def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


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
            steps = epoch * len(trainloader) + batch_idx
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
            accuracy = float(correct / total)

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            mlflow.log_metric("loss", loss.item(), step=steps)
            mlflow.log_metric("acc", accuracy, step=steps)

            loss = train_loss / (batch_idx + 1)
            accuracy = correct / total





            train_iterations_results["loss"].append(loss)
            train_iterations_results["accuracy"].append(accuracy)
            train_iterations_results["epoch"].append(epoch)

        train_loss = train_loss / len(trainloader)
        train_acc = correct / len(trainloader)



        return train_loss, accuracy

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                steps = epoch * len(testloader) + batch_idx
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                accuracy = float(correct / total)

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                mlflow.log_metric("loss", loss.item(), step=steps)
                mlflow.log_metric("acc", accuracy, step=steps)

                loss = test_loss / (batch_idx + 1)
                accuracy = correct / total

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

    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        for epoch in range(start_epoch, start_epoch + cfg.params.epoch_count):
            log_params_from_omegaconf_dict(cfg)
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
    cm = ConfusionMatrix(net, testloader, classes)
    y_pred = cm.y_pred
    y_true = cm.y_true
    print('CM', cm.cm)


    #### SAVE IN TO CSV ####

    if not os.path.exists(f"{OUTPUT_ROUTE}/metrics"):
        os.mkdir(f"{OUTPUT_ROUTE}/metrics")

    # create train iterations metrics dataframe
    train_iterations_metrics_df = pd.DataFrame(train_iterations_results)

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

    # add train_mean_accuracy_iteration per epoch
    epoch_metrics_df['train_mean_accuracy_iteration'] = train_iterations_metrics_df.groupby("epoch")["accuracy"].mean()

    # add train_max_loss_iteration per epoch
    epoch_metrics_df['train_max_loss_iteration'] = train_iterations_metrics_df.groupby("epoch")["loss"].max()

    # add train_min_loss_iteration per epoch
    epoch_metrics_df['train_min_loss_iteration'] = train_iterations_metrics_df.groupby("epoch")["loss"].min()

    # add train_mean_loss_iteration per epoch
    epoch_metrics_df['train_mean_loss_iteration'] = train_iterations_metrics_df.groupby("epoch")["loss"].mean()

    # add test_max_accuracy_iteration per epoch
    epoch_metrics_df['test_max_accuracy_iteration'] = test_iterations_metrics_df.groupby("epoch")["accuracy"].max()

    # add test_min_accuracy_iteration per epoch
    epoch_metrics_df['test_min_accuracy_iteration'] = test_iterations_metrics_df.groupby("epoch")["accuracy"].min()

    # add test_mean_accuracy_iteration per epoch
    epoch_metrics_df['test_mean_accuracy_iteration'] = test_iterations_metrics_df.groupby("epoch")["accuracy"].mean()

    # add test_max_loss_iteration per epoch
    epoch_metrics_df['test_max_loss_iteration'] = test_iterations_metrics_df.groupby("epoch")["loss"].max()

    # add test_min_loss_iteration per epoch
    epoch_metrics_df['test_min_loss_iteration'] = test_iterations_metrics_df.groupby("epoch")["loss"].min()

    # add test_min_loss_iteration per epoch
    epoch_metrics_df['test_mean_loss_iteration'] = test_iterations_metrics_df.groupby("epoch")["loss"].mean()

    # save epoch metrics
    epoch_metrics_df.to_csv(f"{OUTPUT_ROUTE}/metrics/epoch_metrics.csv", index=False)

    # save train max accuracy iteration
    train_max_accuracy_iteration = train_iterations_metrics_df[
        train_iterations_metrics_df["accuracy"] == train_iterations_metrics_df["accuracy"].max()].iloc[-1:]
    train_max_accuracy_iteration.index.name = 'iteration'
    train_max_accuracy_iteration.to_csv(f"{OUTPUT_ROUTE}/metrics/T_train_max_accuracy_iteration.csv", mode='a')

    # save test max accuracy iteration
    test_max_accuracy_iteration = test_iterations_metrics_df[
        test_iterations_metrics_df["accuracy"] == test_iterations_metrics_df["accuracy"].max()].iloc[-1:]
    test_max_accuracy_iteration.index.name = 'iteration'
    test_max_accuracy_iteration.to_csv(f"{OUTPUT_ROUTE}/metrics/T_test_max_accuracy_iteration.csv", mode='a')

    # creating the metrics dataframe
    metrics_df = pd.DataFrame(data=cm.get_metrics(),
                      index=cm.classes,
                      columns=cm.metrics)

    # save metrics
    metrics_df.to_csv(f"{OUTPUT_ROUTE}/metrics/metrics.csv")
    print(metrics_df)

    #### PLOTTING ####
    # make_confusion_matrix(OUTPUT_ROUTE, cm.cm, classes)
    make_freq_confusion_matrix(OUTPUT_ROUTE, cm.cm, classes)
    make_precent_confusion_matrix(OUTPUT_ROUTE, cm.cm, classes)
    plot_moving_average(OUTPUT_ROUTE)
    plot_train_log(OUTPUT_ROUTE)
    plot_loss(OUTPUT_ROUTE)
    plot_accuracy(OUTPUT_ROUTE)

    for i in range(len(classes)):
        cf_matrix = cm.class_confusion_matrix(i)
        make_class_confusion_matrix(OUTPUT_ROUTE, cf_matrix, i, classes)
    return test_acc

if __name__ == "__main__":
    # train_iteration = 1
    # test_iteration = 1
    best_acc = 0  # best test accuracy
    my_app()
