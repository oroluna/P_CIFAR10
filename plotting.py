import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_train_log(route):
    train_log = pd.read_csv(f'{route}/metrics/train_iteration_metrics.csv')
    test_log  = pd.read_csv(f'{route}/metrics/test_iteration_metrics.csv')

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(train_log.epoch, train_log.loss, linewidth=2, label='Train loss', color='b', alpha=0.3)
    line12 = ax1.plot(test_log.epoch, test_log.loss, marker='o', markersize=12, linestyle='', label='Test loss', color='blue')
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)

    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log.epoch, train_log.accuracy, linewidth=2, label='Train accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(test_log.epoch, test_log.accuracy, marker='o', markersize=12, linestyle='', label='Test accuracy', color='red')

    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)

    # added these four lines
    lines  = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg    = ax1.legend(lines, labels, fontsize=16, loc=5)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')
    

    plt.grid()
    if not os.path.exists(f"{route}/artefacts"):
        os.mkdir(f"{route}/artefacts")
        
    plt.savefig(f"{route}/artefacts/train_log.jpg", bbox_inches='tight')
   


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_moving_average(route):

    train_log = pd.read_csv(f'{route}/metrics/train_iteration_metrics.csv')
    test_log  = pd.read_csv(f'{route}/metrics/test_iteration_metrics.csv')

    epoch    = moving_average(np.array(train_log.epoch),40)
    accuracy = moving_average(np.array(train_log.accuracy),40)
    loss     = moving_average(np.array(train_log.loss),40)

    fig, ax1 = plt.subplots(figsize=(12,8),facecolor='w')
    line11 = ax1.plot(train_log.epoch, train_log.loss, linewidth=2, label='Loss', color='b', alpha=0.3)
    line12 = ax1.plot(epoch, loss, label='Loss (averaged)', color='blue')
    ax1.set_xlabel('Epoch',fontweight='bold',fontsize=24,color='black')
    ax1.tick_params('x',colors='black',labelsize=18)
    ax1.set_ylabel('Loss', fontsize=24, fontweight='bold',color='b')
    ax1.tick_params('y',colors='b',labelsize=18)

    ax2 = ax1.twinx()
    line21 = ax2.plot(train_log.epoch, train_log.accuracy, linewidth=2, label='Accuracy', color='r', alpha=0.3)
    line22 = ax2.plot(epoch, accuracy, label='Accuracy (averaged)', color='red')

    ax2.set_ylabel('Accuracy', fontsize=24, fontweight='bold',color='r')
    ax2.tick_params('y',colors='r',labelsize=18)
    ax2.set_ylim(0.,1.0)

    # added these four lines
    lines  = line11 + line12 + line21 + line22
    labels = [l.get_label() for l in lines]
    leg    = ax1.legend(lines, labels, fontsize=16, loc=5)
    leg_frame = leg.get_frame()
    leg_frame.set_facecolor('white')

    plt.grid()
    if not os.path.exists(f"{route}/artefacts"):
        os.mkdir(f"{route}/artefacts")

    plt.savefig(f"{route}/artefacts/moving_average.jpg", bbox_inches='tight')
   


def make_confusion_matrix(route, y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):

    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize our confusion matrix
    n_classes = cm.shape[0]

    # Let's prettify it
    fig, ax = plt.subplots(figsize=figsize)
    # Create a matrix plot
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels to be classes
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted Label",
            ylabel="True Label",
            #xticks=np.arange(n_classes),
            #yticks=np.arange(n_classes),
            xticklabels=labels,
            yticklabels=labels)

    # Set x-axis labels to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min()) / 2

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

    if not os.path.exists(f"{route}/artefacts"):
        os.mkdir(f"{route}/artefacts")

    plt.savefig(f"{route}/artefacts/confusion_matrix.jpg", bbox_inches='tight')

def plot_loss(route):
    train_log = pd.read_csv(f'{route}/metrics/train_iteration_metrics.csv')
    test_log = pd.read_csv(f'{route}/metrics/test_iteration_metrics.csv')
    # Plot loss
    train_epochs = range(len(train_log.loss))
    test_epochs = np.linspace(0, len(train_log.loss), len(test_log.loss))
    plt.figure()
    plt.plot(train_epochs, train_log.loss, label='training_loss')
    plt.plot(test_epochs, test_log.loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if not os.path.exists(f"{route}/artefacts"):
        os.mkdir(f"{route}/artefacts")

    plt.savefig(f"{route}/artefacts/loss_artefact.jpg", bbox_inches='tight')

def plot_accuracy(route):
    train_log = pd.read_csv(f'{route}/metrics/train_iteration_metrics.csv')
    test_log = pd.read_csv(f'{route}/metrics/test_iteration_metrics.csv')
    # Plot loss
    train_epochs = range(len(train_log.accuracy))
    test_epochs = np.linspace(0, len(train_log.accuracy), len(test_log.loss))
    plt.figure()
    plt.plot(train_epochs, train_log.accuracy, label='training_accuracy')
    plt.plot(test_epochs, test_log.accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    if not os.path.exists(f"{route}/artefacts"):
        os.mkdir(f"{route}/artefacts")

    plt.savefig(f"{route}/artefacts/accuracy_artefact.jpg", bbox_inches='tight')





    

    

    
