
import plotly.graph_objs as go
from clearml import Task, Logger
from numpy import random
import numpy as np
import pandas as pd

def clearmlplot(epochs, epoch_train_loss, epoch_train_acc, name):
    # report line plot
    logger = Logger.current_logger()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=epoch_train_loss, name="train_loss"))
    fig.add_trace(go.Scatter(x=epochs, y=epoch_train_acc, name="train_acc"))
    # Log the plot to ClearML
    logger.report_plotly(title="Train Accuracy and loss", series=name, iteration=0, figure=fig)
