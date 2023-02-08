"""This file contains usefull functions

Self Learning using Venn Abers predictors

@Côme Rodriguez, @Vitor Bordini, @Sébastien Destercke and @Benjamin Quost
"""

from typing import List, Tuple
import pandas as pd
from scipy.special import rel_entr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable


class MyDataset(torch.utils.data.Dataset):
    """Custom dataset for pytorch model
    """
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, credal: bool=False):
        """_summary_

        Args:
            x_train (pd.DataFrame): training features
            y_train (pd.Series): training label
            credal (bool, optional): if y_train is credal (i.e an interval)
                or not. Defaults to False.
        """
        x=x_train.values
        y=y_train.values
        self.X_train=torch.tensor(x,dtype=torch.float32)
        if not credal:
            self.Y_train=y.astype("float32")
        else:
            self.Y_train=y
    def __len__(self) -> int:
        """Get length

        Returns:
            int: length of y_train
        """
        return len(self.Y_train)
   
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item

        Args:
            idx (int): indice of batch

        Returns:
            batch of X_train and corresponding batch of Y_train
        """
        return self.X_train[idx], self.Y_train[idx]


def KLD_loss_credal(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the KLD Loss D_KL(y_true || y_pred) for binary classification
    with credal sets as Y_train

    Args:
        y_true (torch.Tensor): Y_train in form of credal sets 
        y_pred (torch.Tensor): probabilities output by a binary classifier

    Returns:
        torch.Tensor: KLD Loss (with requires_grad=True for gradient descent)
    """    
    y_true_inf = y_true[:, 0].reshape(-1, 1)
    y_true_sup = y_true[:, 1].reshape(-1, 1)
    used_probas = torch.zeros(y_true.shape[0], 1)
    used_probas[y_pred <= y_true_inf] = y_true_inf[y_pred <= y_true_inf]
    used_probas[y_pred >= y_true_sup] = y_true_sup[y_pred >= y_true_sup]
    used_probas[(y_pred >= y_true_inf) & (y_pred <= y_true_sup)
                ] = y_pred[(y_pred >= y_true_inf) & (y_pred <= y_true_sup)]
    loss = torch.mean(used_probas* torch.log(used_probas/y_pred) + (1-used_probas)*torch.log((1-used_probas)/(1-y_pred)))
    return loss


def KLD_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Compute the KLD Loss D_KL(y_true || y_pred) for binary classification
    with soft labels as Y_train

    Args:
        y_true (torch.Tensor): Y_train in form of soft labels
        y_pred (torch.Tensor): probabilities output by a binary classifier

    Returns:
        torch.Tensor: KLD Loss (with requires_grad=True for gradient descent)
    """
    loss = torch.mean(y_true* torch.log(y_true/y_pred) + (1-y_true)*torch.log((1-y_true)/(1-y_pred)))
    return loss



def get_right_probability_distribution(
    probability_interval: List[float],
    estimated_probability: float
) -> float:
    """Get the right probability distribution to use in the loss function
    when dealing with intervals of probabilities labels. The distribution
    (i.e the probability p(Y=1|X=x)) to use is determined with the following:
    If estimated_probability < inf(probability_interval):
        return inf(probability_interval)
    Else if estimated_probability > sup(probability_interval):
        return sup(probability_interval)
    Else return estimated_probability

    Args:
        probability_interval (List[float]): interval of probabilities (representing
            the uncertainty about the label) we want to learn from
        estimated_probability (float): probability p(Y=1|X=x) estimated by a model

    Returns:
        (float): probability to use in the loss function
    """
    if estimated_probability < probability_interval[0]:
        return probability_interval[0]
    if estimated_probability > probability_interval[1]:
        return probability_interval[1]
    return estimated_probability

def kullback_leibler_divergence(probability: float, estimated_probability: float) -> float:
    """Compute the Kullback Leibler divergence between to distributions
    of probabilities. The formula is the following:
        D_kl(p||q) = p * ln(p/q) + (1-p) * ln((1-p)/(1-q))

    Args:
        probability (float): observed probability
        estimated_probability (float): probability estimated by a model

    Returns:
        (float): Kullback Leibler divergence between probability and
            estimated_probability
    """
    return rel_entr(probability, estimated_probability)\
        + rel_entr(1-probability, 1-estimated_probability)

def get_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute the accuracy score between y_true and y_pred

    Args:
        y_true (pd.Series): true labels
        y_pred (pd.Series): predicted labels

    Returns:
        (float): accuracy score
    """
    accuracy = len(
        y_pred[
            y_pred==y_true
        ]
    )/len(y_pred)
    return accuracy

def plot_decision_boundary(dataset, labels, model, steps=1000, color_map='Paired'):
    color_map = plt.get_cmap(color_map)
    # Define region of interest by data limits
    print()
    xmin, xmax = dataset.to_numpy()[:, 0].min() - 1, dataset.to_numpy()[:, 0].max() + 1
    ymin, ymax = dataset.to_numpy()[:, 1].min() - 1, dataset.to_numpy()[:, 1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    model.eval()
    labels_predicted = model(Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()))

    # Plot decision boundary in region of interest
    labels_predicted = [0 if value <= 0.5 else 1 for value in labels_predicted.detach().numpy()]
    z = np.array(labels_predicted).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.contourf(xx, yy, z, cmap=color_map, alpha=0.2)
    
    return fig, ax

