"""This file contains usefull functions
"""

from typing import List
import pandas as pd
from scipy.special import rel_entr

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
