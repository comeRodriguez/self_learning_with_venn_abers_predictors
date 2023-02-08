"""This file contains the Inductive Venn Abers algorithm.

Self Learning using Venn Abers predictors

@Côme Rodriguez, @Vitor Bordini, @Sébastien Destercke and @Benjamin Quost
"""

from typing import Union, List
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from neural_net import SimpleNeuralNet, SimpleNeuralNetCredal

def venn_abers_pytorch(
    trained_classifier: Union[SimpleNeuralNet, SimpleNeuralNetCredal],
    calibration_features: torch.Tensor,
    calibration_labels: pd.Series,
    test_instance: torch.Tensor
)-> List:
    """Compute the Inductive Venn Abers algorithm to obtain a credal set of probabilities
    for a test instance

    Args:
        trained_classifier (Union[SimpleNeuralNet, SimpleNeuralNetCredal]): A binary classifier
            train on some data and having a 'predict_probas' method. Here, implementation only
            for one of the classifier in 'neural_net.py'.
        calibration_features (torch.Tensor): calibration features set
        calibration_labels (pd.Series): corresponding labels for calibration features
        test_instance (torch.Tensor): new features without a corresponding label

    Returns:
        List: credal set [p0, p1]
    """
    calibrations = trained_classifier.predict_probas(
        calibration_features,
        as_numpy=True,
        from_numpy=False
    )
    predictions = trained_classifier.predict_probas(test_instance, as_numpy=True, from_numpy=False)
    interval = []
    scores = pd.DataFrame()
    scores["s"] = calibrations[:, 0]
    scores["y"] = calibration_labels.to_numpy()
    score = pd.DataFrame()
    score["s"] = predictions[:, 0]
    interval = []

    for i in [0, 1]:
        score["y"] = i
        train = pd.concat([scores, score], ignore_index=True)
        g = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        g.fit(train["s"], train["y"])
        interval.append(g.predict(score["s"])[0])
    interval = np.array(interval)
    return interval
