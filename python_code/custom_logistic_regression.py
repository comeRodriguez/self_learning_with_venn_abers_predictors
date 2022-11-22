"""This file is the first implementation for the
learning from intervals of probabilities
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import (
    get_right_probability_distribution,
    kullback_leibler_divergence,
    get_accuracy
)

class CustomLogisticRegression():
    """Custom logistic regression model. This model use the Kullback Leibler
    divergence as loss function and the label are intervals of probabilities
    """
    def __init__(self, number_iteration: int, learning_rate: float) -> None:
        """Constructor of the class

        Args:
            number_iteration (int): number of iteration during training
            learning_rate (float): learning_rate to use for gradient descent
        """
        self.classical_lr = LogisticRegression(max_iter=1)
        self.number_iterations = number_iteration
        self.weights = None
        self.biais = None
        self.learning_rate = learning_rate
        self.losses = {}
        self.accuracies = {}

    def update_parameters(self, x_train: pd.DataFrame, proba: pd.Series, p_hat: pd.Series):
        """Update the weights and biais of the logistic regression
        using the learning rate and the gradient of the weights and biais

        Args:
            x_train (pd.DataFrame): Predictors
            proba (pd.Series): probabilities observed
            p_hat (pd.Series): probabilities estimated by the model
        """
        self.weights -= self.learning_rate*np.dot(
            x_train.T,
            (p_hat - proba)
        )
        self.biais -= self.learning_rate*(1/len(x_train)*np.sum(p_hat-proba))

    def predict(self, predictors: pd.DataFrame) -> pd.Series:
        """Predict the probabilities p(Y=1|X=x) for each individual

        Args:
            predictors (pd.DataFrame): predictors (i.e X)

        Returns:
            (pd.Series): probabilities estimated by the model
        """
        linear_part = np.dot(self.weights, predictors.T) + self.biais
        estimated_probabilities = 1/(1+np.exp(-linear_part))
        return estimated_probabilities

    def forward_propagation(
        self,
        x_train: pd.DataFrame,
        first_iteration: bool,
        informative: bool,
        y_train: pd.Series,
    ) ->  Tuple[pd.DataFrame, float]:
        """One forward propagation step of the model during training

        Args:
            x_train (pd.DataFrame): predictors
            first_iteration (bool): True if we are in the first iteration
                (usefull for weights and biais initialization)
            informative (bool): if the intervals are informative or not (i.e [0, 1] or not)
            y_train (pd.Series): intervals of probilities to learn from

        Returns:
            (Tuple[pd.DataFrame, float]): Predicted probabilities and loss value
        """
        train_response = pd.DataFrame(y_train.copy())
        train_response.columns = ["interval"]
        if first_iteration:
            if informative:
                hypotetic_label = y_train.apply(lambda x: 1 if x[1] >= 0.5 else 0)

            else:
                hypotetic_label = [np.random.choice([0, 1]) for _ in range(len(train_response))]
            self.classical_lr.fit(x_train, hypotetic_label)
            self.weights = self.classical_lr.coef_
            self.biais = self.classical_lr.intercept_
            train_response["p_hat"] = self.classical_lr.predict_proba(x_train)[:, 1]
        else:
            train_response["p_hat"] = self.predict(predictors=x_train)[0, :]
        train_response["y_hat"] = train_response["p_hat"].apply(lambda x: 1 if x >= 0.5 else 0)
        train_response["proba_used"] = train_response.apply(
            lambda x: get_right_probability_distribution(
                x["interval"],
                x["p_hat"]
            ),
            axis=1
        )
        loss = np.mean(
            kullback_leibler_divergence(
                train_response["proba_used"],
                train_response["p_hat"]
            )
        )

        return train_response, loss


    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, informative=True):
        """Train the model using x_train and y_train

        Args:
            x_train (pd.DataFrame): dataframe containing the predictors
            y_train (pd.DataFrame): intervals of probabilities we want to learn
                from
            informative (bool): True if the intervals are not all [0, 1]
        """
        if informative:
            hypotetic_label = y_train.apply(lambda x: 1 if x[1] >= 0.5 else 0)

        else:
            hypotetic_label = [np.random.choice([0, 1]) for _ in range(len(y_train))]
        for i in range(0, self.number_iterations):
            if i == 0:
                train_response, self.losses[i] = self.forward_propagation(
                    x_train=x_train,
                    first_iteration=True,
                    informative=informative,
                    y_train=y_train
                )
            else:
                train_response, self.losses[i] = self.forward_propagation(
                    x_train=x_train,
                    first_iteration=False,
                    informative=informative,
                    y_train=y_train
                )
            self.update_parameters(x_train, train_response["proba_used"], train_response["p_hat"])
            self.accuracies[i] = get_accuracy(
                y_true=hypotetic_label,
                y_pred=train_response["y_hat"]
                )
