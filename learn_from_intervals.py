"""This file is the first implementation for the
learning from intervals of probabilities 
"""

import pprint
from typing import List
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import rel_entr

def transform_to_interval(variable: str) -> np.ndarray:
    """
    Transform a hard label into an interval of probabilities.
    If the label is 0, the created interval will be in the range of
    [0, 0.49], else it will be in the range of [0.5, 1]
    
    Args:
        variable (int): variable to transform
    
    Returns:
        (np.ndarray): created interval 
    """
    if variable == "B":
        interval = np.sort(
            np.round(
                np.random.uniform(
                    low=0.0,
                    high=0.49,
                    size=(2,)
                ),
                decimals=2
            )
        )
    else:
        interval = np.sort(
            np.round(
                np.random.uniform(
                    low=0.5,
                    high=1.0,
                    size=(2,)
                ),
                decimals=2
            )
        )
    return interval

def get_right_probability_distribution(probas: List, p_hat: float) -> float:
    """Get the right distribution to use for the Kullback-Leibler divergence 

    Args:
        probas (List): _description_
        p_hat (float): _description_

    Returns:
        float: _description_
    """
    if p_hat < probas[0]:
        return probas[0]
    elif p_hat > probas[1]:
        return probas[1]
    else:
        return p_hat

def kullback_leibler_divergence(proba, p_hat):
    """Kullback-Leibler divergence

    Args:
        proba (_type_): _description_
        p_hat (_type_): _description_

    Returns:
        _type_: _description_
    """
    return rel_entr(proba, p_hat) + rel_entr(1-proba, 1-p_hat)

class CustomLogisticRegression():
    """Logistic regression with labels in form of probability intervals
    """
    def __init__(self, number_iteration: int, learning_rate: float) -> None:
        """Initialisation of the class

        Args:
            number_iteration (int): _description_
            learning_rate (float): _description_
        """
        self.lr = LogisticRegression(max_iter=1)
        self.number_iterations = number_iteration
        self.weights = None
        self.biais = None
        self.learning_rate = learning_rate
        self.losses = {}
        self.accuracies = {}
        
    def update_parameters(self, x_train, p, p_hat):
        """Update weights and biais

        Args:
            x_train (_type_): _description_
            p (_type_): _description_
            p_hat (_type_): _description_
        """
        self.weights = self.weights - self.learning_rate*np.dot(
            (p - p_hat).T,
            x_train
        )
        self.biais = self.biais - self.learning_rate*(1/len(x_train)*np.sum(
                p_hat-p
            )
        )
    def predict(self, predictors):
        """Predict p(Y=1|X=x)

        Args:
            predictors (_type_): _description_

        Returns:
            _type_: _description_
        """
        linear_part = np.dot(self.weights, predictors.T) + self.biais
        predictions = 1/(1+np.exp(-linear_part))
        return predictions
    
    def fit(self, x_train, y_train):
        """Train the model

        Args:
            x_train (_type_): _description_
            y_train (_type_): _description_

        Returns:
            _type_: _description_
        """
        train_df = x_train.copy()
        train_response = pd.DataFrame(y_train.copy())
        
        train_response["hypotetic_label"] = train_response.apply(
            lambda x: 1 if x["interval"][1] >= 0.5 else 0,
            axis=1
        )
        self.lr.fit(train_df, train_response["hypotetic_label"])
        self.weights = self.lr.coef_
        self.biais = self.lr.intercept_
        train_response["p_hat"] = self.lr.predict_proba(train_df)[:, 1]
        train_response["proba_used"] = train_response.apply(
            lambda x: get_right_probability_distribution(
                x["interval"],
                x["p_hat"]
            ),
            axis=1
        )
        self.losses[0] = np.mean(
            kullback_leibler_divergence(train_response["proba_used"], train_response["p_hat"]
            )
        )
        self.update_parameters(train_df, train_response["proba_used"], train_response["p_hat"])
        accuracy = len(
            train_response[
                train_response["p_hat"]==train_response["proba_used"]
            ]
        )*100/len(train_response)
        self.accuracies[0] = accuracy
        for i in range(1, self.number_iterations):
            train_response["p_hat"] = self.predict(train_df)[0, :]
            train_response["proba_used"] = train_response.apply(
                lambda x: get_right_probability_distribution(
                    x["interval"],
                    x["p_hat"]
                ),
                axis=1
            )
            self.losses[i] = np.mean(
                kullback_leibler_divergence(train_response["proba_used"], train_response["p_hat"]
                )
            )
            self.update_parameters(train_df, train_response["proba_used"], train_response["p_hat"])
            accuracy = len(
                train_response[
                    train_response["p_hat"]==train_response["proba_used"]
                ]
            )*100/len(train_response)
            self.accuracies[i] = accuracy
            
        return self.losses


if __name__ == "__main__":
    df = pd.read_csv("SynthPara_n1000_p2.csv")
    df["label"] = df["z"].apply(lambda x: 1 if x=="A" else 0)
    df["interval"] = df["z"].apply(transform_to_interval)
    test = CustomLogisticRegression(number_iteration=200, learning_rate=0.0001)
    view = test.fit(x_train=df[["X1", "X2"]], y_train=df["interval"])
    pprint.pprint(view)
    pprint.pprint(test.accuracies)
