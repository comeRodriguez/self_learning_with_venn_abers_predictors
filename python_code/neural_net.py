"""This file contains declaration of 2 neural network classes (one for hard
and soft labels, the other for hard and credal labels).

Self Learning using Venn Abers predictors

@Côme Rodriguez, @Vitor Bordini, @Sébastien Destercke and @Benjamin Quost
"""

from typing import Union
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torch
from torch import nn
from collections import OrderedDict
import tensorflow as tf
from utils import MyDataset, KLD_loss_credal, KLD_loss


class SimpleNeuralNet(nn.Module):
    """Simple Neural Network (with 1 hidden layer) used for tests.
    Only use for binary classification and for hard or soft labels
    """
    def __init__(self, clipping_value: float, n_input_units: int, n_hidden_units: int):
        """Initialisation of the neural network

        Args:
            clipping_value (float): clipping value to avoid exploding gradient
            n_input_units (int): number of neurons for the input layer
            n_hidden_units (int): number of neurons for the hidden layer
        """
        super().__init__()
        self.input_layer = nn.Sequential(
            OrderedDict(
                {
                    'linear': nn.Linear(in_features=n_input_units, out_features=n_hidden_units),
                    'relu': nn.ReLU(inplace=True),
                }
            )
        )
        self.hidden_layer = nn.Linear(in_features=n_hidden_units, out_features=1)
        self.output_layer = nn.Sigmoid()
        self.losses = {}
        self.accuracies = {}
        self.clipping_value = clipping_value

    def forward(self, x_train: torch.Tensor) -> torch.Tensor:
        """Computes the forward propagation

        Args:
            x (torch.Tensor): Features used

        Returns:
            torch.Tensor: probabilities output by the neural network
        """
        y_pred = self.input_layer(x_train)
        y_pred = self.hidden_layer(y_pred)
        y_pred = self.output_layer(y_pred)
        return y_pred

    def __initiate_loss_and_accuracy_dicts(self, n_epochs: int):
        """initiate loss and accuracy dictionnaries to keep track of performances

        Args:
            n_epochs (int): number of epochs during learning
        """
        for i in range(n_epochs):
            self.losses[i] = 0
            self.accuracies[i] = 0

    def fit(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int,
        learning_rate: float,
        verbose=True,
        soft=False
    ):
        """Fit the neural network to the data

        Args:
            x_train (pd.DataFrame): Learning features
            y_train (pd.Series): Learning labels
            epochs (int): number of epochs for training
            learning_rate (float): learning rate for gradient descent
            verbose (bool, optional): Print the training informations as tensorflow does.
                Defaults to True.
            soft (bool, optional): True if y_train is composed of soft labels, else False.
                Defaults to False.
        """
        self.__initiate_loss_and_accuracy_dicts(n_epochs=epochs)
        myDs=MyDataset(x_train=x_train, y_train=y_train)
        train_loader=torch.utils.data.DataLoader(myDs,batch_size=10,shuffle=False)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            n_batches = len(train_loader)
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}')
                pbar = tf.keras.utils.Progbar(target=n_batches)
            for idx, batch in enumerate(train_loader):
                x, y = batch
                optimizer.zero_grad()
                outputs = self(x)
                outputs = torch.where(outputs>0.999, outputs-0.001, outputs)
                outputs = torch.where(outputs<0.001, outputs+0.001, outputs)
                if not soft:
                    loss = nn.BCELoss()
                    loss_value = loss(outputs, y.reshape(-1, 1))
                else:
                    loss_value = KLD_loss(y.reshape(-1, 1), outputs)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
                loss_value.backward()
                optimizer.step()
                if not soft:
                    accuracy = accuracy_score(
                        y_true=y,
                        y_pred=outputs.reshape(-1).detach().numpy().round()
                    )
                    self.accuracies[epoch] += accuracy
                self.losses[epoch] += loss_value.detach().numpy()
                if verbose:
                    pbar.update(
                        idx,
                        values=[
                            ("loss", loss_value.detach().numpy()),
                            ("accuracy", accuracy)
                        ]
                    )
            self.losses[epoch] = self.losses[epoch]/n_batches
            self.accuracies[epoch] = self.accuracies[epoch]/n_batches
            if verbose:
                pbar.update(n_batches, values=None)

    def predict_probas(
        self,
        x: Union[pd.DataFrame, torch.Tensor],
        as_numpy=False,
        from_numpy=True
    ) -> Union[torch.Tensor, np.ndarray]:
        """Compute forward propagation after training to obtain probabilities P(Y=1|X=x)

        Args:
            x (Union[pd.DataFrame, torch.Tensor]): Input features
            as_numpy (bool, optional): If the output type need to be a numpy array.
                Defaults to False.
            from_numpy (bool, optional): If the input features are of type numpy.ndarray.
                Defaults to True.

        Returns:
            Union[torch.Tensor, np.ndarray]: probabilities P(Y=1|X=x) estimated by the
                neural network
        """
        if from_numpy:
            x = torch.tensor(x, dtype=torch.float32)
        outputs = self(x)
        if as_numpy:
            outputs = outputs.detach().numpy()
        return outputs



class SimpleNeuralNetCredal(nn.Module):
    """Simple Neural Network (with 1 hidden layer) used for tests.
    Only use for binary classification and for hard or credal labels
    """

    def __init__(self, clipping_value: float, n_input_units: int, n_hidden_units: int):
        """Initialisation of the neural network

        Args:
            clipping_value (float): clipping value to avoid exploding gradient
            n_input_units (int): number of neurons for the input layer
            n_hidden_units (int): number of neurons for the hidden layer
        """
        super().__init__()
        self.input_layer = nn.Sequential(
            OrderedDict(
                {
                    'linear': nn.Linear(in_features=n_input_units, out_features=n_hidden_units),
                    'relu': nn.ReLU(inplace=True),
                }
            )
        )
        self.hidden_layer = nn.Linear(in_features=n_hidden_units, out_features=1)
        self.output_layer = nn.Sigmoid()
        self.losses = {}
        self.accuracies = {}
        self.clipping_value = clipping_value

    def forward(self, x_train: torch.Tensor) -> torch.Tensor:
        """Computes the forward propagation

        Args:
            x_train (torch.Tensor): Features used

        Returns:
            torch.Tensor: probabilities output by the neural network
        """
        y_pred = self.input_layer(x_train)
        y_pred = self.hidden_layer(y_pred)
        y_pred = self.output_layer(y_pred)
        return y_pred

    def __initiate_loss_and_accuracy_dicts(self, n_epochs: int):
        """initiate loss and accuracy dictionnaries to keep track of performances

        Args:
            n_epochs (int): number of epochs during learning
        """
        for i in range(n_epochs):
            self.losses[i] = 0
            self.accuracies[i] = 0

    def fit(self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int,
        learning_rate: float,
        verbose=True,
        credal=False
    ):
        """Fit the neural network to the data

        Args:
            x_train (pd.DataFrame): Learning features
            y_train (pd.Series): Learning labels
            epochs (int): number of epochs for training
            learning_rate (float): learning rate for gradient descent
            verbose (bool, optional): Print the training informations as tensorflow does.
                Defaults to True.
            credal (bool, optional): True if y_train is composed of credal labels, else False.
                Defaults to False.
        """
        self.__initiate_loss_and_accuracy_dicts(n_epochs=epochs)
        myDs=MyDataset(x_train=x_train, y_train=y_train, credal=credal)
        train_loader=torch.utils.data.DataLoader(myDs,batch_size=10,shuffle=False)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            n_batches = len(train_loader)
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}')
                pbar = tf.keras.utils.Progbar(target=n_batches)
            for idx, batch in enumerate(train_loader):
                x, y = batch
                optimizer.zero_grad()
                outputs = self(x)

                outputs = torch.where(outputs>0.999, outputs-0.001, outputs)
                outputs = torch.where(outputs<0.001, outputs+0.001, outputs)
                if not credal:
                    loss = nn.BCELoss()
                    loss_value = loss(outputs, y.reshape(-1, 1))
                else:
                    loss_value = KLD_loss_credal(y, outputs)
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
                loss_value.backward()
                optimizer.step()
                if not credal:
                    accuracy = accuracy_score(
                        y_true=y,
                        y_pred=outputs.reshape(-1).detach().numpy().round()
                    )
                    self.accuracies[epoch] += accuracy
                self.losses[epoch] += loss_value.detach().numpy()
                if verbose and not credal:
                    pbar.update(
                        idx,
                        values=[
                            ("loss", loss_value.detach().numpy()),
                            ("accuracy", accuracy)
                        ]
                    )
                elif verbose and credal:
                    pbar.update(
                        idx,
                        values=[
                            ("loss", loss_value.detach().numpy()),
                        ]
                    )
            self.losses[epoch] = self.losses[epoch]/n_batches
            self.accuracies[epoch] = self.accuracies[epoch]/n_batches
            if verbose:
                pbar.update(n_batches, values=None)

    def predict_probas(self, x, as_numpy=False, from_numpy=True):
        """Compute forward propagation after training to obtain probabilities P(Y=1|X=x)

        Args:
            x (Union[pd.DataFrame, torch.Tensor]): Input features
            as_numpy (bool, optional): If the output type need to be a numpy array.
                Defaults to False.
            from_numpy (bool, optional): If the input features are of type numpy.ndarray.
                Defaults to True.

        Returns:
            Union[torch.Tensor, np.ndarray]: probabilities P(Y=1|X=x) estimated by the
                neural network
        """
        if from_numpy:
            x = torch.tensor(x,dtype=torch.float32)
        outputs = self(x)
        if as_numpy:
            outputs = outputs.detach().numpy()
        return outputs
