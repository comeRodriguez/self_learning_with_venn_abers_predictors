"""This file contains 3 classes for 3 different Self Learning approaches:
    - classical one (SelfLearning)
    - with soft labels (SelfLearningWithSoft)
    - using Venn Abers predictors (SelfLearningUsingVennAbers)

Self Learning using Venn Abers predictors

@Côme Rodriguez, @Vitor Bordini, @Sébastien Destercke and @Benjamin Quost
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from neural_net import SimpleNeuralNet, SimpleNeuralNetCredal
import torch
from venn_abers import venn_abers_pytorch

class SelfLearning():
    """Classical Self Learning approach: adding a batch of new labeled
    data into the training set and we train the classifier on.
    The classifier is a neural network with one hidden layer.
    """
    def __init__(
        self,
        known_x_train: pd.DataFrame,
        known_y_train: pd.Series,
        unknown_x_train: pd.DataFrame,
        model_convergence_epochs: int,
        model_learning_rate: float,
        n_input_unit: int,
        n_hidden_units: int,
        verbose: bool,
        random_seed: int=0
    ):
        """Initialisation of the class

        Args:
            known_x_train (pd.DataFrame): Training features data for which the label y is known
            known_y_train (pd.Series): Associated label y with known_x_train
            unknown_x_train (pd.DataFrame): Training features data for which the label y
                is not known
            model_convergence_epochs (int): number of iterations for classifier convergence
            model_learning_rate (float): learning rate for model's gradient descent
            n_input_units (int): number of neurons for the model's input layer
            n_hidden_units (int): number of neurons for the model's hidden layer
            verbose (bool): True if you want to print informations during training, else False
            random_seed (int, optional): Seed for weights initialisation. If 0,
                weights will be initialised randomly. Defaults to 0.
        """
        self.model_convergence_epochs = model_convergence_epochs
        self.learning_rate = model_learning_rate
        if random_seed != 0:
            torch.manual_seed(random_seed)
        self.model = SimpleNeuralNet(
            clipping_value=0.001,
            n_input_units=n_input_unit,
            n_hidden_units=n_hidden_units
        )
        self.known_x_train = known_x_train
        self.known_y_train = known_y_train
        self.unknown_x_train = unknown_x_train
        
        self.model.fit(
            self.known_x_train,
            self.known_y_train,
            epochs=self.model_convergence_epochs,
            learning_rate=self.learning_rate,
            verbose=verbose
        )
        self.accuracies = []
    
    def learning(
        self,
        validation_x: pd.DataFrame,
        validation_y: pd.Series,
        batch_adding: int=5,
        verbose: bool=False
    ):
        """Apply the self learning approach: add a batch of new labeled data with hard labels
        to the traning set at each iteration and keep track of the accuracy on a validation
        set.

        Args:
            validation_x (pd.DataFrame): Validation features
            validation_y (pd.Series): Validation labels
            batch_adding (int, optional): Number of instances to add at each iteration (add 
                a batch of most probable instances being 1 and a batch of most probable instances
                being 0). Defaults to 5.
            verbose (bool, optional): True if you want to print informations during learning,
                else False. Defaults to False
        """
        while len(self.unknown_x_train) >= batch_adding:
            self.unknown_x_train["y"] = self.model.predict_probas(
                self.unknown_x_train.to_numpy()
            ).detach().numpy()
            self.unknown_x_train.sort_values("y", ascending=False, inplace=True)
            self.unknown_x_train["y"] = self.unknown_x_train["y"].apply(
                lambda x: 1 if x>0.5 else 0
            )
            self.known_x_train = pd.concat(
                [
                    self.known_x_train,
                    self.unknown_x_train.iloc[0:batch_adding].drop(["y"], axis=1),
                    self.unknown_x_train.iloc[-batch_adding:].drop(["y"], axis=1)
                ]
            )
            self.known_y_train = pd.concat(
                [
                    self.known_y_train,
                    self.unknown_x_train.iloc[0:batch_adding]["y"],
                    self.unknown_x_train.iloc[-batch_adding:]["y"]
                ]
            )
            self.unknown_x_train = self.unknown_x_train.iloc[
                batch_adding:-batch_adding
            ].drop(["y"], axis=1)
            predictions = self.model.predict_probas(
                validation_x.to_numpy()
            ).reshape(-1).detach().numpy().round()
            self.accuracies.append(accuracy_score(validation_y, predictions))
            self.model.fit(
                self.known_x_train,
                self.known_y_train,
                epochs=self.model_convergence_epochs,
                learning_rate=self.learning_rate,
                verbose=verbose
            )

        self.unknown_x_train["y"] = self.model.predict_probas(
            self.unknown_x_train.to_numpy()
        ).detach().numpy()
        self.unknown_x_train["y"] = self.unknown_x_train["y"].apply(lambda x: 1 if x>0.5 else 0)
        self.known_x_train = pd.concat(
            [
                self.known_x_train,
                self.unknown_x_train.drop(["y"], axis=1),
            ]
        )
        self.known_y_train = pd.concat(
            [
                self.known_y_train,
                self.unknown_x_train["y"],
            ]
        )
        self.model.fit(
            self.known_x_train,
            self.known_y_train,
            epochs=self.model_convergence_epochs,
            learning_rate=self.learning_rate,
            verbose=verbose
        )
        predictions = self.model.predict_probas(
            validation_x.to_numpy()
        ).reshape(-1).detach().numpy().round()
        self.accuracies.append(accuracy_score(validation_y, predictions))


class SelfLearningWithSoft():
    """Self Learning approach using soft labels: At each iteration,
    we label the unknown set with the probabilities output by the classifier and
    we train the classifier on the concatenation of the labeled set and the new
    labeled set. The classifier is a neural network with one hidden layer.
    """
    def __init__(
        self,
        known_x_train: pd.DataFrame,
        known_y_train: pd.Series,
        unknown_x_train: pd.DataFrame,
        model_convergence_epochs: int,
        model_learning_rate: float,
        n_input_unit: int,
        n_hidden_units: int,
        verbose: bool,
        random_seed: int=0
    ):
        """Initialisation of the class

        Args:
            known_x_train (pd.DataFrame): Training features data for which the label y is known
            known_y_train (pd.Series): Associated label y with known_x_train
            unknown_x_train (pd.DataFrame): Training features data for which the label y
                is not known
            model_convergence_epochs (int): number of iterations for classifier convergence
            model_learning_rate (float): learning rate for model's gradient descent
            n_input_units (int): number of neurons for the model's input layer
            n_hidden_units (int): number of neurons for the model's hidden layer
            verbose (bool): True if you want to print informations during training, else False
            random_seed (int, optional): Seed for weights initialisation. If 0,
                weights will be initialised randomly. Defaults to 0.
        """
        self.model_convergence_epochs = model_convergence_epochs
        self.learning_rate = model_learning_rate
        if random_seed != 0:
            torch.manual_seed(random_seed)
        self.model = SimpleNeuralNet(
            clipping_value=0.001,
            n_input_units=n_input_unit,
            n_hidden_units=n_hidden_units
        )
        self.known_x_train = known_x_train
        self.known_y_train = known_y_train
        self.unknown_x_train = unknown_x_train
        self.length_known = len(self.known_y_train)
        
        self.model.fit(
            self.known_x_train,
            self.known_y_train,
            epochs=self.model_convergence_epochs,
            learning_rate=self.learning_rate,
            verbose=verbose
        )
        self.accuracies = []
    
    def learning(
        self,
        validation_x: pd.DataFrame,
        validation_y: pd.Series,
        n_epochs: int=10,
        verbose: bool=False
    ):
        """Apply the self learning approach: at each iteration, label the unknonw set
        with the probabilities output by the classifier, add this new labeled set to the
        training set and train the classifier on it.

        Args:
            validation_x (pd.DataFrame): Validation features
            validation_y (pd.Series): Validation labels
            n_epochs (int, optionnal): Number of learning iterations. Defaults to 10.
            verbose (bool, optional): True if you want to print informations during learning,
                else False. Defaults to False
        """
        for epoch in range(n_epochs):
            self.known_y_train = self.known_y_train.apply(
                lambda x: x-0.001 if x==1 else x+0.001 if x==0 else x
            )
            self.unknown_x_train["y"] = self.model.predict_probas(
                self.unknown_x_train.to_numpy()
            ).detach().numpy()
            self.known_x_train = pd.concat(
                [
                    self.known_x_train,
                    self.unknown_x_train.drop(["y"], axis=1),
                ]
            ).drop_duplicates()
            if epoch == 0:
                self.known_y_train = pd.concat(
                    [
                        self.known_y_train,
                        self.unknown_x_train["y"],
                    ]
                )
            else:
                self.known_y_train[self.length_known:] = self.unknown_x_train["y"]
            self.unknown_x_train = self.unknown_x_train.drop(["y"], axis=1)
            predictions = self.model.predict_probas(
                validation_x.to_numpy()
            ).reshape(-1).detach().numpy().round()
            self.accuracies.append(accuracy_score(validation_y, predictions))
            if verbose:
                print(f"Accuracy epochs {epoch}: {accuracy_score(validation_y, predictions)}")
            if epoch != n_epochs-1:
                self.model.fit(
                    self.known_x_train,
                    self.known_y_train,
                    epochs=self.model_convergence_epochs,
                    learning_rate=self.learning_rate,
                    verbose=False,
                    soft=True
                )


class SelfLearningUsingVennAbers():
    """Self Learning approach using Venn Abers predictors: At each iteration,
    we label the unknown set with the credal sets output by the Venn Abers predictors
    over the probabilities output by the classifier, we train the classifier on the concatenation
    of the labeled set and the new labeled set. The classifier is a neural network with one hidden
    layer.
    """
    def __init__(
        self,
        known_x_train: pd.DataFrame,
        known_y_train: pd.Series,
        unknown_x_train: pd.DataFrame,
        calib_x_train: pd.DataFrame, 
        calib_y_train: pd.Series,
        model_convergence_epochs: int,
        model_learning_rate: float,
        n_input_unit: int,
        n_hidden_units: int,
        verbose: bool,
        random_seed: int=0
    ):
        """Initialisation of the class

        Args:
            known_x_train (pd.DataFrame): Training features data for which the label y is known
            known_y_train (pd.Series): Associated label y with known_x_train
            unknown_x_train (pd.DataFrame): Training features data for which the label y
                is not known
            calib_x_train (pd.DataFrame): Calibration features used in the Venn Abers
            calib_y_train (pd.Series): Associated label y with calib_x_train
            model_convergence_epochs (int): number of iterations for classifier convergence
            model_learning_rate (float): learning rate for model's gradient descent
            n_input_units (int): number of neurons for the model's input layer
            n_hidden_units (int): number of neurons for the model's hidden layer
            verbose (bool): True if you want to print informations during training, else False
            random_seed (int, optional): Seed for weights initialisation. If 0,
                weights will be initialised randomly. Defaults to 0.
        """
        self.model_convergence_epochs = model_convergence_epochs
        self.learning_rate = model_learning_rate
        if random_seed != 0:
            torch.manual_seed(random_seed)
        self.model = SimpleNeuralNetCredal(
            clipping_value=0.001,
            n_input_units=n_input_unit,
            n_hidden_units=n_hidden_units
        )
        self.known_x_train = known_x_train
        self.known_y_train = known_y_train
        self.unknown_x_train = unknown_x_train
        self.calib_x_train = calib_x_train
        self.calib_y_train = calib_y_train
        self.length_known = len(self.known_y_train)
        
        self.model.fit(
            self.known_x_train,
            self.known_y_train,
            epochs=self.model_convergence_epochs,
            learning_rate=self.learning_rate,
            verbose=verbose
        )
        self.accuracies = []
    
    def learning(
        self,
        validation_x: pd.DataFrame,
        validation_y: pd.Series,
        n_epochs: int=10,
        verbose: bool=False
    ):
        """Apply the self learning approach: at each iteration, label the unknonw set
        with the credal sets output by the Venn Abers over the probabilities output by
        the classifier, add this new labeled set to the training set and
        train the classifier on it.

        Args:
            validation_x (pd.DataFrame): Validation features
            validation_y (pd.Series): Validation labels
            n_epochs (int, optionnal): Number of learning iterations. Defaults to 10.
            verbose (bool, optional): True if you want to print informations during learning,
                else False. Defaults to False
        """
        self.known_y_train = self.known_y_train.apply(
            lambda x: np.stack(
                [x+0.001, x+0.001] if x == 0 else [x-0.001, x-0.001] if x ==1 else x,
                axis=-1
        ).astype(np.float32))

        for epoch in range(n_epochs):
            self.unknown_x_train["interval"] =  self.unknown_x_train.apply(
                lambda x: venn_abers_pytorch(
                    trained_classifier=self.model,
                    calibration_features=torch.from_numpy(
                        self.calib_x_train.values.astype(np.float32)
                    ),
                    test_instance=torch.from_numpy(x.values.reshape(1, -1).astype(np.float32)),
                    calibration_labels=self.calib_y_train)
                ,
                axis=1
            )
            self.known_x_train = pd.concat(
                [
                    self.known_x_train,
                    self.unknown_x_train.drop("interval", axis=1),
                ]
            ).drop_duplicates()

            if epoch == 0:
                self.known_y_train = pd.concat(
                    [
                        self.known_y_train,
                        self.unknown_x_train["interval"],
                    ]
                )
            else:
                self.known_y_train[self.length_known:] = self.unknown_x_train["interval"]
            
            self.unknown_x_train = self.unknown_x_train.drop("interval", axis=1)
            predictions = self.model.predict_probas(
                validation_x.to_numpy()
            ).reshape(-1).detach().numpy().round()
            self.accuracies.append(accuracy_score(validation_y, predictions))

            if verbose:
                print(f"Accuracy epochs {epoch+1}: {accuracy_score(validation_y, predictions)}")

            if epoch != n_epochs-1:
                self.model.fit(
                    self.known_x_train,
                    self.known_y_train,
                    epochs=self.model_convergence_epochs,
                    learning_rate=self.learning_rate,
                    verbose=False,
                    credal=True
                )
            
    
    def predict_probas(self, x_test: np.ndarray) -> np.ndarray:
        """Predict the probabilities of a features set

        Args:
            x_test (np.ndarray): Features set

        Returns:
            np.ndarray: predicted probabilities
        """
        probs = self.model.predict_probas(x_test, from_numpy=True, as_numpy=True)
        return probs
    
    def predict_credal_sets(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """Predict the credal sets of the probabilities output by the
        classifier with a features set as input

        Args:
            x_test (pd.DataFrame): Features set

        Returns:
            pd.DataFrame: Features set with corresponding credal set
        """
        returns = x_test.copy()
        returns["interval"] = returns.apply(
                lambda x: venn_abers_pytorch(
                    trained_classifier=self.model,
                    calibration_features=torch.from_numpy(
                        self.calib_x_train.values.astype(np.float32)
                    ),
                    test_instance=torch.from_numpy(x.values.reshape(1, -1).astype(np.float32)),
                    calibration_labels=self.calib_y_train)
                ,
                axis=1
            )
        return returns

if __name__ == "__main__":
    """Example of how to use one of the 3 class above
    """
    import os
    print(os.getcwd())
    df = pd.read_csv("data/SynthCross_n1000_p2.csv")
    df["z"] = df["z"].apply(lambda x: 1 if x=="A" else 0)

    X_train, X_test, Y_train, Y_test = train_test_split(df[["X1", "X2"]], df["z"], train_size=0.8)
    X_train, X_calib, Y_train, Y_calib = train_test_split(X_train, Y_train, train_size=0.975)
    trainset = pd.DataFrame(X_train, columns=["X1", "X2"])
    trainset["z"] = Y_train
    trainset.iloc[80:, trainset.columns.get_loc("z")] = np.NaN 
    known_train = trainset.iloc[0:80]
    unknow_train = trainset.iloc[80:]
    known_train.shape, unknow_train.shape, X_calib.shape, X_test.shape
    test = SelfLearningUsingVennAbers(
        known_x_train=known_train[["X1", "X2"]],
        known_y_train=known_train["z"],
        unknown_x_train=unknow_train[["X1", "X2"]],
        calib_x_train=X_calib[["X1", "X2"]],
        calib_y_train=Y_calib,
        model_convergence_epochs=10,
        model_learning_rate=0.2,
        n_input_unit=2,
        n_hidden_units=3,
        verbose=True,
    )
    test.learning(validation_x=X_test, validation_y=Y_test, n_epochs=10, verbose=True)
