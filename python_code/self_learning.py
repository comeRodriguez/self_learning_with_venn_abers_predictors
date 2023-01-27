import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from collections import OrderedDict
import tensorflow as tf


class MyDataset(torch.utils.data.Dataset):
 
    def __init__(self, x_train, y_train, credal=False):
        x=x_train.values
        y=y_train.values
        self.X_train=torch.tensor(x,dtype=torch.float32)
        if not credal:
            self.Y_train=y.astype("float32")
        else:
            self.Y_train=y

    def __len__(self):
        return len(self.Y_train)
   
    def __getitem__(self,idx):
        return self.X_train[idx],self.Y_train[idx]


def KLD_loss_intervals(y_true, y_pred):
    # print("Y TRUE: ", y_true)
    # print("Y PRED: ", y_pred)
    
    y_true_inf = y_true[:, 0].reshape(-1, 1)
    y_true_sup = y_true[:, 1].reshape(-1, 1)
    used_probas = torch.zeros(y_true.shape[0], 1)
    used_probas[y_pred <= y_true_inf] = y_true_inf[y_pred <= y_true_inf]
    used_probas[y_pred >= y_true_sup] = y_true_sup[y_pred >= y_true_sup]
    used_probas[(y_pred >= y_true_inf) & (y_pred <= y_true_sup)
                ] = y_pred[(y_pred >= y_true_inf) & (y_pred <= y_true_sup)]
    # import pdb; pdb.set_trace()
    loss = torch.mean(used_probas* torch.log(used_probas/y_pred) + (1-used_probas)*torch.log((1-used_probas)/(1-y_pred)))
    return loss


def KLD_loss(y_true, y_pred):
    # print("Y TRUE: ", y_true)
    # print("Y PRED: ", y_pred)
    loss = torch.mean(y_true* torch.log(y_true/y_pred) + (1-y_true)*torch.log((1-y_true)/(1-y_pred)))
    # import pdb; pdb.set_trace()
    return loss

def venn_abers_pytorch(trained_classifier, calibration_features, test_instance, calibration_labels):
    calibrations = trained_classifier.predict_probas(calibration_features, as_numpy=True, from_numpy=False)
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
        g = IsotonicRegression()
        g.fit(train["s"], train["y"])
        interval.append(np.round(g.predict(score["s"])[0], 3))
    interval = np.array(interval)
    return interval



class Simple_Neural_Net(nn.Module):
    def __init__(self, clipping_value):
        super().__init__()
        self.input_layer = nn.Sequential(OrderedDict({
            'linear': nn.Linear(in_features=2, out_features=3),
            'relu': nn.ReLU(inplace=True),
        }))
        self.hidden_layer = nn.Linear(in_features=3, out_features=1)
        self.output_layer = nn.Sigmoid()
        self.losses = {}
        self.accuracies = {}
        self.clipping_value = clipping_value
    
    def forward(self, x):
        y = self.input_layer(x)
        y = self.hidden_layer(y)
        y_pred = self.output_layer(y)
        return y_pred
    
    def __initiate_loss_and_accuracy_dicts(self, n_epochs):
        for i in range(n_epochs):
            self.losses[i] = 0
            self.accuracies[i] = 0
    
    def fit(self, x_train, y_train, epochs, learning_rate, verbose=True, soft=False):
        self.__initiate_loss_and_accuracy_dicts(n_epochs=epochs)
        myDs=MyDataset(x_train=x_train, y_train=y_train)
        train_loader=torch.utils.data.DataLoader(myDs,batch_size=10,shuffle=False)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            n_batches = len(train_loader)
            if verbose:
                print(f'Epoch {e+1}/{epochs}')
                pbar = tf.keras.utils.Progbar(target=n_batches)
            for idx, batch in enumerate(train_loader):
                x, y = batch
                optimizer.zero_grad()
                outputs = self(x)
                if torch.any(outputs.isnan()):
                    print("JE SUIS LA")
                    import pdb;pdb.set_trace()
                if not soft:
                    loss = nn.BCELoss()
                    loss_value = loss(outputs, y.reshape(-1, 1))
                else:
                    loss_value = KLD_loss(y.reshape(-1, 1), outputs)
                    # print("LOSS: ", loss_value)
                    
                # torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
                loss_value.backward()
                # if not torch.any(self.hidden_layer.weight.grad.isnan()):
                #     print("HIDDEN LAYER WEIGHTS", self.hidden_layer.weight)
                #     print("HIDDEN LAYER GRAD", self.hidden_layer.weight.grad)
                #     print("INPUT LAYER WEIGHTS", self.hidden_layer.weight)
                #     print("INPUT LAYER GRAD", self.input_layer[0].weight.grad)
                #     optimizer.step()
                #     print("HIDDEN LAYER WEIGHTS AFTER GRAD", self.hidden_layer.weight)
                #     print("INPUT LAYER WEIGHTS AFTER GRAD", self.hidden_layer.weight)
                # else:
                optimizer.step()
                if not soft:
                    accuracy = accuracy_score(
                        y_true=y,
                        y_pred=outputs.reshape(-1).detach().numpy().round()
                    )
                    self.accuracies[e] += accuracy
                self.losses[e] += loss_value.detach().numpy()
                if verbose:
                    pbar.update(
                        idx,
                        values=[
                            ("loss", loss_value.detach().numpy()),
                            ("accuracy", accuracy)
                        ]
                    )
            self.losses[e] = self.losses[e]/n_batches
            self.accuracies[e] = self.accuracies[e]/n_batches
            if verbose:
                pbar.update(n_batches, values=None) 
            
    def predict_probas(self, x, as_numpy=False, from_numpy=True):
        if from_numpy:
            x = torch.tensor(x,dtype=torch.float32)
        outputs = self(x)
        if as_numpy:
            outputs = outputs.detach().numpy()
        return outputs



class Simple_Neural_Net2(nn.Module):
    def __init__(self, clipping_value):
        super().__init__()
        self.input_layer = nn.Sequential(OrderedDict({
            'linear': nn.Linear(in_features=2, out_features=3),
            'relu': nn.ReLU(inplace=True),
        }))
        self.hidden_layer = nn.Linear(in_features=3, out_features=1)
        self.output_layer = nn.Sigmoid()
        self.losses = {}
        self.accuracies = {}
        self.clipping_value = clipping_value
    
    def forward(self, x):
        y = self.input_layer(x)
        y = self.hidden_layer(y)
        y_pred = self.output_layer(y)
        return y_pred
    
    def __initiate_loss_and_accuracy_dicts(self, n_epochs):
        for i in range(n_epochs):
            self.losses[i] = 0
            self.accuracies[i] = 0
            
    def fit(self, x_train, y_train, epochs, learning_rate, verbose=True, credal=False):
        self.__initiate_loss_and_accuracy_dicts(n_epochs=epochs)
        myDs=MyDataset(x_train=x_train, y_train=y_train, credal=credal)
        train_loader=torch.utils.data.DataLoader(myDs,batch_size=10,shuffle=False)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            n_batches = len(train_loader)
            if verbose:
                print(f'Epoch {e+1}/{epochs}')
                pbar = tf.keras.utils.Progbar(target=n_batches)
            for idx, batch in enumerate(train_loader):
                x, y = batch
                optimizer.zero_grad()
                outputs = self(x)
                
                outputs = torch.where(outputs==1, outputs-0.001, outputs)
                outputs = torch.where(outputs==0, outputs+0.001, outputs)
                if torch.any(outputs.isnan()):
                    print("JE SUIS LA")
                    import pdb; pdb.set_trace()
                if not credal:
                    loss = nn.BCELoss()
                    loss_value = loss(outputs, y.reshape(-1, 1))
                else:
                    loss_value = KLD_loss_intervals(y, outputs)
                    if torch.any(loss_value.isnan()):
                        print("IM HERE")
                        import pdb; pdb.set_trace()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
                loss_value.backward()
                optimizer.step()
                if not credal:
                    accuracy = accuracy_score(
                        y_true=y,
                        y_pred=outputs.reshape(-1).detach().numpy().round()
                    )
                    self.accuracies[e] += accuracy
                self.losses[e] += loss_value.detach().numpy()
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
            self.losses[e] = self.losses[e]/n_batches
            self.accuracies[e] = self.accuracies[e]/n_batches
            if verbose:
                pbar.update(n_batches, values=None) 
            
    def predict_probas(self, x, as_numpy=False, from_numpy=True):
        if from_numpy:
            x = torch.tensor(x,dtype=torch.float32)
        outputs = self(x)
        if as_numpy:
            outputs = outputs.detach().numpy()
        return outputs

class SelfLearningWithSoft():
    def __init__(
        self,
        known_x_train,
        known_y_train,
        unknown_x_train,
        model_convergence_epochs 
    ):
        self.model_convergence_epochs = model_convergence_epochs
        self.model = Simple_Neural_Net(clipping_value=0.000001)
        self.known_x_train = known_x_train
        self.known_y_train = known_y_train
        self.unknown_x_train = unknown_x_train
        self.length_known = len(self.known_y_train)
        
        self.model.fit(self.known_x_train, self.known_y_train, epochs=self.model_convergence_epochs, learning_rate=0.001)
        self.accuracies = []
    
    def learning(self, validation_x, validation_y, n_epochs=10):
        for epochs in range(n_epochs):
            # print(self.known_y_train)
            # print()
            self.known_y_train = self.known_y_train.apply(lambda x: x-0.001 if x==1 else x+0.001 if x==0 else x)
            # print(self.known_y_train)
            self.unknown_x_train["y"] = self.model.predict_probas(self.unknown_x_train.to_numpy()).detach().numpy()
            # import pdb; pdb.set_trace()
            self.known_x_train = pd.concat(
                [
                    self.known_x_train,
                    self.unknown_x_train[["X1", "X2"]],
                ]
            ).drop_duplicates()
            if epochs == 0:
                self.known_y_train = pd.concat(
                    [
                        self.known_y_train,
                        self.unknown_x_train["y"],
                    ]
                )
            else:
                self.known_y_train[self.length_known:] = self.unknown_x_train["y"]
            self.unknown_x_train = self.unknown_x_train[["X1", "X2"]]
            predictions = self.model.predict_probas(validation_x.to_numpy()).reshape(-1).detach().numpy().round()
            self.accuracies.append(accuracy_score(validation_y, predictions))
            print(f"Accuracy epochs {epochs}: {accuracy_score(validation_y, predictions)}")
            self.model.fit(
                self.known_x_train, self.known_y_train, epochs=self.model_convergence_epochs, learning_rate=0.1, verbose=False, soft=True
            )



class SelfLearningUsingVennAbers():
    def __init__(
        self,
        known_x_train,
        known_y_train,
        unknown_x_train,
        calib_x_train, 
        calib_y_train,
        model_convergence_epochs
    ):
        self.model_convergence_epochs = model_convergence_epochs
        self.model = Simple_Neural_Net2(clipping_value=0.1)
        self.known_x_train = known_x_train
        self.known_y_train = known_y_train
        self.unknown_x_train = unknown_x_train
        self.calib_x_train = calib_x_train
        self.calib_y_train = calib_y_train
        self.length_known = len(self.known_y_train)
        
        self.model.fit(self.known_x_train, self.known_y_train, epochs=self.model_convergence_epochs, learning_rate=0.2)
        self.accuracies = []
        # fig, ax = plot_decision_boundary(dataset=df[["X1", "X2"]], labels=np.array([0, 1]), model=self.model)
        # sns.scatterplot(data=df, x="X1", y="X2", hue="z", ax=ax)
        # plt.show()
    
    def learning(self, validation_x, validation_y, n_epochs=10):
        self.known_y_train = self.known_y_train.apply(
            lambda x: np.stack(
                [x+0.001, x+0.001] if x == 0 else [x-0.001, x-0.001] if x ==1 else x,
                axis=-1
        ).astype(np.float32))
        for epochs in range(n_epochs):
            self.unknown_x_train["interval"] =  self.unknown_x_train.apply(
                lambda x: venn_abers_pytorch(
                    self.model,
                    torch.from_numpy(self.calib_x_train[["X1", "X2"]].values.astype(np.float32)),
                    torch.from_numpy(x[["X1", "X2"]].values.reshape(1, -1).astype(np.float32)),
                    self.calib_y_train)
                ,
                axis=1
            )
            self.known_x_train = pd.concat(
                [
                    self.known_x_train,
                    self.unknown_x_train[["X1", "X2"]],
                ]
            ).drop_duplicates()
            if epochs == 0:
                self.known_y_train = pd.concat(
                    [
                        self.known_y_train,
                        self.unknown_x_train["interval"],
                    ]
                )
            else:
                self.known_y_train[self.length_known:] = self.unknown_x_train["interval"]
            
            self.unknown_x_train = self.unknown_x_train[["X1", "X2"]]
            predictions = self.model.predict_probas(validation_x.to_numpy()).reshape(-1).detach().numpy().round()
            self.accuracies.append(accuracy_score(validation_y, predictions))
            if (epochs+1) % 5 == 0 or epochs==0:
                print(f"Accuracy epochs {epochs+1}: {accuracy_score(validation_y, predictions)}")
                # fig, ax = plot_decision_boundary(dataset=df[["X1", "X2"]], labels=np.array([0, 1]), model=self.model)
                # sns.scatterplot(data=df, x="X1", y="X2", hue="z", ax=ax)
                # plt.show()
            self.model.fit(
                self.known_x_train,
                self.known_y_train,
                epochs=self.model_convergence_epochs,
                learning_rate=0.2,
                verbose=False,
                credal=True
            )
    def predict_probas(self, x_test):
        returns = x_test.copy()
        returns["interval"] = returns.apply(
                lambda x: venn_abers_pytorch(
                    self.model,
                    torch.from_numpy(self.calib_x_train[["X1", "X2"]].values.astype(np.float32)),
                    torch.from_numpy(x[["X1", "X2"]].values.reshape(1, -1).astype(np.float32)),
                    self.calib_y_train)
                ,
                axis=1
            )
        return returns

if __name__ == "__main__":
    df = pd.read_csv("../data/SynthCross_n1000_p2.csv")
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
        model_convergence_epochs=10
    )
    test.learning(validation_x=X_test, validation_y=Y_test, n_epochs=10)