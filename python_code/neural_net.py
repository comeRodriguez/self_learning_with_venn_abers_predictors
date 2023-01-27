from sklearn.metrics import accuracy_score

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
    y_true_inf = y_true[:, 0].reshape(-1, 1)
    y_true_sup = y_true[:, 1].reshape(-1, 1)
    used_probas = torch.zeros(y_true.shape[0], 1)
    used_probas[y_pred <= y_true_inf] = y_true_inf[y_pred <= y_true_inf]
    used_probas[y_pred >= y_true_sup] = y_true_sup[y_pred >= y_true_sup]
    used_probas[(y_pred >= y_true_inf) & (y_pred <= y_true_sup)
                ] = y_pred[(y_pred >= y_true_inf) & (y_pred <= y_true_sup)]
    loss = torch.mean(used_probas* torch.log(used_probas/y_pred) + (1-used_probas)*torch.log((1-used_probas)/(1-y_pred)))
    return loss


def KLD_loss(y_true, y_pred):
    loss = torch.mean(y_true* torch.log(y_true/y_pred) + (1-y_true)*torch.log((1-y_true)/(1-y_pred)))
    return loss

class Simple_Neural_Net(nn.Module):
    def __init__(self, clipping_value, n_units):
        super().__init__()
        self.input_layer = nn.Sequential(OrderedDict({
            'linear': nn.Linear(in_features=2, out_features=n_units),
            'relu': nn.ReLU(inplace=True),
        }))
        self.hidden_layer = nn.Linear(in_features=n_units, out_features=1)
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
                outputs = torch.where(outputs==1, outputs-0.001, outputs)
                outputs = torch.where(outputs==0, outputs+0.001, outputs)
                if torch.any(outputs.isnan()):
                    print("JE SUIS LA")
                    import pdb;pdb.set_trace()
                if not soft:
                    loss = nn.BCELoss()
                    loss_value = loss(outputs, y.reshape(-1, 1))
                else:
                    loss_value = KLD_loss(y.reshape(-1, 1), outputs)
                    
                loss_value.backward()
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
    def __init__(self, clipping_value, n_units):
        super().__init__()
        self.input_layer = nn.Sequential(OrderedDict({
            'linear': nn.Linear(in_features=2, out_features=n_units),
            'relu': nn.ReLU(inplace=True),
        }))
        self.hidden_layer = nn.Linear(in_features=n_units, out_features=1)
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