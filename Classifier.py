import warnings

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from itertools import product
import statistics

import sys
import time
import datetime
from datetime import datetime as dtime

# Python code to make a classifier


class NN(nn.Module):
    """Class of Neural Networks used in this scipt"""

    def __init__(self):
        zeta , HL = 256 , 5
        super().__init__()
        self.F = nn.ModuleList([nn.Linear(2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, 1, bias=True)])
        #self.F = nn.ModuleList([nn.Linear(2,1) , nn.Tanh()]) # For Linear classifier

    def forward(self, x):
        """Structured Neural Network.
        Inputs:
         - x: Tensor of shape (2,n) - space variable
         """

        x = x.float().T

        # Structure of the solution of the equation

        for i, module in enumerate(self.F):
            x = module(x)

        return x.T

class ML(NN):
    """Training of the neural network for classification"""

    def Data(self, K, p=0.8):
        """Generates data: labelled points.
        Inputs:
        - K: Int - Number of data.
        - p: Float - Proportion of data for train. Default: 0.8"""

        K_0 = int(p * K)
        X = torch.zeros([2, K])
        Y = torch.zeros([1, K])

        print("   ")
        print("Data creation...")
        for k in range(K):
            count = round(100 * (k + 1) / K, 2)
            sys.stdout.write("\r%d  " % count + "%")
            sys.stdout.flush()
            r = torch.rand(1)
            theta = 2*np.pi*torch.rand(1)
            if k % 4 == 0:
                #X[:, k] = 0.15*torch.randn(2) + torch.tensor([0.5 , 0.5])
                #X[:, k] = 0.15 * r * torch.tensor([torch.cos(theta), torch.sin(theta)])
                X[:, k] = 0.15 * theta * torch.tensor([torch.cos(2*theta), torch.sin(2*theta)])
                Y[:, k] = 0.0
            if k % 4 == 1:
                #X[:, k] = 0.15*torch.randn(2) + torch.tensor([0.5 , -0.5])
                #X[:, k] = (0.15 * r + 0.25) * torch.tensor([torch.cos(theta), torch.sin(theta)])
                X[:, k] = 0.15 * theta * torch.tensor([- torch.cos(2*theta), - torch.sin(2*theta)])
                Y[:, k] = 1.0
            if k % 4 == 2:
                #X[:, k] = 0.15*torch.randn(2) + torch.tensor([-0.5 , -0.5])
                #X[:, k] = (0.15 * r + 0.5) * torch.tensor([torch.cos(theta), torch.sin(theta)])
                X[:, k] = 0.15 * theta * torch.tensor([- torch.sin(2*theta), torch.cos(2*theta)])
                Y[:, k] = 2.0
            if k % 4 == 3:
                #X[:, k] = 0.15*torch.randn(2) + torch.tensor([-0.5 , 0.5])
                #X[:, k] = (0.15 * r + 0.75) * torch.tensor([torch.cos(theta), torch.sin(theta)])
                X[:, k] = 0.15 * theta * torch.tensor([torch.sin(2*theta), - torch.cos(2*theta)])
                Y[:, k] = 3.0

        X_train = X[:, 0:K_0]
        X_test = X[:, K_0:K]
        Y_train = Y[:, 0:K_0]
        Y_test = Y[:, K_0:K]

        torch.save((X_train, X_test, Y_train, Y_test), "Data")
        pass

    def Loss(self, X, Y, model):
        """Computes the Loss function between two series of data X and Y
        Inputs:
        - X: Tensor of shape (2,n): Inputs of Neural Network
        - Y: Tensor of shape (1,n): Expected outputs
        - model: Neural network which will be optimized
        Computes a predicted value Yhat which is a tensor of shape (1,n) and returns the mean squared error between Yhat and Y
        => Returns a tensor of shape (1,1)"""
        X = torch.tensor(X, dtype=torch.float32)
        X.requires_grad = True
        Yhat = torch.zeros_like(X)
        Yhat.requires_grad = True

        Yhat = model(X)
        loss = (((Y - Yhat)).abs() ** 2).mean()

        return loss

    def Train(self, model, name_data = "Data" , BS = 64 ,  N_epochs = 100 , N_epochs_print = 10):
        """Makes the training on the data
        Inputs:
        - model: Neural network which will be optimized
        - name_data: Str - Name of the dataset. Default: "Data"
        - BS: Int - Batch size. Default: 64
        - N_epochs: Int - Number of epochs for gradient descent. Default: 100
        - N_epochs_print: Int - Number of epochs between two prints of the Loss. Default: 10
        => Returns the lists Loss_train and Loss_test of the values of the Loss w.r.t. training and test,
        and best_model, which is the best apporoximation of the desired model"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("Training...")
        print(150 * "-")

        Data = torch.load(name_data)

        X_train = Data[0]
        X_test = Data[1]
        Y_train = Data[2]
        Y_test = Data[3]

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-9, amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(X_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                X_batch = X_train[:, ixs]
                Y_batch = Y_train[:, ixs]
                loss_train = self.Loss(X_batch, Y_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(X_test, Y_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Epoch', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        torch.save((Loss_train, Loss_test, best_model) , "model")

        pass

    def Eval(self, name_model = "model", name_data = "Data" , save_fig = False):
        """Evaluates the classifier.
        Inputs:
        - name_model: Str - Name of the trained model. Default: model
        - name_data: Str - Name of the dataset. Default: Data
        - save_fig: Boolean - Saves the figure or not. Default: False"""

        Data = torch.load(name_data)
        Loss_train , Loss_test , model = torch.load(name_model)

        X_train , X_test , Y_train , Y_test = np.array(Data[0]) , np.array(Data[1]) , np.array(Data[2]) , np.array(Data[3])

        idx_0_train = np.where(Y_train == 0.0)[1]
        idx_0_test = np.where(Y_test == 0.0)[1]

        idx_1_train = np.where(Y_train == 1.0)[1]
        idx_1_test = np.where(Y_test == 1.0)[1]

        idx_2_train = np.where(Y_train == 2.0)[1]
        idx_2_test = np.where(Y_test == 2.0)[1]

        idx_3_train = np.where(Y_train == 3.0)[1]
        idx_3_test = np.where(Y_test == 3.0)[1]


        x_grid , y_grid = torch.arange(-1,1.02,0.02) , torch.arange(-1,1.02,0.02)
        grid_x, grid_y = torch.meshgrid(x_grid, y_grid)

        z = torch.zeros_like(grid_x)
        for ix in range(grid_x.shape[1]):
            for iy in range(grid_x.shape[0]):
                z[ix,iy] = model(torch.tensor([grid_x[ix,iy],grid_y[ix,iy]]))

        #z = z - 0.5*torch.ones_like(z)
        z = z.detach().numpy()
        #z = np.round(z , 0)
        z = 1 / (1 + np.exp(-20 * (z-0.5))) + 2 / (1 + np.exp(-20 * (z-1.5))) + 3 / (1 + np.exp(-20 * (z-2.5)))

        plt.figure(figsize=(12,5))
        ax = plt.subplot(1,2,1)
        plt.xlim([-1.0,1.0])
        plt.ylim([-1.0,1.0])
        ax.set_aspect("equal")
        plt.contourf(np.array(grid_x), np.array(grid_y), z , 50 , cmap = "jet")
        #plt.colorbar()

        plt.scatter(X_train[0, idx_0_train], X_train[1, idx_0_train] , color = "white" , marker = "s")
        plt.scatter(X_test[0, idx_0_test], X_test[1, idx_0_test], color="white", marker="o")

        plt.scatter(X_train[0, idx_1_train], X_train[1, idx_1_train] , color = "black" , marker = "s")
        plt.scatter(X_test[0, idx_1_test], X_test[1, idx_1_test], color="black", marker="o")

        plt.scatter(X_train[0, idx_2_train], X_train[1, idx_2_train], color="magenta", marker="s")
        plt.scatter(X_test[0, idx_2_test], X_test[1, idx_2_test], color="magenta", marker="o")

        plt.scatter(X_train[0, idx_3_train], X_train[1, idx_3_train], color="silver", marker="s")
        plt.scatter(X_test[0, idx_3_test], X_test[1, idx_3_test], color="silver", marker="o")

        plt.xlabel("$x$")
        plt.ylabel("$y$")
        #plt.grid()

        ax = plt.subplot(1,2,2)
        plt.plot(list(range(len(Loss_train))),Loss_train , label = "$Loss_{train}$" , color = "green")
        plt.plot(list(range(len(Loss_test))),Loss_test , label = "$Loss_{test}$" , color = "red")
        plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss evolution")
        plt.grid()

        if save_fig == False:
            plt.show()
        else:
            plt.savefig("Classifier.pdf" , dpi = (200))
        pass



