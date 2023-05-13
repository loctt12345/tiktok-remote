import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from model import Model
import numpy as np
import pandas as pd

up_df = pd.read_csv("../RIGHTHANDUP.txt")
down_df = pd.read_csv("../RIGHTHANDDOWN.txt")
X = []
Y = []
no_of_timesteps = 27

dataset = up_df.iloc[:, 1:].values
n_sample = len(dataset)

for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps : i, :])
    Y.append(1)

dataset = down_df.iloc[:, 1:].values
n_sample = len(dataset)

for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps : i, :])
    Y.append(0)

X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


model = Model(input_dim=84, num_classes=2)
model.load('model.pt')
print(Y_train[0])
print(model.forward(torch.Tensor(X_train[0]))[0])