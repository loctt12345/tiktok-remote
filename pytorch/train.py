import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from model import Model
import read_file

class Trainer():

    def __init__(self, num_classes, input_dim, lstm_layer, lr, pretrain,):
       self.num_classes = num_classes
       self.input_dim = input_dim
       self.lstm_layer = lstm_layer
       self.lr = lr
       self.pretrain = pretrain

    def setup(self, path):
        X_UP = read_file.process_data('../data/UP.txt', self.input_dim)
        X_DOWN = read_file.process_data('../data/DOWN.txt', self.input_dim)
        X_LIKE = read_file.process_data('../data/LIKE.txt', self.input_dim)
        X = []
        Y = []
        for i in range(max(len(X_UP), max(len(X_DOWN),len(X_LIKE)))):
            if i < len(X_UP):
                X.append(X_UP[i])
                Y.append(1)
            if i < len(X_DOWN):
                X.append(X_DOWN[i])
                Y.append(0)
            if i < len(X_LIKE):
                X.append(X_LIKE[i])
                Y.append(2)

        #print(Y)
        self.X_train, self.Y_train = X[0 : (4 * len(X)) // 5 ], Y[0 : (4 * len(X)) // 5 ]
        self.X_test, self.Y_test = X[(4 * len(X)) // 5 : ], Y[(4 * len(X)) // 5 : ]
        #print(self.Y_train)
        #print(self.Y_test)
        self.model = Model(input_dim=self.input_dim, num_classes=self.num_classes, lstm_layer=self.lstm_layer)
        if self.pretrain == True:
            self.model.load(path)
        self.loss_criteria = torch.nn.CrossEntropyLoss()
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.lr)    

    def train(self):
        losses = []
        accuarcy = []
        for j in tqdm(range(len(self.X_train))):
            self.optimizer.zero_grad()
            input = self.X_train[j]
            target = self.Y_train[j]
            self.model.reset_hidden()
            # output = 0
            # for i in range(0, no_of_timesteps):
            #     output = model.forward(torch.Tensor(np.array([input[i]])))
            # loss = loss_criteria(output, torch.LongTensor([target]))
            # accuarcy.append(1 if output.max(1)[1] == target else 0)
            output = self.model.forward(torch.Tensor(input))
            loss = self.loss_criteria(output[0].unsqueeze(0), torch.LongTensor([target]))
            accuarcy.append(1 if output[0].max(0)[1] == target else 0)
            
            losses.append(loss.detach().numpy())
            loss.backward()
            self.optimizer.step()

        avg_loss = np.average(losses)
        print("Train loss:" + str(avg_loss))
        print("Train accuracy :" + str(np.average(accuarcy) * 100))

    def validate(self):
        accuarcy = []
        for j in tqdm(range(len(self.X_test))):
            input = self.X_test[j]
            target = self.Y_test[j]
            
            output = self.model.forward(torch.Tensor(input))
            
            accuarcy.append(1 if output[0].max(0)[1] == target else 0)
        print("Test accuracy :" + str(np.average(accuarcy) * 100))

    def run(self, epoch, path):
        self.setup(path)
        for i in (range(epoch)):
            print("Epoch: " + str(i) + "\n")
            print("Train: \n")
            self.train()
        print("Test: \n")
        self.validate()
        self.model.save(path)

def main():
    trainer = Trainer(  num_classes=3, 
                        input_dim=84, 
                        lstm_layer=8, 
                        lr=0.02,
                        pretrain=False)
    #trainer.setup('model_3class2.pt')
    # trainer.validate()
    trainer.run(40, 'pretrain/model3.pt')
if __name__ == '__main__':
    main()