import torch
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=132, lstm_layer=5):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.lstm_layer = lstm_layer
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, lstm_layer)
        self.linear = torch.nn.Linear(hidden_dim + input_dim, num_classes)
        self.hidden = (torch.zeros(lstm_layer, hidden_dim), torch.zeros(lstm_layer, hidden_dim))

    def forward(self, input):
        output, self.hidden = self.lstm(input, self.hidden)
        output = self.linear(torch.concat((output, input), dim=1))
        return F.log_softmax(output, dim=1)

    def reset_hidden(self):
        self.hidden = (torch.zeros(self.lstm_layer, self.hidden_dim), torch.zeros(self.lstm_layer, self.hidden_dim))


    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

def main():
    input = torch.rand(27, 84)
    model = Model(input_dim=84, num_classes=3)
    output = model.forward(input)
    model.save('model.pt')
    print(output)

if __name__ == '__main__':
    main()