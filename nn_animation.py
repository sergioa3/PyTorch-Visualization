import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


x_np = np.linspace(0,3.5*np.pi,1000)
y_np = np.sin(x_np) + 5 + 0*np.random.normal(size=x_np.shape)


x = torch.from_numpy(x_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
x = x.view(x.shape[0], -1)
y = y.view(y.shape[0], -1)
#print(y.shape)
#print(x.shape)
#print(y.view(y.shape[0], -1).shape)
n_samples, n_features = x.shape
#print(n_samples)
#print(n_features)
n_features_y = y.shape[1]
#print(n_features_y)

#model = nn.Linear(n_features, n_features_y)

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(NeuralNet, self).__init__()
#         self.linear1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(hidden_size, output_size)
#         #self.out_activation = nn.Softmax(output_size)

#     def forward(self, x):
#         out = self.linear1(x)
#         out = self.relu(out)
#         out = self.linear2(out)
#         #out = self.out_activation(out)
#         return out

#model = NeuralNet(1, 13, 1)

model = nn.Sequential(
    nn.Linear(1, 30),    # Input to first hidden layer
    nn.ReLU(),           # Activation
    nn.Linear(30, 80),    # Second hidden layer
    nn.ReLU(),           # Another activation function
    nn.Linear(80, 100),      #
    nn.ReLU(),           # Another activation function
    nn.Linear(100, 80),      #
    nn.Tanh(),           # Another activation function
    nn.Linear(80, 50),      #
    nn.ReLU(),           # Another activation function
    nn.Linear(50, 20),      #
    nn.Tanh(),           # Another activation function
             
    nn.Linear(20, 1)
)


#alpha = 2.5
alpha = 0.04
criterion = nn.MSELoss()
#print(model.parameters)
optimizer = torch.optim.ASGD(model.parameters(),
                        lr=alpha)

delta = 0.1**(7)
converged = False
with torch.no_grad():
    error = criterion(model(x), y)/delta
    #error = 10000000000000000000
while not converged:
    y_hat = model(x)
    loss = criterion(y_hat, y)
    
    converged = np.abs(error - loss.detach().numpy()) < delta

    error = loss.detach().numpy()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        prediction = model(x).numpy()
        plt.plot(x_np, y_np,'o', markersize=0.5, color='r')
        plt.plot(x_np, prediction, 'b')
        plt.title('Error={}'.format(np.round(error,decimals=3)))
        plt.pause(0.000001)
        plt.clf()
        #print(error)

with torch.no_grad():
    plt.plot(x_np, y_np,'o', markersize=0.5, color='r')
    plt.plot(x_np, prediction, 'b')
    plt.title('Converged at Error={}, with delta={:.10g}'.format(np.round(error,decimals=3), delta))
    plt.show()
    print(error)