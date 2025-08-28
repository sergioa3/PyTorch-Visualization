import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets


#x_np, y_np = datasets.make_regression(n_samples=100,
#                                      n_features=1,
#                                      noise=20,
#                                      random_state=1)

x_np = np.linspace(0,100,1000)
y_np = 3.2*x_np + 5 + 20*np.random.normal(size=x_np.shape)


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

model = nn.Linear(n_features, n_features_y)

#alpha = 2.5
alpha = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                        lr=alpha)

delta = 0.1**(6)
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