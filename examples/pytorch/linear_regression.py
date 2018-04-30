# Based on PyTorch 0.4.0 version of https://github.com/L1aoXingyu/pytorch-beginner/blob/master/01-Linear%20Regression/Linear_Regression.py
import torch
from torch import nn, optim

# Environment variables
epochs = 1000
input_dimension = 1
output_dimension = 1

# Define training data
x = torch.tensor([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]])

y = torch.tensor([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], requires_grad=False)

# Define Network
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

# Initialize model and optimizer
model = LinearRegression()
optimizer = optim.SGD(model.parameters(), lr=1e-4)


# Training
for epoch in range(epochs):
    # Forward Pass
    out = model(x)
    loss = nn.MSELoss()(out, y)

    # Backword Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: loss {loss[0]}")
