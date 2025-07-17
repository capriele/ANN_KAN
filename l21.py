import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class L21Regularization(nn.Module):
    """Regularizer for L21 regularization in PyTorch."""

    def __init__(self, C=0.0, a=0, b=0, bias=0.0):
        super(L21Regularization, self).__init__()
        self.a = a
        self.b = b
        self.C = (bias + C) * np.square(
            np.concatenate([a - np.array(range(0, a)), b - np.array(range(0, b))])
        )
        self.C = torch.tensor(self.C, dtype=torch.float32)
        print("****Squared weighting enabled****")
        print(self.C)

    def forward(self, x):
        w = torch.sum(torch.abs(x), dim=1)
        w = w[0 : self.a + self.b]
        return torch.sum(w * self.C)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, l21_reg=None):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.l21_reg = l21_reg

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def l21_regularization(self):
        if self.l21_reg is not None:
            l21_loss = 0
            for param in self.parameters():
                l21_loss += self.l21_reg(param)
            return l21_loss
        return 0


if __name__ == "__main__":
    # Example usage:
    input_dim = 100
    output_dim = 10
    l21_reg = L21Regularization(C=0.01, a=5, b=5, bias=0.0001)

    model = NeuralNetwork(input_dim, output_dim, l21_reg=l21_reg)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Example training loop
    num_epochs = 10
    batch_size = 32
    # Assuming you have input_data and target_data ready
    # input_data = torch.randn(1000, input_dim)
    # target_data = torch.randn(1000, output_dim)
    # dataset = torch.utils.data.TensorDataset(input_data, target_data)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            l21_loss = model.l21_regularization()
            total_loss = loss + l21_loss
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {total_loss.item()}")
