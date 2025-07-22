import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Regularizer:
    """Base regularizer class similar to Keras"""

    def __call__(self, weight):
        raise NotImplementedError

    def __mul__(self, other):
        return ScaledRegularizer(self, other)

    def __rmul__(self, other):
        return ScaledRegularizer(self, other)


class ScaledRegularizer(Regularizer):
    """Regularizer scaled by a factor"""

    def __init__(self, regularizer, scale):
        self.regularizer = regularizer
        self.scale = scale

    def __call__(self, weight):
        return self.scale * self.regularizer(weight)


class L1(Regularizer):
    """L1 regularization (equivalent to Keras l1)"""

    def __init__(self, l=0.01):
        self.l = l

    def __call__(self, weight):
        return self.l * torch.sum(torch.abs(weight))


class L2(Regularizer):
    """L2 regularization (equivalent to Keras l2)"""

    def __init__(self, l=0.01):
        self.l = l

    def __call__(self, weight):
        return self.l * torch.sum(weight**2)


class L1L2(Regularizer):
    """L1 + L2 regularization (equivalent to Keras l1_l2)"""

    def __init__(self, l1=0.01, l2=0.01):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, weight):
        l1_loss = self.l1 * torch.sum(torch.abs(weight))
        l2_loss = self.l2 * torch.sum(weight**2)
        return l1_loss + l2_loss


# Convenience functions (similar to Keras)
def l1(l=0.01):
    """L1 regularizer factory function"""
    return L1(l)


def l2(l=0.01):
    """L2 regularizer factory function"""
    return L2(l)


def l1_l2(l1=0.01, l2=0.01):
    """L1+L2 regularizer factory function"""
    return L1L2(l1, l2)


# Alternative: Using PyTorch's built-in weight_decay for L2
class OptimizedL2(Regularizer):
    """L2 regularizer that works with optimizer's weight_decay"""

    def __init__(self, l=0.01):
        self.l = l
        print(
            f"Note: For L2 regularization, consider using weight_decay={l} in your optimizer instead"
        )

    def __call__(self, weight):
        return self.l * torch.sum(weight**2)


# Example usage in a neural network
class RegularizedLinear(nn.Module):
    """Linear layer with regularization"""

    def __init__(self, in_features, out_features, regularizer=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.regularizer = regularizer

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        """Compute regularization loss for this layer"""
        if self.regularizer is not None:
            return self.regularizer(self.linear.weight)
        return 0.0


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
