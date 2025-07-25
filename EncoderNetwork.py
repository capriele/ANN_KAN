import torch
import torch.nn as nn


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        stride_len,
        n_u,
        n_y,
        n_neurons,
        n_layer,
        state_size,
        nonlinearity="relu",
        kernel_regularizer=None,
        constraint_on_input_hidden_layer=None,
        use_group_lasso=False,
        state_reduction=False,
        input_layer_regularizer=None,
        future=0,
    ):
        super(EncoderNetwork, self).__init__()

        self.stride_len = stride_len
        self.n_u = n_u
        self.n_y = n_y
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.state_size = state_size
        self.future = future
        self.kernel_regularizer = kernel_regularizer
        self.constraint_on_input_hidden_layer = constraint_on_input_hidden_layer
        self.use_group_lasso = use_group_lasso
        self.state_reduction = state_reduction
        self.input_layer_regularizer = input_layer_regularizer

        # Input dimensions
        input_dim = (stride_len * n_u) + (stride_len * n_y)

        # Create layers
        self.layers = nn.ModuleList()

        # First layer (equivalent to enc0)
        self.layers.append(nn.Linear(input_dim, n_neurons, bias=True))

        # Hidden layers (equivalent to enc1, enc2, etc.)
        for i in range(n_layer - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons, bias=True))

        # Output layer (equivalent to encf)
        self.layers.append(nn.Linear(n_neurons, state_size, bias=True))

        # Set activation function
        if nonlinearity == "relu":
            self.activation = nn.ReLU()
        elif nonlinearity == "tanh":
            self.activation = nn.Tanh()
        elif nonlinearity == "sigmoid":
            self.activation = nn.Sigmoid()
        elif nonlinearity == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()  # default

        # Apply weight constraints if specified
        if constraint_on_input_hidden_layer is not None:
            self._apply_constraints()

    def _apply_constraints(self):
        """Apply weight constraints (if any) - implement based on your specific constraints"""
        # This would need to be implemented based on your specific constraint requirements
        pass

    def forward(self, inputs_y, inputs_u):
        # Concatenate inputs (equivalent to keras.layers.concatenate)
        x = torch.cat([inputs_y.float(), inputs_u.float()], dim=-1)

        # Pass through all layers except the last one with activation
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)

        # Final layer with linear activation
        x = self.layers[-1](x)

        return x

    def get_regularization_loss(self):
        """Calculate regularization losses if kernel_regularizer is specified"""
        reg_loss = 0.0

        if self.kernel_regularizer is not None:
            for i, layer in enumerate(self.layers):
                if i == 0 and self.use_group_lasso and not self.state_reduction:
                    # Use input_layer_regularizer for first layer if conditions are met
                    if self.input_layer_regularizer is not None:
                        reg_loss += self.input_layer_regularizer * torch.norm(
                            layer.weight, p=2
                        )
                else:
                    # Use standard kernel_regularizer
                    reg_loss += self.kernel_regularizer * torch.norm(layer.weight, p=2)

        return reg_loss
