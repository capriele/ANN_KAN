import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineStructLayer(nn.Module):
    """
    Custom layer to handle the affine structure computation.
    Takes reshaped output and input state to compute affine transformation.
    """

    def __init__(self, state_size, output_window_len):
        super(AffineStructLayer, self).__init__()
        self.state_size = state_size
        self.output_window_len = output_window_len

    def forward(self, x_reshaped, input_state):
        """
        Args:
            x_reshaped: (batch_size, output_window_len, state_size)
            input_state: (batch_size, state_size)
        Returns:
            affine_output: (batch_size, output_window_len)
        """
        batch_size = input_state.size(0)

        # Expand input state for broadcasting
        input_expanded = input_state.unsqueeze(1)  # (batch_size, 1, state_size)

        # Element-wise multiply and sum over last dimension
        affine_output = torch.sum(
            x_reshaped * input_expanded, dim=2
        )  # (batch_size, output_window_len)

        return affine_output


"""
class DecoderNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        n_neurons,
        n_layer,
        nonlinearity,
        output_window_len,
        N_Y,
        affine_struct=False,
        kernel_regularizer=None,
        use_group_lasso=False,
        state_reduction=False,
        input_layer_regularizer=None,
        constraint_on_input_hidden_layer=None,
        future=0,
    ):
        super(DecoderNetwork, self).__init__()

        self.state_size = state_size
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.output_window_len = output_window_len
        self.N_Y = N_Y
        self.affine_struct = affine_struct
        self.kernel_regularizer = kernel_regularizer
        self.use_group_lasso = use_group_lasso
        self.state_reduction = state_reduction
        self.input_layer_regularizer = input_layer_regularizer
        self.constraint_on_input_hidden_layer = constraint_on_input_hidden_layer
        self.future = future

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

        # Build layers
        self.layers = nn.ModuleList()

        # First layer (input layer)
        self.layers.append(nn.Linear(self.state_size, self.n_neurons))

        # Hidden layers
        for i in range(self.n_layer - 1):
            self.layers.append(nn.Linear(self.n_neurons, self.n_neurons))

        # Output layer - always outputs to state_size dimension when affine_struct=True
        if self.affine_struct:
            self.final_layer = nn.Linear(
                self.n_neurons, self.output_window_len * self.state_size
            )
            # Add the affine structure layer
            # self.affine_layer = AffineStructLayer(
            #    self.state_size, self.output_window_len
            # )

            # Second output layer for non-affine output
            self.direct_output_layer = nn.Linear(
                self.n_neurons, self.output_window_len * self.N_Y
            )
        else:
            self.final_layer = nn.Linear(
                self.n_neurons, self.output_window_len * self.N_Y
            )

    def forward(self, inputs_state):
        batch_size = inputs_state.size(0)

        # Store input for affine computation - detach and clone to break any shared memory
        stored_input = inputs_state

        # Start forward pass with original input
        h = inputs_state

        # Pass through all hidden layers
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.activation(h)

        if self.affine_struct:
            # First output: for affine structure
            affine_raw_output = self.final_layer(h)
            # x_reshaped = affine_raw_output.view(
            #    batch_size, self.output_window_len, self.state_size
            # )

            # Apply affine transformation using the dedicated layer
            # affine_output = self.affine_layer(x_reshaped, stored_input)

            # Second output: direct output
            direct_output = self.direct_output_layer(h)

            return affine_raw_output, direct_output
        else:
            output = self.final_layer(h)
            return output, output
"""


class DecoderNetwork(nn.Module):
    def __init__(
        self,
        state_size,
        n_neurons,
        n_layer,
        nonlinearity,
        output_window_len,
        N_Y,
        affine_struct=False,
        kernel_regularizer=None,
        use_group_lasso=False,
        state_reduction=False,
        input_layer_regularizer=None,
        constraint_on_input_hidden_layer=None,
        future=0,
    ):
        super(DecoderNetwork, self).__init__()

        self.state_size = state_size
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.output_window_len = output_window_len
        self.N_Y = N_Y
        self.affine_struct = affine_struct
        self.kernel_regularizer = kernel_regularizer
        self.use_group_lasso = use_group_lasso
        self.state_reduction = state_reduction
        self.input_layer_regularizer = input_layer_regularizer
        self.constraint_on_input_hidden_layer = constraint_on_input_hidden_layer
        self.future = future

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

        # Build layers
        self.layers = nn.ModuleList()

        # First layer (input layer)
        self.layers.append(nn.Linear(self.state_size, self.n_neurons))

        # Hidden layers
        for i in range(self.n_layer - 1):
            self.layers.append(nn.Linear(self.n_neurons, self.n_neurons))

        # Output layer
        if self.affine_struct:
            self.final_layer = nn.Linear(
                self.n_neurons, self.output_window_len * self.state_size
            )
        else:
            self.final_layer = nn.Linear(
                self.n_neurons, self.output_window_len * self.N_Y
            )

    def forward(self, inputs_state):
        # Forward pass through hidden layers
        x = inputs_state

        # First layer with potential regularization differences
        x = self.layers[0](x)
        x = self.activation(x)

        # Hidden layers
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)

        # Final layer (always linear activation)
        x = self.final_layer(x)

        if self.affine_struct:
            # Reshape to (batch_size, output_window_len, state_size)
            x = x.view(-1, self.output_window_len, self.state_size)
            # Dot product with inputs_state along last dimension
            # inputs_state: (batch_size, state_size)
            # x: (batch_size, output_window_len, state_size)
            out = torch.sum(x * inputs_state.unsqueeze(1), dim=-1)
            # out shape: (batch_size, output_window_len)
        else:
            out = x

        return x, out

    def apply_regularization(self):
        """
        Apply regularization during training. Call this in your training loop
        to add regularization losses to your main loss.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        reg_loss = torch.tensor(0.0, device=device, dtype=dtype)

        if self.kernel_regularizer is not None:
            for i, layer in enumerate(self.layers):
                if i == 0 and self.use_group_lasso and self.state_reduction:
                    # Apply input layer regularizer to first layer
                    if self.input_layer_regularizer is not None:
                        weight_norm = torch.norm(layer.weight)
                        reg_loss = reg_loss + self.input_layer_regularizer(weight_norm)
                else:
                    # Apply kernel regularizer to other layers
                    weight_norm = torch.norm(layer.weight)
                    reg_loss = reg_loss + self.kernel_regularizer(weight_norm)

            # Apply regularization to final layer(s)
            final_weight_norm = torch.norm(self.final_layer.weight)
            reg_loss = reg_loss + self.kernel_regularizer(final_weight_norm)

            ### If affine_struct, also regularize the direct output layer
            ##if self.affine_struct:
            ##    direct_weight_norm = torch.norm(self.direct_output_layer.weight)
            ##    reg_loss = reg_loss + self.kernel_regularizer(direct_weight_norm)

        return reg_loss

    def apply_constraints(self):
        """
        Apply weight constraints. Call this ONLY after optimizer.step().
        Never call this during forward pass or before backward pass.
        """
        if self.constraint_on_input_hidden_layer is not None:
            with torch.no_grad():
                if callable(self.constraint_on_input_hidden_layer):
                    # Get current weights
                    current_weights = self.layers[0].weight.data
                    # Apply constraint function
                    constrained_weights = self.constraint_on_input_hidden_layer(
                        current_weights
                    )
                    # Use copy_ to avoid creating new tensor references
                    if not torch.equal(current_weights, constrained_weights):
                        self.layers[0].weight.data.copy_(constrained_weights)
