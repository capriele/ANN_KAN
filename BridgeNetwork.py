import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgeNetwork(nn.Module):
    def __init__(
        self,
        stateSize,
        N_U,
        n_neurons,
        n_layer,
        nonlinearity,
        kernel_regularizer=None,
        constraintOnInputHiddenLayer=None,
        useGroupLasso=False,
        stateReduction=False,
        inputLayerRegularizer=None,
        affineStruct=False,
        future=0,
    ):
        super(BridgeNetwork, self).__init__()

        self.stateSize = stateSize
        self.N_U = N_U
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.nonlinearity = nonlinearity
        self.kernel_regularizer = kernel_regularizer
        self.constraintOnInputHiddenLayer = constraintOnInputHiddenLayer
        self.useGroupLasso = useGroupLasso
        self.stateReduction = stateReduction
        self.inputLayerRegularizer = inputLayerRegularizer
        self.affineStruct = affineStruct
        self.future = future

        # Define activation function
        if nonlinearity == "relu":
            self.activation = F.relu
        elif nonlinearity == "tanh":
            self.activation = torch.tanh
        elif nonlinearity == "sigmoid":
            self.activation = torch.sigmoid
        else:
            self.activation = F.relu  # default

        # Input layer
        self.bridge0 = nn.Linear(stateSize + N_U, n_neurons)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(n_neurons, n_neurons) for i in range(n_layer - 1)]
        )

        # Bias output layer
        self.bridge_bias = nn.Linear(n_neurons, stateSize)

        # Affine structure layer if needed
        if affineStruct:
            self.bridge_f = nn.Linear(n_neurons, stateSize * (stateSize + N_U))

    def forward(self, inputs_novelU, inputs_state):
        # Concatenate inputs
        inputConcat = torch.cat([inputs_state.float(), inputs_novelU.float()], dim=-1)

        # First layer
        x = self.bridge0(inputConcat)
        x = self.activation(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        # Bias output
        bias = self.bridge_bias(x)

        if self.affineStruct:
            # Affine structure computation
            ABunshape = self.bridge_f(x)
            # Reshape to (batch_size, stateSize, stateSize + N_U)
            AB = ABunshape.view(-1, self.stateSize, self.stateSize + self.N_U)

            # Matrix multiplication: AB @ inputConcat
            # inputConcat needs to be reshaped for batch matrix multiplication
            if inputConcat.dim() < 3:
                inputConcat_expanded = inputConcat.unsqueeze(
                    -1
                )  # Add dimension for bmm
            else:
                inputConcat_expanded = inputConcat.view(
                    AB.shape[0], self.stateSize + self.N_U
                )
                inputConcat_expanded = inputConcat_expanded.unsqueeze(
                    -1
                )  # Add dimension for bmm
            out = torch.bmm(AB, inputConcat_expanded).squeeze(
                -1
            )  # Remove the extra dimension

            biasShape = bias.view(AB.shape[0], self.stateSize)
            out = biasShape + out

            return out, AB, bias
        else:
            out = bias
            return out, x, bias

    def apply_regularization(self):
        """
        Apply regularization losses. This should be called during training
        to add regularization terms to the loss function.
        """
        reg_loss = 0.0

        if self.kernel_regularizer is not None:
            # Apply regularization to all layers except possibly the first one
            for name, param in self.named_parameters():
                if "weight" in name:
                    if self.useGroupLasso and self.stateReduction and "bridge0" in name:
                        # Use input layer regularizer for first layer if group lasso is enabled
                        if self.inputLayerRegularizer is not None:
                            reg_loss += self.inputLayerRegularizer * torch.norm(param)
                    else:
                        reg_loss += self.kernel_regularizer * torch.norm(param)

        return reg_loss

    def apply_constraints(self):
        """
        Apply constraints to the input-hidden layer weights if specified.
        This should be called after each optimizer step.
        """
        if self.constraintOnInputHiddenLayer is not None:
            with torch.no_grad():
                # Apply constraint to the first layer weights
                self.bridge0.weight.data = self.constraintOnInputHiddenLayer(
                    self.bridge0.weight.data
                )
