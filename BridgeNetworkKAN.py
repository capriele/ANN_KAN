import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN


class BridgeNetworkKAN(nn.Module):
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
        grid_size=5,
        spline_order=3,
        noise_scale=0.1,
        seed=0
    ):
        super(BridgeNetworkKAN, self).__init__()

        self.stateSize = stateSize
        self.N_U = N_U
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.nonlinearity = nonlinearity
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_regularizer = kernel_regularizer
        self.constraintOnInputHiddenLayer = constraintOnInputHiddenLayer
        self.useGroupLasso = useGroupLasso
        self.stateReduction = stateReduction
        self.inputLayerRegularizer = inputLayerRegularizer
        self.affineStruct = affineStruct
        self.future = future

        # Define the KAN network architecture
        # Input size is stateSize + N_U, hidden layers have n_neurons each
        kan_width = [
            stateSize + N_U,
            n_neurons,
        ]  # * (n_layer - 1) + [n_neurons] #TODO: manage this change

        # Main KAN network for feature extraction
        self.kan_network = KAN(
            width=kan_width,
            grid=grid_size,
            k=spline_order,
            noise_scale=noise_scale,
            seed=seed
        )

        # Disable symbolic training
        self.kan_network.speed()

        # Bias output layer (traditional linear layer for final output)
        self.bridge_bias = nn.Linear(n_neurons, stateSize)

        # Affine structure layer if needed
        if affineStruct:
            self.bridge_f = nn.Linear(n_neurons, stateSize * (stateSize + N_U))

    def forward(self, inputs_novelU, inputs_state):
        # Concatenate inputs
        inputConcat = torch.cat([inputs_state.float(), inputs_novelU.float()], dim=-1)

        # Pass through KAN network
        # KAN handles all the nonlinear transformations with learnable activation functions
        kan_output = self.kan_network(inputConcat)

        # Bias output
        bias = self.bridge_bias(kan_output)

        if self.affineStruct:
            # Affine structure computation
            ABunshape = self.bridge_f(kan_output)
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
            return out, kan_output, bias

    def apply_regularization(self):
        """
        Apply regularization losses. This should be called during training
        to add regularization terms to the loss function.
        """
        reg_loss = 0.0

        if self.kernel_regularizer is not None:
            # Apply regularization to KAN parameters
            for name, param in self.kan_network.named_parameters():
                if self.useGroupLasso and self.stateReduction and "layers.0" in name:
                    # Use input layer regularizer for first layer if group lasso is enabled
                    if self.inputLayerRegularizer is not None:
                        reg_loss += self.inputLayerRegularizer * torch.norm(param)
                else:
                    reg_loss += self.kernel_regularizer * torch.norm(param)

            # Also regularize the final linear layers
            for name, param in self.named_parameters():
                if "bridge_bias" in name or "bridge_f" in name:
                    if "weight" in name:
                        reg_loss += self.kernel_regularizer * torch.norm(param)

        return reg_loss

    def apply_constraints(self):
        """
        Apply constraints to the input layer weights if specified.
        This should be called after each optimizer step.
        Note: For KAN networks, constraints are applied to the first layer's parameters.
        """
        if self.constraintOnInputHiddenLayer is not None:
            with torch.no_grad():
                # Apply constraint to the first KAN layer parameters
                # This is more complex for KAN as it doesn't have simple weight matrices
                for name, param in self.kan_network.named_parameters():
                    if "layers.0" in name and "weight" in name:
                        param.data = self.constraintOnInputHiddenLayer(param.data)

    def prune(self, threshold=1e-2):
        """
        Prune the KAN network by removing less important connections.
        This leverages KAN's built-in pruning capability.
        """
        return self.kan_network.prune(threshold=threshold)

    def plot(self, folder="./figures", beta=3, mask=False, mode="supervised", scale=0.8, tick=True, sample=True, in_vars=None, out_vars=None, title=None):
        """
        Visualize the KAN network structure and learned functions.
        This is one of the key advantages of KAN - interpretability.
        """
        return self.kan_network.plot(
            folder=folder, 
            beta=beta, 
            mask=mask, 
            mode=mode, 
            scale=scale, 
            tick=tick, 
            sample=sample, 
            in_vars=in_vars, 
            out_vars=out_vars, 
            title=title
        )

    def symbolic_formula(self, var=None):
        """
        Extract symbolic formulas from the trained KAN network.
        This provides mathematical interpretability of the learned function.
        """
        return self.kan_network.symbolic_formula(var=var)

    def speed(self):
        """
        Enable speed mode for faster training by disabling symbolic computations.
        Call this before training for better performance.
        """
        self.kan_network.speed()

    def set_mode(self, mode):
        """
        Set the mode of the KAN network.
        mode: 'train' or 'eval'
        """
        if mode == 'train':
            self.kan_network.train()
        else:
            self.kan_network.eval()


# Example usage and comparison with original network
if __name__ == "__main__":
    # Parameters
    stateSize = 10
    N_U = 5
    n_neurons = 20
    n_layer = 3
    batch_size = 32

    # Create KAN-based bridge network
    kan_bridge = KANBridgeNetwork(
        stateSize=stateSize,
        N_U=N_U,
        n_neurons=n_neurons,
        n_layer=n_layer,
        grid_size=5,
        spline_order=3,
        affineStruct=True
    )

    # Enable speed mode for training
    kan_bridge.speed()

    # Create sample inputs
    inputs_state = torch.randn(batch_size, stateSize)
    inputs_novelU = torch.randn(batch_size, N_U)

    # Forward pass
    output, features, bias = kan_bridge(inputs_novelU, inputs_state)

    print(f"Input state shape: {inputs_state.shape}")
    print(f"Input novelU shape: {inputs_novelU.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Bias shape: {bias.shape}")

    # Apply regularization
    reg_loss = kan_bridge.apply_regularization()
    print(f"Regularization loss: {reg_loss}")

    # Example training setup
    optimizer = torch.optim.Adam(kan_bridge.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Dummy target
    target = torch.randn(batch_size, stateSize)

    # Training step
    optimizer.zero_grad()
    loss = criterion(output, target) + reg_loss
    loss.backward()
    
    # Apply constraints if any
    kan_bridge.apply_constraints()
    
    optimizer.step()
    
    print(f"Total loss: {loss.item()}")
