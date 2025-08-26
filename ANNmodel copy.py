import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


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


class ANNModel(nn.Module):

    def __init__(
        self,
        stride_len,
        max_range,
        n_y,
        n_u,
        output_window_len,
        encoder_network,
        decoder_network,
        bridge_network,
    ):
        super(ANNModel, self).__init__()

        self.stride_len = stride_len
        self.max_range = max_range
        self.n_y = n_y
        self.n_u = n_u
        self.output_window_len = output_window_len

        # Store the network components
        self.conv_encoder = encoder_network
        self.output_decoder = decoder_network
        self.bridge_network = bridge_network

    def forward(self, inputs_y, inputs_u):
        """
        Forward pass of the ANN model

        Args:
            inputs_y: Input tensor of shape (batch_size, (stride_len + max_range) * n_y)
            inputs_u: Input tensor of shape (batch_size, (stride_len + max_range) * n_u)

        Returns:
            tuple: (predicted_ok_first, state_k_first, one_step_error, forwarded_error, forward_error)
        """
        batch_size = inputs_y.size(0)

        prediction_error_collection = []
        forward_error_collection = []
        forwarded_predicted_error_collection = []
        predicted_ok_collection = []
        state_k_collection = []

        forwarded_state = None

        for k in range(self.max_range):
            # Extract input slices for current step k
            i_yk = inputs_y[:, k : self.stride_len + k]
            i_uk = inputs_u[:, k : self.stride_len + k]

            # Extract target slice
            target_start = self.stride_len + k - self.output_window_len + 1
            target_end = self.stride_len + k + 1
            i_target_k = inputs_y[:, target_start:target_end]

            # Extract novel input
            novel_i_uk = inputs_u[:, self.stride_len + k : self.stride_len + k + 1]

            # Encode current state
            state_k = self.conv_encoder(i_yk, i_uk)

            # Decode to get prediction
            predicted_ok = self.output_decoder(state_k)
            if isinstance(predicted_ok, (list, tuple)):
                predicted_ok = predicted_ok[1]

            predicted_ok_collection.append(predicted_ok)
            state_k_collection.append(state_k)

            # Compute prediction error
            prediction_error_k = torch.abs(predicted_ok - i_target_k)
            prediction_error_collection.append(prediction_error_k)

            # Handle forward prediction
            if forwarded_state is not None:
                forwarded_state_n = []

                # Add bridge network output for current state
                bridge_output = self.bridge_network(novel_i_uk, state_k)
                if isinstance(bridge_output, (list, tuple)):
                    bridge_output = bridge_output[0]
                forwarded_state_n.append(bridge_output)

                # Process each forwarded state
                for this_f in forwarded_state:
                    # Compute forward error
                    if isinstance(state_k, (list, tuple)):
                        current_state = state_k[0]
                    else:
                        current_state = state_k

                    forward_error_k = torch.abs(current_state - this_f)
                    forward_error_collection.append(forward_error_k)

                    # Compute forwarded predicted output
                    forwarded_predicted_output_k = self.output_decoder(this_f)
                    if isinstance(forwarded_predicted_output_k, (list, tuple)):
                        forwarded_predicted_output_k = forwarded_predicted_output_k[1]

                    # Compute forwarded prediction error
                    forwarded_predicted_error_k = (
                        forwarded_predicted_output_k - i_target_k
                    )
                    forwarded_predicted_error_collection.append(
                        forwarded_predicted_error_k
                    )

                    # Add bridge network output for forwarded state
                    bridge_output_f = self.bridge_network(novel_i_uk, this_f)
                    if isinstance(bridge_output_f, (list, tuple)):
                        bridge_output_f = bridge_output_f[0]
                    forwarded_state_n.append(bridge_output_f)

                forwarded_state = forwarded_state_n
            else:
                # Initialize forwarded state
                bridge_output = self.bridge_network(novel_i_uk, state_k)
                if isinstance(bridge_output, (list, tuple)):
                    bridge_output = bridge_output[0]
                forwarded_state = [bridge_output]

        # Concatenate prediction errors
        one_step_ahead_prediction_error = torch.cat(prediction_error_collection, dim=1)

        # Handle forwarded predicted errors
        if len(forwarded_predicted_error_collection) > 1:
            forwarded_predicted_error = torch.cat(
                forwarded_predicted_error_collection, dim=1
            )
        elif len(forwarded_predicted_error_collection) == 1:
            forwarded_predicted_error = torch.abs(
                forwarded_predicted_error_collection[0]
            )
        else:
            # Create a dummy tensor if no forwarded errors
            forwarded_predicted_error = torch.zeros_like(
                one_step_ahead_prediction_error[:, :1]
            )

        # Handle forward errors
        if len(forward_error_collection) > 1:
            forward_error = torch.cat(forward_error_collection, dim=1)
        elif len(forward_error_collection) == 1:
            forward_error = torch.abs(forward_error_collection[0])
        else:
            # Create a dummy tensor if no forward errors
            forward_error = torch.zeros_like(one_step_ahead_prediction_error[:, :1])

        return (
            predicted_ok_collection[0],
            state_k_collection[0],
            one_step_ahead_prediction_error,
            forwarded_predicted_error,
            forward_error,
        )
