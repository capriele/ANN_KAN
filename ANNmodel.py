import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class EncoderNetwork(nn.Module):
    """Encoder network for the classical ANN model."""

    def __init__(
        self,
        stride_len: int,
        n_u: int,
        n_y: int,
        n_neurons: int,
        n_layer: int,
        state_size: int,
        nonlinearity: str = "relu",
        kernel_regularizer: Optional[str] = None,
        constraint_on_input_hidden_layer: Optional[str] = None,
        use_group_lasso: bool = False,
        state_reduction: bool = False,
        input_layer_regularizer: Optional[str] = None,
        future: int = 0,
    ):
        super().__init__()
        self.stride_len = stride_len
        self.n_u = n_u
        self.n_y = n_y
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.state_size = state_size
        self.future = future
        self.activation = self._get_activation(nonlinearity)

        input_dim = (stride_len * n_u) + (stride_len * n_y)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, n_neurons))
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Linear(n_neurons, state_size))

        # self._init_weights()

    def _init_weights(self):
        """Initialize all Linear layers with Xavier init."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_activation(self, nonlinearity: str) -> nn.Module:
        """Return activation function based on input string."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(nonlinearity, nn.ReLU())

    def forward(self, inputs_y: torch.Tensor, inputs_u: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.cat(
            [inputs_y.float().to(device), inputs_u.float().to(device)], dim=-1
        ).to(device)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DecoderNetwork(nn.Module):
    """Decoder network for the classical ANN model."""

    def __init__(
        self,
        state_size: int,
        n_neurons: int,
        n_layer: int,
        nonlinearity: str = "relu",
        output_window_len: int = 1,
        N_Y: int = 1,
        affine_struct: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.state_size = state_size
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.output_window_len = output_window_len
        self.N_Y = N_Y
        self.affine_struct = affine_struct
        self.activation = self._get_activation(nonlinearity)

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, n_neurons))
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        out_dim = (
            output_window_len * state_size if affine_struct else output_window_len * N_Y
        )
        self.final_layer = nn.Linear(n_neurons, out_dim)

        # self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in list(self.layers) + [self.final_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _get_activation(self, nonlinearity: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(nonlinearity, nn.ReLU())

    def forward(self, inputs_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        x = inputs_state.to(device)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.final_layer(x)
        if self.affine_struct:
            x = x.view(-1, self.output_window_len, self.state_size)
            # out = torch.sum(x * inputs_state.unsqueeze(1), dim=1)
            out = torch.sum(x * inputs_state.unsqueeze(1), dim=1)
            # out = torch.mul(x, inputs_state.unsqueeze(1))
            return x, out
        return x, x


class BridgeNetwork(nn.Module):
    """Bridge network for the classical ANN model."""

    def __init__(
        self,
        state_size: int,
        N_U: int,
        n_neurons: int,
        n_layer: int,
        nonlinearity: str = "relu",
        affine_struct: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.state_size = state_size
        self.N_U = N_U
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.affine_struct = affine_struct
        self.activation = self._get_activation(nonlinearity)

        self.bridge0 = nn.Linear(state_size + N_U, n_neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(n_neurons, n_neurons) for _ in range(n_layer - 1)]
        )
        self.bridge_bias = nn.Linear(n_neurons, state_size)
        if affine_struct:
            self.bridge_f = nn.Linear(n_neurons, state_size * (state_size + N_U))

        # self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""
        for layer in [self.bridge0, self.bridge_bias] + list(self.hidden_layers):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if self.affine_struct:
            nn.init.xavier_normal_(self.bridge_f.weight)
            nn.init.zeros_(self.bridge_f.bias)

    def _get_activation(self, nonlinearity: str) -> callable:
        """Return activation function based on input string."""
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return activations.get(nonlinearity, F.relu)

    def forward(self, inputs_novelU: torch.Tensor, inputs_state: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = next(self.parameters()).device
        input_concat = torch.cat(
            [inputs_state.float().to(device), inputs_novelU.float().to(device)], dim=-1
        ).to(device)
        x = self.activation(self.bridge0(input_concat))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        bias = self.bridge_bias(x)
        if self.affine_struct:
            AB = self.bridge_f(x).view(-1, self.state_size, self.state_size + self.N_U)
            input_concat_expanded = input_concat.unsqueeze(-1)
            out = torch.bmm(AB, input_concat_expanded).squeeze(-1) + bias.view(
                AB.shape[0], self.state_size
            )
            return out, AB, bias
        return bias, x, bias


class ANNModel(nn.Module):
    """Main ANN model integrating Encoder, Decoder, and Bridge networks."""

    def __init__(
        self,
        stride_len: int,
        max_range: int,
        n_y: int,
        n_u: int,
        output_window_len: int,
        encoder_network: EncoderNetwork,
        decoder_network: DecoderNetwork,
        bridge_network: BridgeNetwork,
    ):
        super().__init__()
        self.stride_len = stride_len
        self.max_range = max_range
        self.n_y = n_y
        self.n_u = n_u
        self.output_window_len = output_window_len
        self.conv_encoder = encoder_network
        self.output_decoder = decoder_network
        self.bridge_network = bridge_network

        if self.output_window_len <= 0:
            self.output_window_len = 2

    def forward(
        self, inputs_y: torch.Tensor, inputs_u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        device = next(self.parameters()).device
        inputs_y = inputs_y.to(device)
        inputs_u = inputs_u.to(device)

        batch_size = inputs_y.size(0)
        (
            prediction_error_collection,
            forward_error_collection,
            forwarded_predicted_error_collection,
        ) = ([], [], [])
        predicted_ok_collection, state_k_collection = [], []
        forwarded_state = None

        total_length_y = inputs_y.shape[1]
        total_length_u = inputs_u.shape[1]
        for k in range(self.max_range):
            # Slice the input sequences for y and u
            start_y = (self.n_y * k) % total_length_y
            end_y = (self.n_y * k + self.n_y * self.stride_len) % total_length_y

            # Handle the case where the indices wrap around
            if start_y < end_y:
                i_yk = inputs_y[:, start_y:end_y]
            else:
                i_yk = torch.cat((inputs_y[:, start_y:], inputs_y[:, :end_y]), dim=1)

            # Calculate the start and end indices for i_uk with modulo
            start_u = (self.n_u * k) % total_length_u
            end_u = (self.n_u * k + self.n_u * self.stride_len) % total_length_u

            # Handle the case where the indices wrap around
            if start_u < end_u:
                i_uk = inputs_u[:, start_u:end_u]
            else:
                i_uk = torch.cat((inputs_u[:, start_u:], inputs_u[:, :end_u]), dim=1)

            # Calculate target indices for y (output size is now n_u)
            # target_start = (self.stride_len + k + 1) % inputs_y.shape[1]
            # target_end = (
            #     self.stride_len + k + 1 + self.n_y * self.output_window_len
            # ) % inputs_y.shape[
            #     1
            # ]  # Output size is n_y
            target_start = self.n_y * k % inputs_y.shape[1]
            target_end = self.n_y * (k + 1) % inputs_y.shape[1]

            # Handle wrap-around for i_target_k
            if target_end < target_start:
                i_target_k = torch.cat(
                    (inputs_y[:, target_start:], inputs_y[:, :target_end]), dim=1
                )
            else:
                i_target_k = inputs_y[:, target_start:target_end]

            # Calculate novel input indices for u (input size is now n_u)
            novel_start = (self.n_u * k) % inputs_u.shape[1]
            novel_end = (self.n_u * (k + 1)) % inputs_u.shape[1]  # Input size is n_u

            # Handle wrap-around for novel_i_uk
            if novel_end < novel_start:
                novel_i_uk = torch.cat(
                    (inputs_u[:, novel_start:], inputs_u[:, :novel_end]), dim=1
                )
            else:
                novel_i_uk = inputs_u[:, novel_start:novel_end]

            # Encode and decode
            # print(inputs_y.shape)
            # print(inputs_u.shape)
            # print(target_start, target_end)
            # print(i_target_k.shape)
            # print(novel_i_uk.shape)
            state_k = self.conv_encoder(i_yk, i_uk)
            predicted_ok = self.output_decoder(state_k)[1]
            predicted_ok_collection.append(predicted_ok)
            state_k_collection.append(state_k)
            # print(predicted_ok.shape)
            i_target_k = torch.reshape(i_target_k, predicted_ok.shape)
            prediction_error_collection.append(torch.abs(predicted_ok - i_target_k))

            if forwarded_state is not None:
                forwarded_state_n = []
                bridge_output = self.bridge_network(novel_i_uk, state_k)[0]
                forwarded_state_n.append(bridge_output)
                for this_f in forwarded_state:
                    forward_error_collection.append(torch.abs(state_k - this_f))
                    forwarded_predicted_output_k = self.output_decoder(this_f)[1]
                    forwarded_predicted_error_collection.append(
                        forwarded_predicted_output_k - i_target_k
                    )
                    bridge_output_f = self.bridge_network(novel_i_uk, this_f)[0]
                    forwarded_state_n.append(bridge_output_f)
                forwarded_state = forwarded_state_n
            else:
                bridge_output = self.bridge_network(novel_i_uk, state_k)[0]
                forwarded_state = [bridge_output]

        one_step_ahead_prediction_error = torch.cat(prediction_error_collection, dim=1)
        forwarded_predicted_error = (
            torch.cat(forwarded_predicted_error_collection, dim=1)
            if forwarded_predicted_error_collection
            else torch.zeros_like(one_step_ahead_prediction_error[:, :1]).to(device)
        )
        forward_error = (
            torch.cat(forward_error_collection, dim=1)
            if forward_error_collection
            else torch.zeros_like(one_step_ahead_prediction_error[:, :1]).to(device)
        )

        return (
            predicted_ok_collection[0],
            state_k_collection[0],
            one_step_ahead_prediction_error,
            forwarded_predicted_error,
            forward_error,
        )
