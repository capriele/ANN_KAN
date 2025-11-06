import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from kan import KAN


class EncoderNetwork(nn.Module):
    """Encoder network for the KAN-based ANN model."""

    def __init__(
        self,
        stride_len: int,
        n_u: int,
        n_y: int,
        n_neurons: int,
        n_layer: int,
        state_size: int,
        nonlinearity: str = "relu",
        grid_size: int = 5,
        spline_order: int = 3,
        noise_scale: float = 0.1,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.stride_len = stride_len
        self.n_u = n_u
        self.n_y = n_y
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.state_size = state_size
        input_dim = (stride_len * n_u) + (stride_len * n_y)
        width = (
            [input_dim] + [n_neurons] * (n_layer - 1) + [state_size]
            if n_layer > 1
            else [input_dim, state_size]
        )
        self.kan_network = KAN(
            base_fun="identity",
            width=width,
            grid=grid_size,
            k=spline_order,
            noise_scale=noise_scale,
            seed=seed,
            auto_save=False,
            symbolic_enabled=False,
        )
        # self.kan_network.speed(compile=True)

    def prune(self, threshold: float = 1e-2) -> None:
        self.kan_network.prune()

    def symbolic(self, lib=None) -> None:
        self.kan_network.symbolic_enabled = True
        self.kan_network.auto_symbolic()
        self.kan_network.symbolic_enabled = False

    def get_formulas(self):
        self.kan_network.plot()
        self.kan_network.symbolic_enabled = True
        formulas = self.kan_network.symbolic_formula()[0]
        self.kan_network.symbolic_enabled = False
        return formulas

    def forward(self, inputs_y: torch.Tensor, inputs_u: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.cat(
            [inputs_y.float().to(device), inputs_u.float().to(device)], dim=-1
        ).to(device)
        return self.kan_network.forward(x)


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

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, n_neurons))
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        out_dim = (
            output_window_len * state_size * self.N_Y
            if affine_struct
            else output_window_len * N_Y
        )
        self.final_layer = nn.Linear(n_neurons, out_dim)

    def forward(self, inputs_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        x = inputs_state.to(device)
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        if self.affine_struct:
            x = x.view(-1, self.output_window_len, self.N_Y, self.state_size)
            out = torch.sum(x * inputs_state.unsqueeze(1).unsqueeze(1), dim=-1)
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

        self.bridge0 = nn.Linear(state_size + N_U, n_neurons)
        self.bridge_bias = nn.Linear(n_neurons, state_size)
        if affine_struct:
            self.bridge_f = nn.Linear(n_neurons, state_size * (state_size + N_U))

    def forward(self, inputs_novelU: torch.Tensor, inputs_state: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = next(self.parameters()).device
        input_concat = torch.cat(
            [inputs_state.float().to(device), inputs_novelU.float().to(device)], dim=-1
        ).to(device)
        x = self.bridge0(input_concat)
        bias = self.bridge_bias(x)
        if self.affine_struct:
            AB = self.bridge_f(x).view(-1, self.state_size, self.state_size + self.N_U)
            out = torch.bmm(AB, input_concat.unsqueeze(-1)).squeeze(-1) + bias
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

        for k in range(self.max_range):
            i_yk = inputs_y[:, k : self.stride_len + k]
            i_uk = inputs_u[:, k : self.stride_len + k]
            target_start = self.stride_len + k - self.output_window_len + 1
            target_end = self.stride_len + k + 1
            i_target_k = inputs_y[:, target_start:target_end]
            novel_i_uk = inputs_u[:, self.stride_len + k : self.stride_len + k + 1]

            state_k = self.conv_encoder(
                i_yk.reshape(i_yk.shape[0], self.stride_len * self.n_y),
                i_uk.reshape(i_uk.shape[0], self.stride_len * self.n_u),
            )
            predicted_ok = self.output_decoder(state_k)[1]
            predicted_ok_collection.append(predicted_ok)
            state_k_collection.append(state_k)
            i_target_k = i_target_k.reshape(predicted_ok.shape)
            prediction_error_collection.append(torch.abs(predicted_ok - i_target_k))

            if forwarded_state is not None:
                forwarded_state_n = []
                bridge_output = self.bridge_network(
                    novel_i_uk.reshape(novel_i_uk.shape[0], self.n_u), state_k
                )[0]
                forwarded_state_n.append(bridge_output)
                for this_f in forwarded_state:
                    forward_error_collection.append(torch.abs(state_k - this_f))
                    forwarded_predicted_output_k = self.output_decoder(this_f)[1]
                    forwarded_predicted_error_collection.append(
                        forwarded_predicted_output_k - i_target_k
                    )
                    bridge_output_f = self.bridge_network(
                        novel_i_uk.reshape(novel_i_uk.shape[0], self.n_u), this_f
                    )[0]
                    forwarded_state_n.append(bridge_output_f)
                forwarded_state = forwarded_state_n
            else:
                bridge_output = self.bridge_network(
                    novel_i_uk.reshape(novel_i_uk.shape[0], self.n_u), state_k
                )[0]
                forwarded_state = [bridge_output]

        # Equivalent to: oneStepAheadPredictionError = keras.layers.concatenate(predictionErrorCollection, name='oneStepDecoderError')
        one_step_ahead_prediction_error = torch.cat(
            prediction_error_collection, dim=-1
        )  # Concatenate along the last dimension

        # Equivalent to the forwardedPredictedError logic
        if len(forwarded_predicted_error_collection) > 1:
            forwarded_predicted_error = torch.cat(
                forwarded_predicted_error_collection, dim=-1
            )
        else:
            forwarded_predicted_error = torch.abs(
                forwarded_predicted_error_collection[0]
            )

        # Equivalent to the forwardError logic
        if len(forward_error_collection) > 1:
            forward_error = torch.cat(forward_error_collection, dim=-1)
        else:
            forward_error = torch.abs(forward_error_collection[0])

        return (
            predicted_ok_collection[0],
            state_k_collection[0],
            one_step_ahead_prediction_error,
            forwarded_predicted_error,
            forward_error,
        )
