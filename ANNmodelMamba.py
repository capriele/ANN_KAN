import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)
        x = self.layer_norm(x)
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class S6Layer(nn.Module):
    def __init__(self, d_model, d_state, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand

        # Projection layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # State space parameters
        self.A_log = nn.Parameter(
            torch.log(
                1 + torch.arange(d_state, dtype=torch.float32).repeat(self.d_inner, 1)
            )
            / d_state
        )
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # Selection projection
        self.B_proj = nn.Linear(self.d_inner, d_state * self.d_inner, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state * self.d_inner, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        dt = F.softplus(self.dt_proj(x)) + 1e-4  # Avoid zero
        dt = torch.clamp(dt, min=-5, max=5)  # Tighter clamp
        A = -torch.exp(torch.clamp(self.A_log.float(), min=-5, max=5))
        D = torch.clamp(self.D.float(), min=0.1, max=2)  # Avoid near-zero D

        B = self.B_proj(x)
        C = self.C_proj(x)
        B = B.view(batch, seq_len, self.d_inner, self.d_state)
        C = C.view(batch, seq_len, self.d_inner, self.d_state)

        y = self.selective_scan(x, dt, A, B, C, D)
        y = self.out_proj(y * z)
        return y

    def selective_scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, d_inner, d_state = B.shape
        dtA = torch.exp(dt.unsqueeze(-1) * A)
        dtB = dt.unsqueeze(-1) * B
        x = torch.zeros(batch, d_inner, d_state, device=u.device)
        ys = []
        for i in range(seq_len):
            x = dtA[:, i] * x + dtB[:, i]
            x = torch.clamp(x, min=-1e6, max=1e6)
            y = (x * C[:, i]).sum(dim=-1)
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        return y


class EncoderNetwork(nn.Module):
    def __init__(
        self,
        stride_len: int,
        n_u: int,
        n_y: int,
        n_neurons: int,
        n_layer: int,
        state_size: int,
        nonlinearity: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.stride_len = stride_len
        self.n_u = n_u
        self.n_y = n_y
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.state_size = state_size
        self.activation = self._get_activation(nonlinearity)

        input_dim = (stride_len * n_u) + (stride_len * n_y)
        self.input_proj = nn.Linear(input_dim, input_dim)
        nh = input_dim // 64
        if nh == 0:
            nh = 1
        self.attention = MultiHeadAttention(input_dim, num_heads=min(1, nh))

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, n_neurons))
        for _ in range(n_layer - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
        self.layers.append(nn.Linear(n_neurons, state_size))

        self._initialize_weights()

    def _get_activation(self, nonlinearity: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "linear": nn.Identity(),
        }
        return activations.get(nonlinearity, nn.ReLU())

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:  # Skip 0D/1D tensors (biases/scalars)
                if "bias" in name:
                    nn.init.constant_(param, 0.01)  # Small bias
                continue
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.1)  # Smaller gain

    def forward(self, inputs_y: torch.Tensor, inputs_u: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.cat(
            [inputs_y.float().to(device), inputs_u.float().to(device)], dim=-1
        ).to(device)
        x = self.input_proj(x)
        x = x + self.attention(x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DecoderNetwork(nn.Module):
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
            output_window_len * state_size * self.N_Y
            if affine_struct
            else output_window_len * N_Y
        )
        self.final_layer = nn.Linear(n_neurons, out_dim)
        self._initialize_weights()

    def _get_activation(self, nonlinearity: str) -> nn.Module:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "linear": nn.Identity(),
        }
        return activations.get(nonlinearity, nn.ReLU())

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:  # Skip 0D/1D tensors (biases/scalars)
                if "bias" in name:
                    nn.init.constant_(param, 0.01)  # Small bias
                continue
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.1)  # Smaller gain

    def forward(self, inputs_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        x = inputs_state.to(device)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.final_layer(x)
        if self.affine_struct:
            x = x.view(-1, self.output_window_len, self.N_Y, self.state_size)
            out = torch.sum(x * inputs_state.unsqueeze(1).unsqueeze(1), dim=-1)
            return x, out
        return x, x


class BridgeNetwork(nn.Module):
    def __init__(
        self,
        state_size: int,
        N_U: int,
        n_neurons: int,
        n_layer: int,
        nonlinearity: str = "relu",
        affine_struct: bool = False,
        d_state: int = 1,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.state_size = state_size
        self.N_U = N_U
        self.n_neurons = n_neurons
        self.n_layer = n_layer
        self.affine_struct = affine_struct
        self.d_state = state_size
        self.d_conv = d_conv
        self.expand = expand

        self.bridge0 = S6Layer(state_size + N_U, state_size, d_conv, expand)
        self.bridge_bias = nn.Linear(state_size + N_U, state_size)
        if affine_struct:
            self.bridge_f = nn.Linear(state_size + N_U, state_size * (state_size + N_U))
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:  # Skip 0D/1D tensors (biases/scalars)
                if "bias" in name:
                    nn.init.constant_(param, 0.01)  # Small bias
                continue
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.1)  # Smaller gain

    def forward(self, inputs_novelU: torch.Tensor, inputs_state: torch.Tensor) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        device = next(self.parameters()).device
        input_concat = torch.cat(
            [inputs_state.float().to(device), inputs_novelU.float().to(device)], dim=-1
        ).to(device)
        x = self.bridge0(input_concat.unsqueeze(1)).squeeze(1)
        bias = self.bridge_bias(x)
        if self.affine_struct:
            AB = self.bridge_f(x).view(-1, self.state_size, self.state_size + self.N_U)
            out = torch.bmm(AB, input_concat.unsqueeze(-1)).squeeze(-1) + bias
            return out, AB, bias
        return bias, x, bias


class ANNModel(nn.Module):
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
