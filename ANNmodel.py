import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


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
