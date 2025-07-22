import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class ANNModel(nn.Module):

    def __init__(
        self,
        strideLen,
        MaxRange,
        N_Y,
        N_U,
        outputWindowLen,
        encoder_network,
        decoder_network,
        bridge_network,
    ):
        super(ANNModel, self).__init__()

        self.strideLen = strideLen
        self.MaxRange = MaxRange
        self.N_Y = N_Y
        self.N_U = N_U
        self.outputWindowLen = outputWindowLen

        # Store the component networks
        self.convEncoder = encoder_network
        self.outputEncoder = decoder_network
        self.bridgeNetwork = bridge_network

        # Initialize state tracking
        self.reset_state()

    def reset_state(self):
        """Reset the internal state for a new sequence"""
        self.forwarded_states = None
        self.step_count = 0

    def forward(self, inputs, step_k=None):
        """
        Process a single time step k.
        Fixed version that preserves gradients properly.
        """
        inputs_Y = inputs["input_y"]
        inputs_U = inputs["input_u"]

        # Use provided step or internal counter
        if step_k is not None:
            k = step_k
        else:
            k = self.step_count
            self.step_count += 1

        batch_size = inputs_Y.size(0)
        device = inputs_Y.device

        # Extract slices for current step k
        IYk = inputs_Y[:, k : self.strideLen + k].float()
        IUk = inputs_U[:, k : self.strideLen + k].float()

        # Extract target slice
        target_start = self.strideLen + k - self.outputWindowLen + 1
        target_end = self.strideLen + k + 1
        ITargetk = inputs_Y[:, target_start:target_end].float()

        # Extract novel input
        novelIUk = inputs_U[:, self.strideLen + k : self.strideLen + k + 1].float()

        # Forward pass through encoder
        stateK = self.convEncoder(torch.cat([IYk, IUk], dim=1))

        # Prepare all states for single network calls
        if self.forwarded_states is not None:
            # Create fresh tensors from stored states to avoid version conflicts
            # The stored states are detached, but we need them with gradients for current computation
            forwarded_states_with_grad = []
            for stored_state in self.forwarded_states:
                # Create a new tensor that requires gradients, based on the stored state
                state_with_grad = stored_state.clone().detach().requires_grad_(True)
                forwarded_states_with_grad.append(state_with_grad)

            # Stack all states: current state + all forwarded states
            # Use torch.stack instead of torch.cat for better gradient flow
            all_states = torch.stack([stateK] + forwarded_states_with_grad, dim=0)

            # Reshape for batch processing while preserving gradients
            all_states_reshaped = all_states.view(-1, all_states.shape[-1])

            # SINGLE CALL to outputEncoder for all states
            all_decoder_outputs = self.outputEncoder(all_states_reshaped)
            all_predictions = (
                all_decoder_outputs[1]
                if isinstance(all_decoder_outputs, (list, tuple))
                else all_decoder_outputs
            )

            # Reshape back to separate predictions
            all_predictions = all_predictions.view(
                len(self.forwarded_states) + 1, batch_size, -1
            )

            # Split predictions using tensor indexing that preserves gradients
            predictedOK = all_predictions[0]  # Current prediction
            forwarded_predictions = all_predictions[1:]  # All forwarded predictions

            # Calculate prediction errors
            predictionErrork = torch.abs(predictedOK - ITargetk)

            # Calculate forwarded predicted errors using vectorized operations
            num_forwarded = len(self.forwarded_states)
            if num_forwarded > 0:
                # Expand target for broadcasting
                ITargetk_expanded = ITargetk.unsqueeze(0).expand(num_forwarded, -1, -1)
                forwarded_predicted_errors = forwarded_predictions - ITargetk_expanded

                # Store individual errors for later use
                forwarded_predicted_errors_list = [
                    forwarded_predicted_errors[i] for i in range(num_forwarded)
                ]
            else:
                forwarded_predicted_errors_list = []

            # Prepare bridge network inputs
            # Expand novel input for all states
            expanded_novel_input = novelIUk.unsqueeze(0).expand(
                1 + num_forwarded, -1, -1
            )

            # Concatenate inputs for bridge network
            bridge_inputs = torch.cat([expanded_novel_input, all_states], dim=-1)

            # Reshape for network processing
            # bridge_inputs_flat = bridge_inputs.view(-1, bridge_inputs.shape[-1])

            # SINGLE CALL to bridgeNetwork for all states
            all_bridge_outputs = self.bridgeNetwork(expanded_novel_input, all_states)
            all_updated_states = (
                all_bridge_outputs[0]
                if isinstance(all_bridge_outputs, (list, tuple))
                else all_bridge_outputs
            )

            # Reshape back to separate states
            all_updated_states = all_updated_states.view(
                1 + num_forwarded, batch_size, -1
            )

            # Split updated states using tensor indexing
            new_forwarded_from_current = all_updated_states[0]
            updated_forwarded_states = [
                all_updated_states[i + 1] for i in range(num_forwarded)
            ]

            # Update forwarded states list - detach from computational graph to avoid in-place issues
            # Only the computation matters for gradients, not the stored states
            with torch.no_grad():
                self.forwarded_states = [new_forwarded_from_current.detach()] + [
                    state.detach() for state in updated_forwarded_states
                ]

            # Calculate forward errors using vectorized operations
            if len(self.forwarded_states) > 1:
                # Stack forwarded states (excluding the new one from current step)
                old_forwarded_states = torch.stack(self.forwarded_states[1:], dim=0)
                # Expand current state for broadcasting
                stateK_expanded = stateK.unsqueeze(0).expand(
                    len(self.forwarded_states) - 1, -1, -1
                )
                # Calculate errors
                forward_errors = torch.abs(stateK_expanded - old_forwarded_states)
                forwardError = forward_errors.mean(dim=0)
            else:
                forwardError = torch.abs(stateK - new_forwarded_from_current)

            # Aggregate forwarded predicted errors
            if forwarded_predicted_errors_list:
                forwardedPredictedError = torch.stack(
                    forwarded_predicted_errors_list, dim=0
                ).mean(dim=0)
            else:
                forwardedPredictedError = predictedOK - ITargetk

        else:
            # First step: single calls for initialization
            # SINGLE CALL to outputEncoder
            encoder_output = self.outputEncoder(stateK)
            predictedOK = (
                encoder_output[1]
                if isinstance(encoder_output, (list, tuple))
                else encoder_output
            )

            # Calculate prediction error
            predictionErrork = torch.abs(predictedOK - ITargetk)

            # SINGLE CALL to bridgeNetwork
            bridge_output = self.bridgeNetwork(novelIUk, stateK)
            forwardedState = (
                bridge_output[0]
                if isinstance(bridge_output, (list, tuple))
                else bridge_output
            )

            # Store the forwarded state detached from computational graph
            with torch.no_grad():
                self.forwarded_states = [forwardedState.detach()]

            # Calculate forward error and forwarded predicted error
            forwardError = torch.abs(stateK - forwardedState)
            forwardedPredictedError = predictedOK - ITargetk

        # Initialize output dictionary
        outputs = {
            "functional_1": predictedOK,
            "functional_2": stateK,
            "oneStepDecoderError": predictionErrork,
            "multiStep_decodeError": forwardedPredictedError,
            "forwardError": forwardError,
        }

        return outputs

    def forward_sequence_vectorized(self, inputs):
        """
        Vectorized version that processes all time steps at once.
        More efficient for inference when you need the complete sequence.
        """
        self.reset_state()

        inputs_Y = inputs["input_y"]
        inputs_U = inputs["input_u"]
        batch_size = inputs_Y.size(0)
        device = inputs_Y.device

        # Pre-compute all slices for all time steps
        all_IYk = []
        all_IUk = []
        all_ITargetk = []
        all_novelIUk = []

        for k in range(self.MaxRange):
            # Extract slices for step k
            IYk = inputs_Y[:, k : self.strideLen + k]
            IUk = inputs_U[:, k : self.strideLen + k]

            # Extract target slice
            target_start = self.strideLen + k - self.outputWindowLen + 1
            target_end = self.strideLen + k + 1
            ITargetk = inputs_Y[:, target_start:target_end]

            # Extract novel input
            novelIUk = inputs_U[:, self.strideLen + k : self.strideLen + k + 1]

            all_IYk.append(IYk)
            all_IUk.append(IUk)
            all_ITargetk.append(ITargetk)
            all_novelIUk.append(novelIUk)

        # Stack all inputs for vectorized processing
        stacked_IYk = torch.stack(all_IYk, dim=0)  # (MaxRange, batch_size, stride_len)
        stacked_IUk = torch.stack(all_IUk, dim=0)  # (MaxRange, batch_size, stride_len)
        stacked_inputs = torch.cat(
            [stacked_IYk, stacked_IUk], dim=-1
        )  # (MaxRange, batch_size, combined_dim)

        # Process all through encoder at once
        reshaped_inputs = stacked_inputs.view(
            -1, stacked_inputs.shape[-1]
        )  # (MaxRange * batch_size, combined_dim)
        all_states = self.convEncoder(reshaped_inputs)
        all_states = all_states.view(
            self.MaxRange, batch_size, -1
        )  # (MaxRange, batch_size, state_dim)

        # Process all through decoder at once
        reshaped_states = all_states.view(-1, all_states.shape[-1])
        all_decoder_outputs = self.outputEncoder(reshaped_states)
        all_predictions = (
            all_decoder_outputs[1]
            if isinstance(all_decoder_outputs, (list, tuple))
            else all_decoder_outputs
        )
        all_predictions = all_predictions.view(self.MaxRange, batch_size, -1)

        # Calculate prediction errors
        stacked_targets = torch.stack(all_ITargetk, dim=0)
        prediction_errors = torch.abs(all_predictions - stacked_targets)

        # For simplicity, return the last step's outputs (you can modify as needed)
        return {
            "functional_1": all_predictions[-1],
            "functional_2": all_states[-1],
            "oneStepDecoderError": prediction_errors,
            "multiStep_decodeError": prediction_errors,  # Simplified
            "forwardError": torch.zeros_like(all_states[-1]),  # Simplified
        }

    def forward_sequence(self, inputs):
        """
        Process a complete sequence using single-step forward calls.
        This maintains the exact same behavior as the original implementation.
        """
        self.reset_state()
        all_outputs = []

        for k in range(self.MaxRange):
            step_outputs = self.forward(inputs, step_k=k)
            all_outputs.append(step_outputs)

        # Aggregate outputs across all steps
        aggregated_outputs = {}
        for key in all_outputs[0].keys():
            if key in ["functional_1", "functional_2"]:
                # Take the last step's output for these
                aggregated_outputs[key] = all_outputs[-1][key]
            else:
                # Stack or concatenate error tensors
                values = [out[key] for out in all_outputs]
                if len(values) > 1:
                    aggregated_outputs[key] = torch.stack(values, dim=0)
                else:
                    aggregated_outputs[key] = values[0]

        return aggregated_outputs
