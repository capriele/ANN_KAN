import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from l21 import L21Regularization


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def __call__(self, validation_loss):
        if (self.min_validation_loss - validation_loss) > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class DatasetLoadUtility:
    def loadDatasetFromMATfile(self, filename="-1"):
        dataset = scipy.io.loadmat(filename)
        U, Y, U_val, Y_val = [dataset.get(x) for x in ["U", "Y", "U_val", "Y_val"]]
        return U, Y, U_val, Y_val

    def loadFieldFromMATFile(self, filename, fields):
        dataset = scipy.io.loadmat(filename)
        return [dataset.get(x) for x in fields]


class ANNModel(nn.Module):
    def __init__(
        self,
        stride_len,
        max_range,
        n_y,
        n_u,
        output_window_len,
        bridgeNetwork,
        encoderNetwork,
        decoderNetwork,
    ):
        super(ANNModel, self).__init__()

        self.stride_len = stride_len
        self.max_range = max_range
        self.n_y = n_y
        self.n_u = n_u
        self.output_window_len = output_window_len

        self.bridgeNetwork = bridgeNetwork
        self.encoderNetwork = encoderNetwork
        self.decoderNetwork = decoderNetwork

    def forward(self, inputs_y, inputs_u):
        prediction_error_collection = []
        forward_error_collection = []
        forwarded_predicted_error_collection = []
        predicted_ok_collection = []
        state_k_collection = []

        forwarded_state = None

        for k in range(0, self.max_range):
            iy_k = torch.as_tensor(
                inputs_y[:, k : k + self.stride_len], dtype=torch.float32
            )
            iu_k = torch.as_tensor(
                inputs_u[:, k : k + self.stride_len], dtype=torch.float32
            )
            itarget_k = inputs_y[
                :,
                self.stride_len
                + k
                - self.output_window_len
                + 1 : self.stride_len
                + k
                + 1,
            ]
            novel_iu_k = torch.as_tensor(
                inputs_u[:, self.stride_len + k : self.stride_len + k + 1],
                dtype=torch.float32,
            )
            state_k = torch.as_tensor(
                self.encoderNetwork(torch.cat([iy_k, iu_k], dim=1)), dtype=torch.float32
            )
            predicted_ok = self.decoderNetwork(state_k)
            # Traspose the output if necessary
            if predicted_ok.shape[0] != itarget_k.shape[0]:
                predicted_ok = predicted_ok.T
            # if predicted_ok.dim() == 2:
            #    predicted_ok = predicted_ok[:, 0]
            if predicted_ok.dim() == 3:
                predicted_ok = predicted_ok[:, :, 0]
            # print("itarget_k output shape:", itarget_k.shape)
            # print("Predicted output shape:", predicted_ok.shape)

            predicted_ok_collection.append(predicted_ok)
            state_k_collection.append(state_k)

            prediction_error_k = torch.abs(predicted_ok - itarget_k)
            prediction_error_collection.append(prediction_error_k)

            if forwarded_state is not None:
                forwarded_state_n = [
                    self.bridgeNetwork(torch.cat([novel_iu_k, state_k], dim=1))[0]
                ]
                for this_f in forwarded_state:
                    forward_error_k = torch.abs(state_k[0] - this_f)
                    forward_error_collection.append(forward_error_k)

                    if state_k.dim() != this_f.dim():
                        # If dimensions are not the same, adjust this_f to match state_k
                        temp_f = this_f
                        for _ in range(1, state_k.shape[0]):
                            temp_f = torch.cat((temp_f, this_f), dim=0)

                    temp_f = torch.reshape(temp_f, state_k.shape)

                    forwarded_predicted_output_k = self.decoderNetwork(temp_f)[:, :, 0]
                    forwarded_predicted_error_k = torch.abs(
                        forwarded_predicted_output_k - itarget_k
                    )
                    forwarded_predicted_error_collection.append(
                        forwarded_predicted_error_k
                    )
                    forwarded_state_n.append(
                        self.bridgeNetwork(torch.cat([novel_iu_k, temp_f], dim=1))[0]
                    )
                forwarded_state = forwarded_state_n
            else:
                forwarded_state = [
                    self.bridgeNetwork(torch.cat([novel_iu_k, state_k], dim=1))[0]
                ]
                for this_f in forwarded_state:
                    forward_error_k = torch.abs(state_k[0] - this_f)
                    forward_error_collection.append(forward_error_k)

                    if state_k.dim() != this_f.dim():
                        # If dimensions are not the same, adjust this_f to match state_k
                        temp_f = this_f
                        for _ in range(1, state_k.shape[0]):
                            temp_f = torch.cat((temp_f, this_f), dim=0)

                    temp_f = torch.reshape(temp_f, state_k.shape)

                    forwarded_predicted_output_k = self.decoderNetwork(temp_f)
                    if forwarded_predicted_output_k.dim() == 3:
                        forwarded_predicted_output_k = forwarded_predicted_output_k[
                            :, :, 0
                        ]
                    forwarded_predicted_error_k = torch.abs(
                        forwarded_predicted_output_k - itarget_k
                    )
                    forwarded_predicted_error_collection.append(
                        forwarded_predicted_error_k
                    )

        one_step_ahead_prediction_error = torch.cat(prediction_error_collection, dim=1)

        if len(forwarded_predicted_error_collection) > 1:
            forwarded_predicted_error = torch.cat(
                forwarded_predicted_error_collection, dim=1
            )
        elif len(forwarded_predicted_error_collection) == 1:
            forwarded_predicted_error = torch.abs(
                forwarded_predicted_error_collection[0]
            )

        if len(forward_error_collection) > 1:
            forward_error = forward_error_collection[-1]
        elif len(forward_error_collection) == 1:
            forward_error = torch.abs(forward_error_collection[0])

        return (
            predicted_ok_collection[0],
            state_k_collection[0],
            one_step_ahead_prediction_error,
            forwarded_predicted_error,
            forward_error,
        )


class AdvAutoencoder(nn.Module):
    def __init__(
        self,
        nonlinearity="relu",
        n_neurons=30,
        n_layer=3,
        fitHorizon=5,
        useGroupLasso=True,
        stateReduction=False,
        validation_split=0.05,
        stateSize=-1,
        strideLen=10,
        outputWindowLen=2,
        affineStruct=True,
        regularizerWeight=0.0005,
    ):
        super(AdvAutoencoder, self).__init__()

        self.nonlinearity = nonlinearity
        self.outputWindowLen = outputWindowLen
        self.stateSize = stateSize
        self.n_neurons = n_neurons
        self.stateReduction = stateReduction
        self.validation_split = validation_split
        self.strideLen = strideLen
        self.n_layer = n_layer
        self.affineStruct = affineStruct
        self.MaxRange = fitHorizon
        self.regularizerWeight = regularizerWeight
        self.useGroupLasso = useGroupLasso
        self.model = None

    def mean_pred(self, y_pred, y_true):
        return torch.mean(y_pred**2)

    def encoderNetwork(self):
        layers = []
        input_size = self.strideLen * (self.N_U + self.N_Y)
        layers.append(nn.Linear(input_size, self.n_neurons))
        layers.append(self.get_activation(self.nonlinearity))

        for _ in range(self.n_layer - 1):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(self.get_activation(self.nonlinearity))

        layers.append(nn.Linear(self.n_neurons, self.stateSize))
        print("\n\nDecoder network", layers)
        return nn.Sequential(*layers)

    def decoderNetwork(self):
        layers = []
        layers.append(nn.Linear(self.stateSize, self.n_neurons))
        layers.append(self.get_activation(self.nonlinearity))

        for _ in range(self.n_layer - 1):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(self.get_activation(self.nonlinearity))

        if self.affineStruct:
            layers.append(
                nn.Linear(self.n_neurons, self.outputWindowLen * self.stateSize)
            )
            layers.append(nn.Unflatten(1, (self.outputWindowLen, self.stateSize)))
        else:
            layers.append(nn.Linear(self.n_neurons, self.outputWindowLen * self.N_Y))
        print("\n\nDecoder network", layers)
        return nn.Sequential(*layers)

    def bridgeNetwork(self):
        layers = []
        layers.append(nn.Linear(self.stateSize + self.N_U, self.n_neurons))
        layers.append(self.get_activation(self.nonlinearity))

        for _ in range(self.n_layer - 1):
            layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            layers.append(self.get_activation(self.nonlinearity))

        layers.append(nn.Linear(self.n_neurons, self.stateSize))
        print("\n\nBridge network", layers)
        return nn.Sequential(*layers)

    def ANNModel(self):
        return ANNModel(
            stride_len=self.strideLen,
            max_range=self.MaxRange,
            n_y=self.N_Y,
            n_u=self.N_U,
            output_window_len=self.outputWindowLen,
            bridgeNetwork=self.bridgeNetwork(),
            encoderNetwork=self.encoderNetwork(),
            decoderNetwork=self.decoderNetwork(),
        )

    def get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()

    def setDataset(self, U, Y, U_val, Y_val):
        if U is not None:
            self.U = U.copy()
            self.N_U = U.shape[1]
        if Y is not None:
            self.Y = Y.copy()
            self.N_Y = Y.shape[1]
        if U_val is not None:
            self.U_val = U_val.copy()
        if Y_val is not None:
            self.Y_val = Y_val.copy()

        if self.model is None:
            self.model = self.ANNModel()

    def prepareDataset(self, U=None, Y=None):
        if U is None:
            U = self.U
        if Y is None:
            Y = self.Y

        pad = self.MaxRange - 2
        strideLen = self.strideLen + pad
        lenDS = U.shape[0]
        inputVector = np.zeros((lenDS - 2, self.N_U * (strideLen + 2)))
        outputVector = np.zeros((lenDS - 2, self.N_Y * (strideLen + 2)))
        offset = self.strideLen + 1 + pad

        for i in range(offset, lenDS - 1):
            regressor_StateInputs = np.ravel(U[i - strideLen - 1 : i + 1])
            regressor_StateOutputs = np.ravel(Y[i - strideLen - 1 : i + 1])
            inputVector[i - offset] = regressor_StateInputs.copy()
            outputVector[i - offset] = regressor_StateOutputs.copy()

        return (
            torch.tensor(inputVector[: i - offset + 1].copy()),
            torch.tensor(outputVector[: i - offset + 1].copy()),
        )

    def compute_gradients(self, model, train_stateVector, train_inputVector):
        train_stateVector.requires_grad_(True)
        train_inputVector.requires_grad_(True)
        outputs = model(train_stateVector, train_inputVector)
        loss = sum(output.abs().sum() for output in outputs)
        loss.backward()
        gradients = [param.grad for param in model.parameters()]
        return gradients

    def trainModel(
        self,
        batch_size=32,
        validation_split=0.2,
        epochs=150,
    ):
        inputVector, outputVector = self.prepareDataset()
        print("Input vector shape:", inputVector.shape)
        print("Output vector shape:", outputVector.shape)
        dataset = TensorDataset(outputVector, inputVector)
        train_size = int((1 - validation_split) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=0.002, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.3, patience=3, min_lr=0.001
        )
        early_stopping = EarlyStopping(patience=8, min_delta=0.001)

        for epoch in range(epochs):
            self.model.train()
            for batch_output, batch_input in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_output, batch_input)
                loss = sum(output.abs().sum() for output in outputs)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_input, batch_output in test_loader:
                    outputs = self.model(batch_input, batch_output)
                    val_loss += sum(output.abs().sum() for output in outputs)

            scheduler.step(val_loss)
            if early_stopping(val_loss):
                break

    def getModel(self):
        return self.model

    def evaluateNetwork(self, U_val, Y_val):
        inputVector, outputVector = self.prepareDataset(U_val, Y_val)
        with torch.no_grad():
            output = self(torch.FloatTensor(inputVector))
        return output.numpy(), outputVector, 0

    def validateModel(self, plot=True):
        fitted_Y, train_outputVector, _ = self.evaluateNetwork(self.U_val, self.Y_val)
        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(fitted_Y)
            plt.plot(train_outputVector)
            plt.show()
        return fitted_Y, train_outputVector, 0


# Example usage:
# utility = DatasetLoadUtility()
# U, Y, U_val, Y_val = utility.loadDatasetFromMATfile("your_dataset.mat")
# model = AdvAutoencoder()
# model.setDataset(U, Y, U_val, Y_val)
# model.trainModel()
# fitted_Y, train_outputVector, _ = model.validateModel()
