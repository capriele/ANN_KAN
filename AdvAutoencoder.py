import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from l21 import *
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path
from ANNmodel import *
from BridgeNetwork import *
from EncoderNetwork import *
from DecoderNetwork import *

torch.set_num_threads(8)
torch.set_num_interop_threads(4)

# log into a file and console
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,  # Set the logging level to INFO
    encoding="utf-8",
    # filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
    # handlers=[
    # logging.FileHandler("app.log"),
    # logging.StreamHandler()
    # ],
)


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
        self.model = None
        self.affineStruct = affineStruct
        self.MaxRange = fitHorizon
        self.regularizerWeight = regularizerWeight
        self.kernel_regularizer = l2(self.regularizerWeight)
        self.useGroupLasso = useGroupLasso
        self.shuffledIndexes = None
        self.constraintOnInputHiddenLayer = None
        self.batch_size = 24
        if useGroupLasso and regularizerWeight > 0.0:
            # self.constraintOnInputHiddenLayer=unit_norm();
            if stateReduction:
                self.inputLayerRegularizer = L21Regularization(
                    self.regularizerWeight, a=0, b=stateSize
                )
                # self.inputLayerRegularizer=l1(self.regularizerWeight)
            else:
                self.inputLayerRegularizer = L21Regularization(
                    self.regularizerWeight, a=strideLen, b=strideLen
                )
            #                self.inputLayerRegularizer=l1(self.regularizerWeight)
            self.kernel_regularizer = l2(0.0001)
            print(
                "=>if GroupLasso is used,  l2 regularizer is set to 0.0001 in all bridge, encoder, and decoder"
            )
            # print("=>if GroupLasso is used,  l2 regularizer is set to the same weigth in all bridge, encoder, and decoder")
        else:
            self.inputLayerRegularizer = self.kernel_regularizer

    def mean_pred(self, y_pred, y_true):
        return torch.mean(y_pred**2)

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

    def encoderNetwork(self, future=0):
        en = EncoderNetwork(
            stride_len=self.strideLen,
            n_u=self.N_U,
            n_y=self.N_Y,
            n_neurons=self.n_neurons,
            n_layer=self.n_layer,
            state_size=self.stateSize,
            nonlinearity=self.nonlinearity,
            kernel_regularizer=self.kernel_regularizer,
            constraint_on_input_hidden_layer=self.constraintOnInputHiddenLayer,
            use_group_lasso=self.useGroupLasso,
            state_reduction=self.stateReduction,
            input_layer_regularizer=self.inputLayerRegularizer,
            future=future,
        )
        return en

    def decoderNetwork(self, future=0):
        dn = DecoderNetwork(
            state_size=self.stateSize,
            n_neurons=self.n_neurons,
            n_layer=self.n_layer,
            nonlinearity=self.nonlinearity,
            output_window_len=self.outputWindowLen,
            N_Y=self.N_Y,
            affine_struct=self.affineStruct,
            kernel_regularizer=self.kernel_regularizer,
            use_group_lasso=self.useGroupLasso,
            state_reduction=self.stateReduction,
            input_layer_regularizer=self.inputLayerRegularizer,
            constraint_on_input_hidden_layer=self.constraintOnInputHiddenLayer,
            future=future,
        )
        return dn

    def bridgeNetwork(self, future=0):
        bn = BridgeNetwork(
            stateSize=self.stateSize,
            N_U=self.N_U,
            n_neurons=self.n_neurons,
            n_layer=self.n_layer,
            nonlinearity=self.nonlinearity,
            kernel_regularizer=self.kernel_regularizer,
            constraintOnInputHiddenLayer=self.constraintOnInputHiddenLayer,
            useGroupLasso=self.useGroupLasso,
            stateReduction=self.stateReduction,
            inputLayerRegularizer=self.inputLayerRegularizer,
            affineStruct=self.affineStruct,
            future=future,
        )
        return bn

    def ANNModel(self):
        bridgeNetwork = self.bridgeNetwork()
        convEncoder = self.encoderNetwork()
        outputEncoder = self.decoderNetwork()
        ann = ANNModel(
            stride_len=self.strideLen,
            max_range=self.MaxRange,
            n_y=self.N_Y,
            n_u=self.N_U,
            output_window_len=self.outputWindowLen,
            encoder_network=convEncoder,
            decoder_network=outputEncoder,
            bridge_network=bridgeNetwork,
        )
        logging.info(f"\n{ann}")
        return ann, convEncoder, outputEncoder, bridgeNetwork

    def get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()

    def prepareDataset(self, U=None, Y=None):
        if U is None:
            U = self.U
        if Y is None:
            Y = self.Y
        pad = self.MaxRange - 2
        strideLen = self.strideLen + pad
        print(strideLen)
        lenDS = U.shape[0]
        inputVector = np.zeros((lenDS - 2, self.N_U * (strideLen + 2)))
        outputVector = np.zeros((lenDS - 2, self.N_Y * (strideLen + 2)))
        offset = self.strideLen + 1 + pad

        for i in range(offset, lenDS):
            regressor_StateInputs = np.ravel(U[i - strideLen - 1 : i + 1])
            regressor_StateOutputs = np.ravel(Y[i - strideLen - 1 : i + 1])
            inputVector[i - offset] = regressor_StateInputs.copy()
            outputVector[i - offset] = regressor_StateOutputs.copy()

        return (
            torch.tensor(inputVector[: i - offset + 1].copy()),
            torch.tensor(outputVector[: i - offset + 1].copy()),
        )

    def trainModel(self, epochs=150, shuffled: bool = True):
        tmp = self.privateTrainModel(
            shuffled, None, kFPE=0.0, kAEPrediction=10, kForward=0.3, epochs=epochs
        )
        tmp = self.privateTrainModel(
            shuffled, tmp, 1, 0.0, 10, "checkpoints", None, epochs, 8, 0.001, True
        )

    def privateTrainModel(
        self,
        shuffled: bool = True,
        tmp=None,
        kFPE=0.0,
        kAEPrediction=10,
        kForward=0.3,
        checkpoint_path: Optional[str] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        epochs: int = 150,
        early_stopping_patience: int = 8,
        min_delta: float = 0.001,
        save_best_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model with improved error handling, logging, and checkpointing.

        Args:
            shuffled: Whether to shuffle training data
            checkpoint_path: Path to load model checkpoint from
            loss_weights: Dictionary of loss component weights
            epochs: Maximum number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            min_delta: Minimum change to qualify as improvement
            save_best_model: Whether to save the best model during training
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary containing training results and model state
        """

        try:
            # Prepare data
            logging.info("Preparing dataset...")
            inputVector, outputVector = self.prepareDataset()

            # Device setup with better detection
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
            # elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            #    device = torch.device("mps")
            #    logging.info("Using Apple Silicon GPU (MPS)")
            else:
                device = torch.device("cpu")
                logging.info("Using CPU")

            # Move data to device
            inputVector = inputVector.to(device, non_blocking=True)
            outputVector = outputVector.to(device, non_blocking=True)

            # Model initialization
            if checkpoint_path is not None and Path(checkpoint_path).exists():
                logging.info(f"Loading model from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(
                    f"{checkpoint_path}/best_model.pth", map_location=device
                )
                model = self.model
                model.load_state_dict(checkpoint["model_state_dict"])
                convEncoder = model.conv_encoder
                outputEncoder = model.output_decoder
                bridgeNetwork = model.bridge_network
                checkpoint_dir = Path(checkpoint_path)
            else:
                logging.info("Initializing new model...")
                model, convEncoder, outputEncoder, bridgeNetwork = self.ANNModel()

                # Update checkpoints path and create dir if not exists
                if checkpoint_path is None:
                    checkpoint_path = "checkpoints"
                checkpoint_dir = Path(checkpoint_path)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Move models to device
            model = model.to(device)
            convEncoder = convEncoder.to(device)
            outputEncoder = outputEncoder.to(device)
            bridgeNetwork = bridgeNetwork.to(device)

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logging.info(
                f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}"
            )

            # Optimizer setup with better defaults
            optimizer = optim.AdamW(  # AdamW often works better than Adam
                model.parameters(),
                lr=0.002,
                weight_decay=1e-4,  # L2 regularization
                amsgrad=True,
            )

            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.3,
                patience=3,
                threshold=min_delta,
                threshold_mode="abs",
                verbose=True,
                min_lr=1e-6,
            )

            # Prepare data splits
            data_size = outputVector.shape[0]
            if self.shuffledIndexes is None:
                self.shuffledIndexes = np.random.permutation(data_size)

            shuffledIndexes = self.shuffledIndexes
            input_y = outputVector[shuffledIndexes]
            input_u = inputVector[shuffledIndexes]

            # Prepare targets
            model_size = 6
            target_zeros = torch.zeros(data_size, model_size, device=device)

            # Split data
            val_split_idx = int(data_size * (1 - self.validation_split))

            # Training data
            train_input_y = input_y[:val_split_idx]
            train_input_u = input_u[:val_split_idx]
            train_targets = target_zeros[:val_split_idx]

            # Validation data
            val_input_y = input_y[val_split_idx:]
            val_input_u = input_u[val_split_idx:]
            val_targets = target_zeros[val_split_idx:]

            # Create DataLoaders with better settings
            train_dataset = TensorDataset(train_input_y, train_input_u, train_targets)
            val_dataset = TensorDataset(val_input_y, val_input_u, val_targets)

            # Use num_workers for faster data loading (adjust based on your system)
            num_workers = min(4, torch.get_num_threads())

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffled,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
                drop_last=True,  # Helps with batch norm stability
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if device.type == "cuda" else False,
            )

            # Default loss weights
            if loss_weights is None:
                loss_weights = {
                    "multiStep_decodeError": kFPE,
                    "oneStepDecoderError": kAEPrediction,
                    "forwardError": kForward,
                    "functional_1": 1.0,
                    "functional_2": 1.0,
                }

            logging.info(f"Loss weights: {loss_weights}")

            # Training setup
            criterion = nn.MSELoss()
            best_val_loss = float("inf")
            patience_counter = 0
            train_losses = []
            val_losses = []

            logging.info(f"Starting training for {epochs} epochs...")

            # Training loop
            for epoch in range(epochs):
                start_time = time.time()

                # === TRAINING PHASE ===
                model.train()
                train_loss = 0.0
                num_batches = 0

                for batch_idx, (
                    batch_input_y,
                    batch_input_u,
                    batch_targets,
                ) in enumerate(train_loader):
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(batch_input_y.detach(), batch_input_u.detach())

                    # Compute loss
                    output_dict = {
                        "multiStep_decodeError": outputs[3],
                        "oneStepDecoderError": outputs[1],
                        "forwardError": outputs[4],
                        "functional_1": outputs[0],
                        "functional_2": outputs[1],
                    }
                    batch_loss = self.calculate_weighted_loss(
                        output_dict, loss_weights, criterion
                    )

                    # Backward pass
                    batch_loss.backward()

                    # Optional: gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                    # Update weights
                    optimizer.step()

                    # Apply model-specific constraints and regularization
                    model.output_decoder.apply_regularization()
                    model.output_decoder.apply_constraints()
                    model.bridge_network.apply_constraints()

                    # Accumulate loss
                    train_loss += batch_loss.item()
                    num_batches += 1

                avg_train_loss = train_loss / max(num_batches, 1)
                train_losses.append(avg_train_loss)

                # === VALIDATION PHASE ===
                model.eval()
                with torch.no_grad():
                    avg_val_loss = self._validate_model(
                        model, val_loader, loss_weights, criterion
                    )
                    val_losses.append(avg_val_loss)

                # === LOGGING ===
                elapsed = time.time() - start_time
                logging.info(
                    f"Epoch [{epoch + 1}/{epochs}] | "
                    f"Train Loss: {avg_train_loss:.6f} | "
                    f"Val Loss: {avg_val_loss:.6f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )

                # === LEARNING RATE SCHEDULING ===
                scheduler.step(avg_val_loss)

                # === EARLY STOPPING & CHECKPOINTING ===
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0

                    if save_best_model and checkpoint_dir:
                        self._save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            best_val_loss,
                            checkpoint_dir / "best_model.pth",
                        )
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                if save_best_model and checkpoint_dir and (epoch + 1) % 10 == 0:
                    self._save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        avg_val_loss,
                        checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
                    )

        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            raise

        finally:
            # Clean up GPU memory
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Store models
        self.model = model
        self.convEncoder = convEncoder
        self.outputEncoder = outputEncoder
        self.bridgeNetwork = bridgeNetwork

        # Prepare results
        results = {
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

        logging.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return results

    @staticmethod
    def calculate_weighted_loss(outputs, loss_weights, criterion):
        """Calculate weighted loss from model outputs"""
        loss_components = []
        for output_name, weight in loss_weights.items():
            if output_name in outputs:
                # Ensure targets have the right shape for this output
                output_tensor = outputs[output_name]
                adjusted_targets = torch.zeros_like(outputs[output_name])
                loss_component = criterion(output_tensor, adjusted_targets)
                weighted_loss = weight * loss_component
                loss_components.append(weighted_loss)
        if loss_components:
            # Use torch.stack and sum instead of in-place addition
            total_loss = torch.stack(loss_components).sum()
        else:
            total_loss = torch.tensor(0.0, requires_grad=True)
        return total_loss

    def _validate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        loss_weights: Dict[str, float],
        criterion: nn.Module,
    ) -> float:
        """Run validation and return average loss."""
        model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_input_y, batch_input_u, batch_targets in val_loader:
                try:
                    outputs = model(batch_input_y.detach(), batch_input_u.detach())
                    """outputs = model.forward_sequence(
                        {"input_y": batch_input_y, "input_u": batch_input_u}
                    )"""
                    output_dict = {
                        "multiStep_decodeError": outputs[3],
                        "oneStepDecoderError": outputs[1],
                        "forwardError": outputs[4],
                        "functional_1": outputs[0],
                        "functional_2": outputs[1],
                    }
                    batch_loss = self.calculate_weighted_loss(
                        output_dict, loss_weights, criterion
                    )
                    val_loss += batch_loss.item()
                    num_batches += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise

        return val_loss / max(num_batches, 1)

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epoch: int,
        val_loss: float,
        filepath: Path,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved: {filepath}")

    def getModel(self):
        return self.model

    def evaluateNetwork(self, U_val, Y_val):
        train_stateVector, train_inputVector, train_outputVector = self.prepareDataset(
            U_val, Y_val
        )
        t = time.time()
        fitted_Y = self.model([train_stateVector, train_inputVector])
        elapsed = time.time() - t

        return fitted_Y, train_outputVector, elapsed

    def validateModel(self, plot: bool = True):
        fitted_Y, train_outputVector, _ = self.evaluateNetwork(self.U_val, self.Y_val)
        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(fitted_Y)
            plt.plot(train_outputVector)
            plt.show()
        return fitted_Y, train_outputVector, 0
