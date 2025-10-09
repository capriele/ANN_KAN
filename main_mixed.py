import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import warnings
import scipy.io
import time
import sys
from scipy import optimize
from AdvAutoencoder import AdvAutoencoder, DatasetLoadUtility
from DynamicalSystem import LinearSystem
from TwoTanks import TwoTanks
from DummyModel import DummyModel
from multiprocessing import Process, freeze_support
from SpacecraftCW import SpacecraftNonlinear
from AUV import AUV
from AUVDataset import AUVDataset

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)
torch.use_deterministic_algorithms(True)

# Matplotlib settings
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 14
plt.rcParams["text.usetex"] = False

# torch.set_default_dtype(torch.float32)  # or torch.float16, torch.float32, etc.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# @unique
class SystemSelectorEnum:
    @staticmethod
    def load_from_dataset(filename, non_linear_input_char=False):
        # Placeholder for dynamic model loading
        dynamic_model = DummyModel()
        # Placeholder for dataset loading utility
        ds_loading = DatasetLoadUtility()
        u_vero, y_vero, uv, yv = ds_loading.loadDatasetFromMATfile(filename)
        numel = u_vero.shape[0]
        numel_v = uv.shape[0]
        u_n = np.reshape(u_vero.T[0], (numel, 1))
        y_n = np.reshape(y_vero.T[0], (numel, 1))
        u_vn = np.reshape(uv.T[0], (numel_v, 1))
        y_vn = np.reshape(yv.T[0], (numel_v, 1))

        mean_y = np.mean(y_n)
        mean_u = np.mean(u_n)
        std_y = np.std(y_n)
        std_u = np.std(u_n)

        y_n = (y_n - mean_y) / std_y
        y_vn = (y_vn - mean_y) / std_y
        u_n = (u_n - mean_u) / std_u
        u_vn = (u_vn - mean_u) / std_u

        return dynamic_model, u_n, y_n, u_vn, y_vn

    @staticmethod
    def MAGNETO_dataset():
        print("MAGNETO_dataset")
        return SystemSelectorEnum().load_from_dataset("datasets/Magneto.mat")

    @staticmethod
    def TANKS_dataset():
        print("TANKS_dataset")
        return SystemSelectorEnum().load_from_dataset("datasets/TwoTanksMatlab.mat")

    @staticmethod
    def SILVERBOX_dataset():
        print("SILVERBOX_dataset")
        return SystemSelectorEnum().load_from_dataset("datasets/Silverbox.mat")

    def TWOTANKS(self, non_linear_input_char=False):
        print("TWOTANKS")
        dynamic_model = TwoTanks(Option.nonLinearInputChar)
        u, y, u_val, y_val = dynamic_model.prepareDataset(20000, 1000)
        return dynamic_model, u, y, u_val, y_val

    def BILINEAR(self, non_linear_input_char=False):
        print("BILINEAR")
        dynamic_model = LinearSystem(Option.nonLinearInputChar)
        u, y, u_val, y_val = dynamic_model.prepareDataset(10000, 1000)
        return dynamic_model, u, y, u_val, y_val

    def SpacecraftNonlinearModel(self, non_linear_input_char=False):
        print("SpacecraftNonlinear")
        dynamic_model = SpacecraftNonlinear()
        u, y, u_val, y_val = dynamic_model.prepareDataset(20000, 1000)
        return dynamic_model, u, y, u_val, y_val

    def AUVNonlinearModel(self, non_linear_input_char=False):
        print("AUV")
        dynamic_model = AUV()
        u, y, u_val, y_val = dynamic_model.prepareDataset(25000, 1000)
        return dynamic_model, u, y, u_val, y_val

    def AUVDatasetNonlinear(self, non_linear_input_char=False):
        print("AUVDatasetNonlinear")
        dynamic_model = AUVDataset()
        u, y, u_val, y_val = dynamic_model.prepareDataset(25000, 1000)
        return dynamic_model, u, y, u_val, y_val


class Options:
    def __init__(self):
        self.nonLinearInputChar = True
        self.dynamicalSystemSelector = SystemSelectorEnum().AUVNonlinearModel
        self.stringDynamicalSystemSelector = "AUVNonlinearModel"
        self.affineStruct = True
        self.openLoopStartingPoint = 15
        self.horizon = 10
        self.TRsteps = 1
        self.fitHorizon = 7
        self.n_a = 15
        self.useGroupLasso = False
        self.stateReduction = True
        self.regularizerWeight = 0.0001
        self.closedLoopSim = True
        self.enablePlot = True
        self.stateSize = 6
        self.inputSize = 1
        self.outputSize = 1
        self.outputWindowLen = 2
        self.n_layers = 3
        self.n_neurons = 30
        self.epochs = 150
        self.batch_size = 24 * 2
        self.early_stopping_patience = 8
        self.min_delta = 0.001  # 0.0000001
        self.modelSelector = False
        self.modelKind = "ann"
        self.testName = "Test"
        self.alpha = 0


if __name__ == "__main__":
    freeze_support()
    Option = Options()

    # %% Parameter parsing
    print("Epochs", Option.epochs)
    print("Parameters", sys.argv)
    if len(sys.argv) > 2:
        print(f"Option.fitHorizon = {int(sys.argv[2])}")
        Option.fitHorizon = int(sys.argv[2])

    if len(sys.argv) > 3:
        print(f"Option.dynamicalSystemSelector = {int(sys.argv[3])}")
        if int(sys.argv[3]) == 1:
            Option.dynamicalSystemSelector = SystemSelectorEnum().TWOTANKS
            Option.stringDynamicalSystemSelector = "TWOTANKS"
        elif int(sys.argv[3]) == 2:
            # It's actually the hammerstein-wiener! But the old name stuck
            Option.dynamicalSystemSelector = SystemSelectorEnum().BILINEAR
            Option.stringDynamicalSystemSelector = "BILINEAR"
        elif int(sys.argv[3]) == 3:
            Option.dynamicalSystemSelector = SystemSelectorEnum.MAGNETO_dataset
            Option.stringDynamicalSystemSelector = "MAGNETO_dataset"
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 4:
            Option.dynamicalSystemSelector = SystemSelectorEnum.TANKS_dataset
            Option.stringDynamicalSystemSelector = "TANKS_dataset"
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 5:
            Option.dynamicalSystemSelector = SystemSelectorEnum.SILVERBOX_dataset
            Option.stringDynamicalSystemSelector = "SILVERBOX_dataset"
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 6:
            Option.dynamicalSystemSelector = (
                SystemSelectorEnum().SpacecraftNonlinearModel
            )
            Option.stringDynamicalSystemSelector = "SpacecraftNonlinearModel"
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 7:
            Option.dynamicalSystemSelector = SystemSelectorEnum().AUVNonlinearModel
            Option.stringDynamicalSystemSelector = "AUVNonlinearModel"
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 8:
            Option.dynamicalSystemSelector = SystemSelectorEnum().AUVDatasetNonlinear
            Option.stringDynamicalSystemSelector = "AUVdatasetNonlinear"
            Option.closedLoopSim = False

    if len(sys.argv) > 4:
        print(f"Option.nonLinearInputChar = {int(sys.argv[4])}")
        if int(sys.argv[4]) == 1:
            Option.nonLinearInputChar = True
        else:
            Option.nonLinearInputChar = False

    if len(sys.argv) > 5:
        print(f"Option.stateSize = {int(sys.argv[5])}")
        Option.stateSize = int(sys.argv[5])

    if len(sys.argv) > 6:
        print(f"Option.n_a = {int(sys.argv[6])}")
        Option.n_a = int(sys.argv[6])

    if len(sys.argv) > 7:
        print(f"Option.affineStruct = {int(sys.argv[7])}")
        if int(sys.argv[7]) == 1:
            Option.affineStruct = True
        else:
            Option.affineStruct = False

    if len(sys.argv) > 8:
        if int(sys.argv[8]) == 1:
            Option.affineStruct = False
            Option.useGroupLasso = True
            Option.stateReduction = True
            Option.regularizerWeight = 0.0003
        elif int(sys.argv[8]) == 2:
            Option.useGroupLasso = True
            Option.affineStruct = False
            Option.stateReduction = not True
            Option.regularizerWeight = 0.0003
        else:
            Option.useGroupLasso = False
            Option.regularizerWeight = 0.0001
            pass
        print(float(sys.argv[8]))

    # Check KAN mode flag
    if len(sys.argv) > 10:
        if int(sys.argv[10]) == 1:
            print("Enable KAN model")
            Option.modelSelector = 1
            Option.modelKind = "kan"
            Option.n_neurons = 8
            Option.epochs = 300
        elif int(sys.argv[10]) == 2:
            print("Enable Koopman model")
            Option.modelSelector = 2
            Option.modelKind = "koopman"
        elif int(sys.argv[10]) == 3:
            print("Enable KAN + Koopman model")
            Option.modelSelector = 3
            Option.modelKind = "kan_koopman"
            Option.n_neurons = 8
            Option.epochs = 300
        elif int(sys.argv[10]) == 4:
            print("Enable Mamba model")
            Option.modelSelector = 4
            Option.modelKind = "mamba"
        elif int(sys.argv[10]) == 5:
            print("Enable Mixed model")
            Option.modelSelector = 5
            Option.modelKind = "mixed"
        else:
            Option.modelKind = "ann"
            Option.modelSelector = False
        print(f"Option.modelKind = {Option.modelKind}")
        print(f"Option.modelSelector = {Option.modelSelector}")

    # Find model test name
    if len(sys.argv) > 11:
        Option.alpha = float(sys.argv[11])
        Option.testName = str(sys.argv[-1]) + "_" + str(sys.argv[11]).replace(".", "_")
        print(f"Option.alpha = {Option.alpha}")
        print(f"Option.testName = {Option.testName}")

    warnings.filterwarnings("ignore")

    # %% DS generation and model learning
    simulatedSystem, U_n, Y_n, U_Vn, Y_Vn = Option.dynamicalSystemSelector()
    if isinstance(simulatedSystem, DummyModel):
        simulatedSystem.stateSize = Option.stateSize

    Option.inputSize = simulatedSystem.inputSize
    Option.outputSize = simulatedSystem.outputSize

    model = AdvAutoencoder(
        affineStruct=Option.affineStruct,
        useGroupLasso=Option.useGroupLasso,
        stateReduction=Option.stateReduction,
        fitHorizon=Option.fitHorizon,
        strideLen=Option.n_a,  # n_a=n_b
        outputWindowLen=Option.outputWindowLen,  # +1 wrt the paper
        n_layer=Option.n_layers,
        n_neurons=Option.n_neurons,
        regularizerWeight=Option.regularizerWeight,
        stateSize=Option.stateSize,
        inputSize=Option.inputSize,
        outputSize=Option.outputSize,
        batch_size=Option.batch_size,
        modelSelector=Option.modelSelector,
    )
    model.setDataset(U_n.copy(), Y_n.copy(), U_Vn.copy(), Y_Vn.copy())

    inputU, inputY = model.prepareDataset()
    print(f"inputU shape: {inputU.shape}")
    print(inputU)
    print(f"inputY shape: {inputY.shape}")
    print(inputY)
    model.trainModel(
        epochs=Option.epochs,
        early_stopping_patience=Option.early_stopping_patience,
        min_delta=Option.min_delta,
        device=device,
        batchMode=True,
        alpha=Option.alpha,
        # batchMode=(Option.modelSelector == 4),  # Batch mode only for mamba
    )
    torch.save(
        model.model.state_dict(),
        f"results_mixed/{Option.modelKind}/{Option.testName}/model.pth",
    )
    # If you want load a previous model without training
    # model.model, _, _, _ = model.ANNModel()
    # model.model.load_state_dict(
    #     torch.load(
    #         f"results_mixed/{Option.modelKind}/{Option.testName}/model.pth",
    #         map_location=torch.device("cpu"),
    #         weights_only=False,
    #     ),
    #     # strict=False,
    # )
    (
        predictedLeft,
        stateLeft,
        oneStepAheadPredictionError,
        forwardedPredictedError,
        forwardError,
    ) = model.model(inputY, inputU)

    # %% Functions definition
    def prepareMatrices(uSequence, x0):
        logY = []
        logX = []
        uSequence = np.array(uSequence)

        for u in uSequence:
            u = np.reshape(u, (1, Option.inputSize))
            x0 = np.reshape(x0.detach().cpu().numpy(), (1, Option.stateSize))
            x0 = model.model.bridge_network(
                torch.tensor(u, dtype=torch.float32),
                torch.tensor(x0, dtype=torch.float32),
            )
            y = model.model.output_decoder(x0[0])
            logY += [y]
            logX += [x0]
            x0 = x0[0]
        return logX, logY

    def costFunction(uSequence, r, um1, logAB, logC, x0):
        logY = []
        uSequence = np.array(uSequence)
        um1 = np.array(um1)
        i = 0
        x0 = x0.detach().cpu().numpy()
        for u in uSequence:
            x0 = np.reshape(x0, (Option.stateSize))
            u = np.reshape(u, (Option.inputSize))
            asda = np.concatenate(
                [
                    x0,
                    u,
                ]
            )

            # check if logab is a torch tensor
            if isinstance(logAB[i], tuple):
                logAB[i] = list(logAB[i])
            if isinstance(logAB[i], torch.Tensor):
                logAB[i] = logAB[i].detach().cpu().numpy()
            if isinstance(logAB[i][1], torch.Tensor):
                logAB[i][1] = logAB[i][1].detach().cpu().numpy()
            if isinstance(logAB[i][2], torch.Tensor):
                logAB[i][2] = logAB[i][2].detach().cpu().numpy()
            if isinstance(logC[i], tuple):
                logC[i] = list(logC[i])
            if isinstance(logC[i][0], torch.Tensor):
                logC[i][0] = logC[i][0].detach().cpu().numpy()
            if isinstance(logC[i][1], torch.Tensor):
                logC[i][1] = logC[i][1].detach().cpu().numpy()

            # Fixed: Proper reshaping to match original dimensions
            asda = np.reshape(asda, (Option.stateSize + Option.inputSize, 1))

            # State update using logAB[i][1] (A matrix) and logAB[i][2] (bias)
            x0 = np.dot(logAB[i][1], asda)
            x0 = x0 + np.reshape(logAB[i][2], (Option.stateSize, 1))

            # Output computation using logC[i][1] (C matrix)
            y = np.dot(logC[i][0], x0)
            logY += [y[0][-1]]

            # Update x0 for next iteration - convert back to proper shape
            x0 = x0.squeeze()

            i = i + 1
        #    logY+=[y[0][1]]
        logY = np.array(logY)
        #    print(logY-r)
        cost = (
            0.04 * np.sum(np.square(uSequence))
            + 0.1 * np.sum(np.square(uSequence[1:] - uSequence[:-1]))
            + 0.1 * np.sum(np.square(uSequence[0] - um1))
            + np.sum(np.square(logY - r)) * 5
        )
        return cost

    def evaluateFeatureImportance():
        from matplotlib.ticker import MaxNLocator

        if not Option.stateReduction:
            w = model.model.conv_encoder.get_layer("enc00").get_weights()
            ax = plt.figure(figsize=[8, 2]).gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            neuronsCount = np.sum(abs(w[0]) > 1e-3, 1)
            #        print(len(neuronsCount))
            windowsLen = int(len(neuronsCount) / 2)
            yAxis = range(0, windowsLen)[::-1]
            print(neuronsCount, "encoder=>")
            plt.title("$encoder$")
            plt.step(yAxis, neuronsCount[0:windowsLen], where="mid")
            plt.step(yAxis, neuronsCount[windowsLen:], where="mid")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()

        else:
            w1 = model.model.bridge_network.get_layer("bridge00").get_weights()
            w = model.model.output_decoder.get_layer("dec00").get_weights()
            neuronsCount = np.sum(abs(w1[0][0:-1]) > 1e-3, 1)
            yAxis = range(0, len(neuronsCount))
            print(neuronsCount, "bridge=>")
            plt.figure(figsize=[8, 2])
            plt.title("$bridge$")
            plt.step(yAxis, neuronsCount, where="mid")
            plt.tight_layout()
            neuronsCount = np.sum(abs(w[0]) > 1e-3, 1)
            print(neuronsCount, "decoder=>")
            yAxis = range(0, len(neuronsCount))
            ax = plt.figure(figsize=[8, 2]).gca()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.title("$decoder$")
            plt.step(yAxis, neuronsCount, where="mid")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
        pass

    def openLoopValidation(
        validationOnMultiHarmonic=True, _reset=-1, YTrue=None, U_Vn=None
    ):
        openLoopStartingPoint = Option.openLoopStartingPoint
        pastY = torch.zeros((model.strideLen, Option.outputSize)).to(device)
        pastU = torch.zeros((model.strideLen, Option.inputSize)).to(device)
        if YTrue is None:
            x0RealSystem = np.zeros((simulatedSystem.stateSize,))

        logY = []
        logU = []
        logYR = []
        finalRange = 1000
        if not (YTrue is None):
            finalRange = YTrue.shape[0]
        for i in range(0, finalRange):
            x0 = model.model.conv_encoder(
                pastY.reshape(1, -1),
                pastU.reshape(1, -1),
            )
            # Default construction of u as a vector with shape (Option.inputSize, 1)
            u_scalar = 0.5 * np.sin(i / (20 + 0.01 * i)) + 0.5
            # Create a (Option.inputSize, 1) vector with the same value in all positions
            u = np.full((1, Option.inputSize), u_scalar)

            if not validationOnMultiHarmonic:
                u = np.reshape(U_Vn[i], (1, Option.inputSize))  # Ensure u is reshaped

            if YTrue is None:
                y_kReal, x0RealSystem_ = simulatedSystem.loop(x0RealSystem, u)
                x0RealSystem = np.reshape(x0RealSystem_, (simulatedSystem.stateSize,))
            else:
                y_kReal = YTrue[i]
                u = np.reshape(U_Vn[i], (1, Option.inputSize))  # Ensure u is reshaped

            pastU = torch.cat(
                (
                    pastU.to(device),
                    torch.tensor(u, dtype=torch.float32)
                    .reshape(1, Option.inputSize)
                    .to(device),
                ),
                dim=0,
            )[1:]
            pastY = torch.cat(
                (
                    pastY.to(device),
                    torch.tensor(y_kReal, dtype=torch.float32)
                    .reshape(1, Option.outputSize)
                    .to(device),
                ),
                dim=0,
            )[1:]
            if i < openLoopStartingPoint or (i % _reset == 0 and _reset > 0):
                x0 = model.model.conv_encoder(
                    torch.tensor(pastY, dtype=torch.float32).reshape(1, -1).to(device),
                    torch.tensor(pastU, dtype=torch.float32).reshape(1, -1).to(device),
                )
                print("*", end="")
            else:
                _u = torch.tensor(u, dtype=torch.float32).reshape(1, Option.inputSize)
                _x0 = torch.tensor(x0, dtype=torch.float32).reshape(1, Option.stateSize)
                # print(_u.shape)
                # print(_x0.shape)
                x0 = model.model.bridge_network(
                    _u.to(device),
                    _x0.to(device),
                )[0]

            # print(x0.shape)

            y = model.model.output_decoder(x0)[1]
            if i >= openLoopStartingPoint:
                logY += [
                    np.reshape(
                        (y[0][-2]).detach().cpu().numpy(), (1, Option.outputSize)
                    )
                ]
                logYR += [np.reshape(y_kReal, (1, Option.outputSize))]
                logU += [u]
            print(".", end="")
        print("\n")
        logY = np.array(logY[:-1])
        logYR = np.array(logYR[:-1])

        # Creazione della maschera per i valori non NaN
        # non_nan_mask = ~np.isnan(logY) & ~np.isnan(logYR)

        # Applicazione della maschera per rimuovere i NaN
        # logY = logY[non_nan_mask]
        # logYR = logYR[non_nan_mask]

        # logYR = logYR.reshape(logYR.shape[0], 1)
        a = np.linalg.norm(np.array(logY) - np.array(logYR))
        b = np.linalg.norm(np.mean(np.array(logY)) - np.array(logYR))
        if b == 0:
            b = 1
        fit = 1 - (a / b)
        NRMSE = 1 - np.sqrt(np.mean(np.square(np.array(logY) - np.array(logYR)))) / (
            np.max(logYR) - np.min(logYR)
        )
        fit = np.max([0, fit])
        NRMSE = np.max([0, NRMSE])
        print("fit: ", fit)
        print("NRMSE: ", NRMSE)
        if Option.enablePlot:
            logY = np.reshape(logY, (logY.shape[0], Option.outputSize, 1))
            logYR = np.reshape(logYR, (logYR.shape[0], Option.outputSize, 1))
            for i in range(Option.outputSize):
                plt.figure()
                plt.title(
                    f"Open loop simulation for component {i+1} (k={openLoopStartingPoint}, fit={fit})"
                )

                # Plot logY and logYR for the i-th component
                (y,) = plt.plot(logY[:, i], label="$\hat{y}_{i}$")
                (yr,) = plt.plot(logYR[:, i], label="$y_{i}$")
                (et,) = plt.plot(logY[:, i] - logYR[:, i], label="$e_{i}$")

                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    f"open_loop_simulation_component_{i+1}_{validationOnMultiHarmonic}_{_reset}.png"
                )
                plt.close()  # Close the figure to free memory
        return fit, NRMSE, logY, logYR

    # %% Model Validation Validation
    validationOnMultiHarmonic = [True, False]
    reset = [1, 10, -1]
    for r in reset:
        for voM in validationOnMultiHarmonic:
            start = time.time()
            YtrueToPass = None
            if "dataset" in Option.stringDynamicalSystemSelector:
                YtrueToPass = Y_Vn.copy()
            fit, NRMSE, logY, logYR = openLoopValidation(
                validationOnMultiHarmonic=voM,
                _reset=r,
                YTrue=YtrueToPass,
                U_Vn=U_Vn.copy(),
            )
            end = time.time()
            print("elapsed time in simulation:", end - start)
            print("validationOnMultiHarmonic:", voM, end=" ")
            print("reset every:", r, end=" ")
            print("fit: ", fit, " NRMSE: ", NRMSE)

    # %% Closed loop Simulation with MPC
    u = [U_Vn[0]]

    if Option.closedLoopSim and Option.affineStruct:
        print("Closed Loop")
        NUM_ITERATIONS = 400
        REF_AMPLITUDE = 0.7
        REF_PERIOD = 20
        REF_DECAY = 0.01
        logY, logU, logYR = [], [], []
        MPCHorizon = Option.horizon
        pastY = torch.zeros((model.strideLen, Option.outputSize)).to(device)
        pastU = torch.zeros((model.strideLen, Option.inputSize)).to(device)
        x0RealSystem = np.zeros((simulatedSystem.stateSize,))
        x0 = model.model.conv_encoder(
            pastY.reshape(1, -1),
            pastU.reshape(1, -1),
        )
        bounds = [(-1.5, 1.5) for _ in range(MPCHorizon * 1 * Option.inputSize)]
        pastRes = np.zeros((MPCHorizon, 1, Option.inputSize))
        u = np.zeros((1, Option.inputSize))
        start = time.time()

        for i in range(NUM_ITERATIONS):
            x0 = model.model.conv_encoder(
                pastY.reshape(1, -1),
                pastU.reshape(1, -1),
            )
            r = [
                REF_AMPLITUDE * np.array([[np.sin(j / (REF_PERIOD + REF_DECAY * j))]])
                + REF_AMPLITUDE
                for j in range(i, i + MPCHorizon)
            ]
            logY.append(r[0][0])

            for _ in range(Option.TRsteps):
                logAB, logC = prepareMatrices(pastRes, x0)

                def lambdaCostFunction(x):
                    x = x.reshape(MPCHorizon, 1, Option.inputSize)
                    return costFunction(x, r, u, logAB, logC, x0.T)

                result = optimize.minimize(
                    lambdaCostFunction,
                    pastRes.reshape(-1),
                    bounds=bounds,
                    method="SLSQP",
                )
                result = result.x.reshape(MPCHorizon, 1, Option.inputSize)
                u = np.array(result[0]).reshape((1, Option.inputSize))
                pastRes = result

            pastRes[:-1] = pastRes[1:]
            y_kReal, x0RealSystem = simulatedSystem.loop(x0RealSystem, u)
            x0RealSystem = x0RealSystem.copy()
            pastU = torch.cat(
                (
                    pastU.to(device),
                    torch.tensor(u, dtype=torch.float32)
                    .reshape(1, Option.inputSize)
                    .to(device),
                ),
                dim=0,
            )[1:]
            pastY = torch.cat(
                (
                    pastY.to(device),
                    torch.tensor(y_kReal, dtype=torch.float32)
                    .reshape(1, Option.outputSize)
                    .to(device),
                ),
                dim=0,
            )[1:]
            logYR.append(y_kReal.flatten())
            logU.append(u.flatten())

        end = time.time()
        print("\nElapsed time in MPC:", end - start)

        logY = np.array(logY)
        logYR = np.array(logYR)
        logU = np.array(logU)

        if Option.enablePlot:
            le = []
            lv = []
            plt.figure()
            plt.title("Closed loop simulation")
            plt.tight_layout()
            plt.grid()

            for i in range(logU.shape[1]):
                if logU.ndim == 1:
                    (uk,) = plt.plot(logU[i])
                else:
                    (uk,) = plt.plot(logU[:, i])
                le.append(uk)
                lv.append("$u_{k}$")

            for i in range(logYR.shape[1]):
                if logYR.ndim == 1:
                    (yk,) = plt.plot(logYR[i])
                else:
                    (yk,) = plt.plot(logYR[:, i])
                le.append(yk)
                lv.append("$y_{k}$")

            if logYR.ndim == 1:
                (rk,) = plt.plot(logY[i])
            else:
                (rk,) = plt.plot(logY[:, i])
            le.append(rk)
            lv.append("$r_{k}$")
            plt.legend(le, lv)
            plt.savefig("closed_loop_simulation.png")
    # print(fit)
    # %% Feature Importance
    if Option.useGroupLasso:
        if Option.affineStruct:
            print("******WARNING: affine struct is enabled******")
        print("evaluating state importance=>" + str(Option.stateReduction))
        # evaluateFeatureImportance()

    # %% These functions are used to generate plots for the paper
    def prettyPrintStatsUseNA(aOutput, aInput):
        aOutput = np.array(aOutput)
        aInput = np.array(aInput)
        xAxis = range(0, aInput.shape[1])[::-1]
        from matplotlib.ticker import MaxNLocator

        ax = plt.figure(figsize=[8, 2]).gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Notice: they are inverted with respect to the output of the feature importance function
        (lineOutput,) = plt.step(xAxis, aOutput.T, where="mid")
        (lineInput,) = plt.step(xAxis, aInput.T, where="mid")
        plt.legend([lineOutput, lineInput], ["$\\{y_k\\}$", "$\\{u_k\\}$"])
        plt.grid()
        plt.xlabel("time-step~delay")
        plt.tight_layout()

    def prettyPrintStatsUseNX(ADecoder, aBridge):
        aBridge = np.array(aBridge)
        ADecoder = np.array(ADecoder)
        xAxis = range(1, ADecoder.shape[1] + 1)[::-1]
        from matplotlib.ticker import MaxNLocator

        ax = plt.figure(figsize=[8, 2]).gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        (lineBridge,) = plt.step(xAxis, aBridge.T, where="mid")
        (lineDecoder,) = plt.step(xAxis, ADecoder.T, where="mid")
        plt.legend([lineBridge, lineDecoder], ["$bridge$", "$decoder$"])
        plt.grid()
        plt.xlabel("state~component")
        plt.tight_layout()

    print(Option.__dict__)
    scipy.io.matlab.savemat(
        "dumps/dump_{0}.mat".format(Option.testName),
        {
            "U": U_n,
            "Y": Y_n,
            "U_val": U_Vn,
            "Y_val": Y_Vn,
            "Option": str(Option.__dict__),
        },
    )
