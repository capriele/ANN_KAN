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

# Set random seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# Matplotlib settings
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 14
plt.rcParams["text.usetex"] = False


# @unique
class SystemSelectorEnum:
    @staticmethod
    def load_from_dataset(filename, non_linear_input_char=False):
        # Placeholder for dynamic model loading
        dynamic_model = DummyModel()
        # Placeholder for dataset loading utility
        ds_loading = DatasetLoadUtility()
        u_vero, y_vero, uv, yv = ds_loading.load_dataset_from_mat_file(filename)
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
        return SystemSelectorEnum().load_from_dataset("Magneto.mat")

    @staticmethod
    def TANKS_dataset():
        return SystemSelectorEnum().load_from_dataset("TwoTanksMatlab.mat")

    @staticmethod
    def SILVERBOX_dataset():
        return SystemSelectorEnum().load_from_dataset("Silverbox.mat")

    def TWOTANKS(self, non_linear_input_char=False):
        dynamic_model = TwoTanks(Option.nonLinearInputChar)
        u, y, u_val, y_val = dynamic_model.prepareDataset(20000, 1000)
        return dynamic_model, u, y, u_val, y_val

    def BILINEAR(self, non_linear_input_char=False):
        dynamic_model = LinearSystem(Option.nonLinearInputChar)
        u, y, u_val, y_val = dynamic_model.prepareDataset(10000, 1000)
        return dynamic_model, u, y, u_val, y_val


class Options:
    def __init__(self):
        self.nonLinearInputChar = True
        self.dynamicalSystemSelector = SystemSelectorEnum().TWOTANKS
        self.stringDynamicalSystemSelector = (
            str(self.dynamicalSystemSelector)
            .replace("<function SystemSelectorEnum.", "")
            .split(" at ")[0]
        )
        self.affineStruct = True
        self.openLoopStartingPoint = 15
        self.horizon = 5
        self.TRsteps = 1
        self.fitHorizon = 5
        self.n_a = 10
        self.useGroupLasso = False
        self.stateReduction = True
        self.regularizerWeight = 0.0001
        self.closedLoopSim = True
        self.enablePlot = False
        self.stateSize = 6
        self.outputWindowLen = 2
        self.n_layers = 3
        self.n_neurons = 30
        self.epochs = 150


if __name__ == "__main__":
    freeze_support()
    Option = Options()

    # %% Parameter parsing
    print("Epochs", Option.epochs)
    print("Parameters", sys.argv)
    sys.argv = ["rep_package/v2/main.py", "1", "5", "1", "1", "6", "10", "1", "0"]
    if len(sys.argv) > 2:
        Option.fitHorizon = int(sys.argv[2])
        print(int(sys.argv[2]))

    if len(sys.argv) > 3:
        if int(sys.argv[3]) == 1:
            Option.dynamicalSystemSelector = SystemSelectorEnum().TWOTANKS
        elif int(sys.argv[3]) == 2:
            # It's actually the hammerstein-wiener! But the old name stuck
            Option.dynamicalSystemSelector = SystemSelectorEnum().BILINEAR
        elif int(sys.argv[3]) == 3:
            Option.dynamicalSystemSelector = SystemSelectorEnum.MAGNETO_dataset
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 4:
            Option.dynamicalSystemSelector = SystemSelectorEnum.TANKS_dataset
            Option.closedLoopSim = False
        elif int(sys.argv[3]) == 5:
            Option.dynamicalSystemSelector = SystemSelectorEnum.SILVERBOX_dataset
            Option.closedLoopSim = False

        Option.stringDynamicalSystemSelector = (
            str(Option.dynamicalSystemSelector)
            .replace("<bound method SystemSelectorEnum.", "")
            .split(" of ")[0]
        )
        print(int(sys.argv[3]))

    if len(sys.argv) > 4:
        if int(sys.argv[4]) == 1:
            Option.nonLinearInputChar = True
        else:
            Option.nonLinearInputChar = False
        print(int(sys.argv[4]))

    if len(sys.argv) > 5:
        Option.stateSize = int(sys.argv[5])
        print(int(sys.argv[5]))

    if len(sys.argv) > 6:
        Option.n_a = int(sys.argv[6])
        print(float(sys.argv[6]))

    if len(sys.argv) > 7:
        if int(sys.argv[7]) == 1:
            Option.affineStruct = True
        else:
            Option.affineStruct = False
        print(float(sys.argv[7]))

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

    warnings.filterwarnings("ignore")

    # %% DS generation and model learning
    simulatedSystem, U_n, Y_n, U_Vn, Y_Vn = Option.dynamicalSystemSelector()

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
    )
    model.setDataset(U_n.copy(), Y_n.copy(), U_Vn.copy(), Y_Vn.copy())

    inputU, inputY = model.prepareDataset()
    model.trainModel(epochs=Option.epochs)
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
            u = np.reshape(u, (1, 1))
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
        for u in uSequence:
            # u=np.reshape(u,(1,1))
            x0 = torch.tensor(x0, dtype=torch.float32)
            utensor = torch.tensor([[u]], dtype=torch.float32)
            asda = (
                torch.cat(
                    [
                        x0.T,
                        utensor,
                    ]
                )
                .detach()
                .numpy()
            )

            # check if logab is a torch tensor
            if isinstance(logAB[i], tuple):
                logAB[i] = list(logAB[i])
            if isinstance(logAB[i], torch.Tensor):
                logAB[i] = logAB[i].detach().numpy()
            if isinstance(logAB[i][1], torch.Tensor):
                logAB[i][1] = logAB[i][1].detach().numpy()
            if isinstance(logAB[i][2], torch.Tensor):
                logAB[i][2] = logAB[i][2].detach().numpy()
            if isinstance(logC[i], tuple):
                logC[i] = list(logC[i])
            if isinstance(logC[i][0], torch.Tensor):
                logC[i][0] = logC[i][0].detach().numpy()
            if isinstance(logC[i][1], torch.Tensor):
                logC[i][1] = logC[i][1].detach().numpy()

            asda = np.reshape(asda, (1 + Option.stateSize, 1))
            x0 = np.dot(logAB[i][1], asda).T
            x0 = x0.squeeze()
            x0 = np.reshape(x0, (1, Option.stateSize))

            # TODO: reshape a 2 x 6 ((out size) x (state size))
            y = np.dot(logC[i][0].squeeze(), x0.T)
            logY += [y[0][-1]]
            i = i + 1
        #    logY+=[y[0][1]]
        logY = np.array(logY)
        #    print(logY-r)
        cost = (
            0.001 * np.sum(np.square(uSequence))
            + 0.01 * np.sum(np.square(uSequence[1:] - uSequence[:-1]))
            + 0.01 * np.sum(np.square(uSequence[0] - um1))
            + np.sum(np.square(logY - r)) * 1
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
        pastY = np.zeros((model.strideLen, 1))
        pastU = np.zeros((model.strideLen, 1))
        if YTrue is None:
            x0RealSystem = np.zeros((simulatedSystem.stateSize,))

        x0 = model.model.conv_encoder(
            torch.tensor(pastY, dtype=torch.float32).T,
            torch.tensor(pastU, dtype=torch.float32).T,
        )
        logY = []
        logU = []
        logYR = []
        finalRange = 1000
        if not (YTrue is None):
            finalRange = YTrue.shape[0]
        for i in range(0, finalRange):
            u = 0.5 * np.array([[np.sin(i / (20 + 0.01 * i))]]) + 0.5
            if not validationOnMultiHarmonic:
                u = [U_Vn[i]]
            if YTrue is None:
                y_kReal, x0RealSystem_ = simulatedSystem.loop(x0RealSystem, u)
                x0RealSystem = np.reshape(x0RealSystem_, (simulatedSystem.stateSize,))
            else:
                y_kReal = YTrue[i]
                u = [U_Vn[i]]

            pastU = np.reshape(np.append(pastU, u)[1:], (model.strideLen, 1))
            pastY = np.reshape(np.append(pastY, y_kReal)[1:], (model.strideLen, 1))
            if i < openLoopStartingPoint or (i % _reset == 0 and _reset > 0):
                x0 = model.model.conv_encoder(
                    torch.tensor(pastY, dtype=torch.float32).T,
                    torch.tensor(pastU, dtype=torch.float32).T,
                )
                print("*", end="")
            else:
                x0 = model.model.bridge_network(
                    torch.tensor(u, dtype=torch.float32),
                    torch.tensor(x0, dtype=torch.float32),
                )[0]
            y = model.model.output_decoder(x0)[1]
            if i >= openLoopStartingPoint:
                logY += [(y[0][-2]).detach().numpy()]
                logYR += [y_kReal[0]]
                logU += [u[0]]
            print(".", end="")
        print("\n")
        logY = np.array(logY)
        logYR = np.array(logYR)
        # logYR = logYR.reshape(logYR.shape[0], 1)
        print(logY)
        print("########")
        print(logYR)
        a = np.linalg.norm(np.array(logY) - np.array(logYR))
        b = np.linalg.norm(np.mean(np.array(logY)) - np.array(logYR))
        fit = 1 - (a / b)
        NRMSE = 1 - np.sqrt(np.mean(np.square(np.array(logY) - np.array(logYR)))) / (
            np.max(logYR) - np.min(logYR)
        )
        fit = np.max([0, fit])
        NRMSE = np.max([0, NRMSE])
        print("fit: ", fit)
        print("NRMSE: ", NRMSE)
        if Option.enablePlot:
            plt.figure()
            plt.title(
                "open loop simulation from k="
                + str(openLoopStartingPoint)
                + " fit="
                + str(fit)
            )
            (y,) = plt.plot(logY)
            (yr,) = plt.plot(logYR)
            (et,) = plt.plot(np.array(logY) - np.array(logYR))
            plt.tight_layout()
            plt.legend([y, yr, et], ["$\hat y$", "$y_{real}$", "estimation error"])
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
        logY = []
        logU = []
        logYR = []
        MPCHorizon = Option.horizon
        pastY = np.zeros((model.strideLen, 1))
        pastU = np.zeros((model.strideLen, 1))
        x0RealSystem = np.zeros((simulatedSystem.stateSize,))
        x0 = model.model.conv_encoder(
            torch.tensor(pastY, dtype=torch.float32).T,
            torch.tensor(pastU, dtype=torch.float32).T,
        )
        bounds = [(-0.8, 0.8) for i in range(0, MPCHorizon)]
        #    bounds=[(-1,1) for i in range(0,MPCHorizon)]
        pastRes = np.ones((MPCHorizon)) * 0
        start = time.time()
        logY += [0]
        for i in range(0, 400):
            x0 = model.model.conv_encoder(
                torch.tensor(pastY, dtype=torch.float32).T,
                torch.tensor(pastU, dtype=torch.float32).T,
            )
            r = [
                0.7 * np.array([[np.sin(j / (20 + 0.01 * j))]]) + 0.7
                for j in range(i, i + MPCHorizon)
            ]
            #        if i>200:
            #            r=1+r*0
            #        else:
            #            r=-1+r*0
            #        r=np.array([[.5+1.5*np.sin(i/(50+i/100))]])
            #    r=0.5*np.array([[np.sin(i/(20+0.01*i))]])+1
            #    r=np.array([[1.5+np.sin(i/(50+i/100))]])
            logY += [r[0][0]]
            for _ in range(0, Option.TRsteps):
                logAB, logC = prepareMatrices(pastRes, x0)

                def lamdaCostFunction(x):
                    return costFunction(x, r, u[0][0], logAB, logC, x0)

                result = optimize.minimize(lamdaCostFunction, pastRes, bounds=bounds)
                u = np.array(result.x[0]).reshape((1, 1))
                pastRes = result.x
            pastRes[0:-1] = pastRes[1:]
            # pastRes[-1]=0
            y_kReal, x0RealSystem = simulatedSystem.loop(x0RealSystem, u)
            x0RealSystem = x0RealSystem.copy()
            pastU = np.reshape(np.append(pastU, u)[1:], (model.strideLen, 1))
            pastY = np.reshape(np.append(pastY, y_kReal)[1:], (model.strideLen, 1))
            logYR += [y_kReal[0]]
            logU += [u[0]]
            print(".", end="")

        end = time.time()
        print("\n")
        if Option.enablePlot:
            plt.figure()
            plt.title("Closed loop simulaton")
            (uP,) = plt.plot(logU)
            plt.grid()
            (yP,) = plt.plot(logYR)
            (rP,) = plt.plot(logY)
            plt.tight_layout()
            plt.legend([uP, yP, rP], ["$u_k$", "$y_k$", "$r_k$"])
        print("elapsed time in MPC:", end - start)
    # print(fit)
    # %% Feature Importance
    if Option.useGroupLasso:
        if Option.affineStruct:
            print("******WARNING: affine struct is enabled******")
        print("evaluating state importance=>" + str(Option.stateReduction))
        evaluateFeatureImportance()

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
        "dump_{0}_{1}.mat".format(
            Option.stringDynamicalSystemSelector, Option.nonLinearInputChar
        ),
        {
            "U": U_n,
            "Y": Y_n,
            "U_val": U_Vn,
            "Y_val": Y_Vn,
            "Option": str(Option.__dict__),
        },
    )
