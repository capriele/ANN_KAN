#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from functools import partial
import scipy.integrate
import pandas as pd


class AUVDataset:
    def __init__(self):
        self.stateSize = 4
        self.inputSize = 4
        self.outputSize = 4

    def innerDynamic(self, xT, uT, para, intgralTermRef=0):
        # Ensure inputs are NumPy arrays
        xT = np.reshape(xT, (self.stateSize,))
        uT = np.reshape(uT, (self.inputSize,))
        para = np.reshape(para, (self.paraSize,)) * 0 + 1

        m = 500.0
        Jz = 300.0
        Xu = 6.106
        Xuu = 5.0
        Yv = 11.203
        Yvv = 10.114
        Nr = 210.0
        Nrr = 3.0
        l1x, l1y, alpha1 = -1.01, -0.353, 0.7853981633974483
        l2x, l2y, alpha2 = -1.01, 0.353, -0.7853981633974483
        l3x, l3y, alpha3 = 1.01, -0.353, -0.7853981633974483
        l4x, l4y, alpha4 = 1.01, 0.353, 0.7853981633974483
        h1, h2, h3, h4 = para[0], para[1], para[2], para[3]
        pG = 38

        def integrand(tempo, xT):
            F1_y = np.cos(alpha1) * uT[0] * pG
            F2_y = np.cos(alpha2) * uT[1] * pG
            F3_y = np.cos(alpha3) * uT[2] * pG
            F4_y = np.cos(alpha4) * uT[3] * pG

            F1_x = np.sin(alpha1) * uT[0] * pG
            F2_x = np.sin(alpha2) * uT[1] * pG
            F3_x = np.sin(alpha3) * uT[2] * pG
            F4_x = np.sin(alpha4) * uT[3] * pG

            x1dot = (
                1
                / m
                * (
                    -Xu * xT[0]
                    - Xuu * xT[0] ** 2
                    + m * xT[1] * xT[2]
                    + h1 * F1_x
                    + F2_x * h2
                    + F3_x * h3
                    + F4_x * h4
                )
            )
            x2dot = (
                1
                / m
                * (
                    -Yv * xT[1]
                    - Yvv * xT[1] ** 2
                    - m * xT[0] * xT[2]
                    + h1 * F1_y
                    + h2 * F2_y
                    + h3 * F3_y
                    + h4 * F4_y
                )
            )
            x3dot = (
                1
                / Jz
                * (
                    -Nr * xT[2]
                    - Nrr * xT[2] ** 2
                    + h1 * (-F1_x * l1y + F1_y * l1x)
                    + h2 * (-F2_x * l2y + F2_y * l2x)
                    + h3 * (-F3_x * l3y + F3_y * l3x)
                    + h4 * (-F4_x * l4y + F4_y * l4x)
                )
            )
            # x4dot = xT[2]
            return np.reshape(np.hstack((x1dot, x2dot, x3dot)), (self.stateSize,))

        Ts = 0.1
        # xN = np.zeros((self.stateSize, 1))
        # xN[0] = xT[0] + Ts * x1dot
        # xN[1] = xT[1] + Ts * x2dot
        # xN[2] = xT[2] + Ts * x3dot
        res = scipy.integrate.solve_ivp(integrand, [0, Ts], xT, method="BDF")
        xN = res.y[:, -1]
        # xN[3] = xT[3] + Ts * x4dot

        return xN.reshape((self.stateSize, 1))

    def stateMap(self, x, u, para):
        # Ensure inputs are NumPy arrays
        x = np.asarray(x)
        u = np.asarray(u)
        para = np.asarray(para)
        return self.innerDynamic(x, u, para).flatten()

    def outputMap(self, xk, u):
        # Ensure inputs are NumPy array
        return np.reshape(xk, (self.outputSize, 1))

    def loop(self, x_k, du_k):
        # Simulate system response to input sequence
        y_n = []
        x_k = np.reshape(np.asarray(x_k), (self.stateSize, 1))
        para = np.ones(self.paraSize)  # Placeholder
        for i in range(len(du_k)):
            u = np.reshape(np.asarray(du_k[i]), (self.inputSize,))
            u = np.multiply(u, self.stdU.reshape(-1, 1)) + self.meanU.reshape(-1, 1)
            y = self.outputMap(x_k, u)
            y = (y - self.meanY.reshape(-1, 1)) / self.stdY.reshape(-1, 1)
            y_n.append(y)
            x_k = self.stateMap(x_k, u, para)
        return np.array(y_n).squeeze(), x_k

    def prepareDataset(self, sizeT, sizeV):
        # Load data from CSV
        data = pd.read_csv(
            # "./sys-id-OpenMAUVe/results/Glider_Lib.Simulations.TestAUV_5d_CEGIS_LMI_monopile_v3/TestAUV_5d_CEGIS_LMI_monopile_res.csv"
            "./sys-id-OpenMAUVe/results/Glider_Lib.Simulations.TestAUV_5d_CEGIS_LMI_monopile_v3/TestAUV_5d_CEGIS_LMI_monopile_res_2.csv"
        )

        # Extract relevant columns
        u = data["generic_AUV_3d.out_lin_vel_u"].values  # state x1
        v = data["generic_AUV_3d.out_lin_vel_v"].values  # state x2
        r = data["generic_AUV_3d.out_ang_vel_r"].values  # state x3
        psi = data["generic_AUV_3d.out_angles_DCM[3]"].values  # state x4

        # Stack states to form x_k (if needed)
        x_k = np.column_stack((u, v, r, psi))  # shape: (num_samples, stateSize)

        # Extract inputs (u) and outputs (y) for training and validation
        # Assuming:
        # - y_n is the output (e.g., next state or target)
        # - u_n is the input (e.g., control input or current state)
        # Replace these with your actual logic
        y_n = x_k[:sizeT, :]  # Example: use all states except last as output
        u_n = x_k[:sizeT, :]  # Example: use all states except last as input
        y_Vn = x_k[-sizeV:, :]  # Last 'sizeV' samples for validation output
        u_Vn = x_k[-sizeV:, :]  # Last 'sizeV' samples for validation input

        print(y_n.shape)
        print(u_n.shape)

        # Normalize data
        self.meanY = np.mean(y_n, axis=0)
        self.meanU = np.mean(u_n, axis=0)
        self.stdY = np.std(y_n, axis=0)
        self.stdU = np.std(u_n, axis=0)

        # Avoid division by zero
        self.stdY[self.stdY == 0] = 1.0
        self.stdU[self.stdU == 0] = 1.0

        y_n = (y_n - self.meanY) / self.stdY
        y_Vn = (y_Vn - self.meanY) / self.stdY
        u_n = (u_n - self.meanU) / self.stdU
        u_Vn = (u_Vn - self.meanU) / self.stdU

        # Reshape for training/validation
        return (
            u_n.reshape((u_n.shape[0], self.inputSize)),
            y_n.reshape((y_n.shape[0], self.outputSize)),
            u_Vn.reshape((sizeV, self.inputSize)),
            y_Vn.reshape((sizeV, self.outputSize)),
        )
