#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from functools import partial


class AUV:
    def __init__(self):
        self.stateSize = 5
        self.inputSize = 4
        self.outputSize = 5
        self.paraSize = 4

        # Initialize NumPy arrays for state-space matrices
        self.A = np.zeros((self.stateSize, self.stateSize))
        self.B = np.zeros((self.stateSize, self.inputSize))
        self.C = np.eye(self.outputSize, self.stateSize)  # Identity for simplicity
        self.D = np.zeros((self.outputSize, self.inputSize))

    def innerDynamic(self, xT, uT, para, intgralTermRef=0):
        # Ensure inputs are NumPy arrays
        xT = np.reshape(xT, (self.stateSize, 1))
        uT = np.reshape(uT, (self.inputSize,))
        para = np.reshape(para, (self.paraSize,))

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
        x4dot = xT[2]
        x5dot = xT[3] - np.asarray(intgralTermRef).ravel()[0]

        Ts = 0.01
        xN = np.zeros((self.stateSize, 1))
        xN[0] = xT[0] + Ts * x1dot
        xN[1] = xT[1] + Ts * x2dot
        xN[2] = xT[2] + Ts * x3dot
        xN[3] = xT[3] + Ts * x4dot
        xN[4] = xT[4] + Ts * x5dot

        return xN.reshape((self.stateSize, 1))

    def stateMap(self, x, u, para):
        # Ensure inputs are NumPy arrays
        x = np.asarray(x)
        u = np.asarray(u)
        para = np.asarray(para)
        return self.innerDynamic(x, u, para).flatten()

    def outputMap(self, xk, u):
        # Ensure inputs are NumPy arrays
        xk = np.asarray(xk)
        u = np.asarray(u)
        return self.C @ xk

    def systemDynamics(self, dim, flag=True):
        # Simulate system dynamics for `dim` steps
        x_k = np.ones((self.stateSize, 1))
        y_n = np.zeros((dim, self.outputSize))
        u_n = np.random.normal(1, 1.0, size=(dim, self.inputSize))
        para = np.ones(self.paraSize)  # Placeholder

        for i in range(dim):
            u = u_n[i : i + 1, :].T
            x_k = self.stateMap(x_k, u, para)
            y_n[i] = self.outputMap(x_k, u).flatten()

        return y_n, u_n

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
        # Generate training and validation datasets
        y_n, u_n = self.systemDynamics(sizeT, True)
        y_Vn, u_Vn = self.systemDynamics(sizeV, True)
        self.meanY = np.mean(y_n)
        self.meanU = np.mean(u_n)
        self.stdY = np.std(y_n)
        self.stdU = np.std(u_n)
        y_n = (y_n - self.meanY) / self.stdY
        y_Vn = (y_Vn - self.meanY) / self.stdY
        u_n = (u_n - self.meanU) / self.stdU
        u_Vn = (u_Vn - self.meanU) / self.stdU
        return (
            u_n.reshape((sizeT, self.inputSize)),
            y_n.reshape((sizeT, self.outputSize)),
            u_Vn.reshape((sizeV, self.inputSize)),
            y_Vn.reshape((sizeV, self.outputSize)),
        )
