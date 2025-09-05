#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sys = drss(6,6,3)

sys =

  A =
              x1         x2         x3         x4         x5         x6
   x1    -0.2915     -1.399      1.989    -0.4098     -3.603     0.7189
   x2   -0.06878     0.3576     0.7603  6.834e-05    -0.8252     0.4593
   x3    -0.1776     0.1864     -1.732    0.09156      1.407    -0.3699
   x4    -0.3453      -3.37     -1.291   0.003601     -2.168       1.22
   x5    0.08058     0.5327    -0.9208   -0.04469      1.819    -0.7999
   x6    -0.4925      2.136    -0.3314      0.172     0.9935    -0.7792

  B =
            u1       u2       u3
   x1   -1.089   -1.492  -0.1924
   x2        0  -0.7423   0.8886
   x3   0.5525   -1.062  -0.7648
   x4    1.101     2.35   -1.402
   x5    1.544  -0.6156   -1.422
   x6        0   0.7481        0

  C =
             x1        x2        x3        x4        x5        x6
   y1         0   -0.8045    -1.148  -0.08249    0.1001     1.712
   y2   -0.1961         0         0         0   -0.5445   -0.1941
   y3     1.419    0.8351         0    -0.439    0.3035    -2.138
   y4    0.2916   -0.2437     2.585    -1.795   -0.6003   -0.8396
   y5         0    0.2157   -0.6669    0.8404      0.49     1.355
   y6     1.588    -1.166    0.1873    -0.888    0.7394         0

  D =
            u1       u2       u3
   y1    0.961    2.908    1.098
   y2        0        0        0
   y3        0        0        0
   y4   -1.961        0   -2.052
   y5  -0.1977        0        0
   y6   -1.208        0        0

"""

import numpy as np


class SpacecraftNonlinear:

    def __init__(self):
        # State-space matrices
        self.A = np.array(
            [
                [-0.2915, -1.399, 1.989, -0.4098, -3.603, 0.7189],
                [-0.06878, 0.3576, 0.7603, 6.834e-05, -0.8252, 0.4593],
                [-0.1776, 0.1864, -1.732, 0.09156, 1.407, -0.3699],
                [-0.3453, -3.37, -1.291, 0.003601, -2.168, 1.22],
                [0.08058, 0.5327, -0.9208, -0.04469, 1.819, -0.7999],
                [-0.4925, 2.136, -0.3314, 0.172, 0.9935, -0.7792],
            ]
        )

        self.B = np.array(
            [
                [-1.089, -1.492, -0.1924],
                [0, -0.7423, 0.8886],
                [0.5525, -1.062, -0.7648],
                [1.101, 2.35, -1.402],
                [1.544, -0.6156, -1.422],
                [0, 0.7481, 0],
            ]
        )

        self.C = np.array(
            [
                [0, -0.8045, -1.148, -0.08249, 0.1001, 1.712],
                [-0.1961, 0, 0, 0, -0.5445, -0.1941],
                [1.419, 0.8351, 0, -0.439, 0.3035, -2.138],
                [0.2916, -0.2437, 2.585, -1.795, -0.6003, -0.8396],
                [0, 0.2157, -0.6669, 0.8404, 0.49, 1.355],
                [1.588, -1.166, 0.1873, -0.888, 0.7394, 0],
            ]
        )

        self.D = np.array(
            [
                [0.961, 2.908, 1.098],
                [0, 0, 0],
                [0, 0, 0],
                [-1.961, 0, -2.052],
                [-0.1977, 0, 0],
                [-1.208, 0, 0],
            ]
        )

        self.stateSize = 6
        self.inputSize = 3
        self.outputSize = 6

    def stateMap(self, x, u):
        # Linear state update: x_{k+1} = A x_k + B u_k
        return (self.A @ x) + (self.B @ u)

    def outputMap(self, xk, u):
        # Linear output: y_k = C x_k + D u_k
        return (self.C @ xk) + (self.D @ u)

    def systemDynamics(self, dim, flag=True):
        x_k = np.ones((self.stateSize, 1))
        y_n = np.zeros((dim, self.outputSize))
        u_n = np.random.normal(1, 1.0, size=(dim, self.inputSize))

        for i in range(dim):
            if i % 10000 == 0:
                print(".", end="")
            u = u_n[i : i + 1, :].T
            y_n[i, :] = self.outputMap(x_k, u).flatten()
            x_k = self.stateMap(x_k, u)

        return y_n, u_n

    def loop(self, x_k, du_k):
        """
        Simulates the system's response to a sequence of input changes.

        Args:
            x_k: Initial state vector (6x1).
            du_k: Sequence of input changes (list of 3x1 arrays).

        Returns:
            y_n: Output sequence (list of 6x1 arrays).
            x_k: Final state vector (6x1).
        """
        y_n = []
        x_k = np.reshape(np.array(x_k), (self.stateSize, 1))
        for i in range(len(du_k)):
            u = np.reshape(np.array(du_k[i]), (self.inputSize, 1))
            # u = np.multiply(u, self.stdU.reshape(-1, 1)) + self.meanU.reshape(-1, 1)
            y = self.outputMap(x_k, u)
            # y = (y - self.meanY.reshape(-1, 1)) / self.stdY.reshape(-1, 1)
            y_n.append(y)
            x_k = self.stateMap(x_k, u)

        return np.array(y_n).squeeze(), x_k

    def prepareDataset(self, sizeT, sizeV):
        y_n, u_n = self.systemDynamics(sizeT, True)
        y_Vn, u_Vn = self.systemDynamics(sizeV, True)

        self.meanY = np.mean(y_n, axis=0)
        self.meanU = np.mean(u_n, axis=0)
        self.stdY = np.std(y_n, axis=0)
        self.stdU = np.std(u_n, axis=0)

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
