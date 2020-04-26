import numpy as np

ket_0 = np.array([[1, 0]]).T
ket_1 = np.array([[0, 1]]).T
ket_p = 1 / np.sqrt(2) * (ket_0 + ket_1)
ket_m = 1 / np.sqrt(2) * (ket_0 - ket_1)
ket_00 = np.array([[1, 0, 0, 0]]).T
ket_11 = np.array([[0, 0, 0, 1]]).T

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
