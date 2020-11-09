import numpy as np
from .state import State


ket_0 = State("Zero", np.array([[1, 0]]).T, "|0>")
ket_1 = State("One", np.array([[0, 1]]).T, "|1>")
ket_p = State("Plus", 1 / np.sqrt(2) * np.array([[1, 1]]).T, "|+>")
ket_m = State("Minus", 1 / np.sqrt(2) * np.array([[1, -1]]).T, "|->")
