from circuit_simulation.states_and_gates import *
import numpy as np
from scipy import sparse as sp
import random


def state_repr(state):
    if np.array_equal(state, ket_0):
        return "|0>"
    if np.array_equal(state, ket_1):
        return "|1>"
    if np.array_equal(state, ket_p):
        return "|+>"
    if np.array_equal(state, ket_m):
        return "|->"


def gate_name(gate):
    if np.array_equal(gate, X):
        return "X"
    if np.array_equal(gate, Y):
        return "Y"
    if np.array_equal(gate, Z):
        return "Z"
    if np.array_equal(gate, I):
        return "I"
    if np.array_equal(gate, H):
        return "H"
    return None


def get_value_by_prob(array, p):
    r = random.random()
    index = 0
    while r >= 0 and index < len(p):
        r -= p[index]
        index += 1
    return array[index - 1]


def KP(*args):
    result = None
    for state in args:
        if state is None:
            continue
        if result is None:
            result = sp.csr_matrix(state)
            continue
        result = sp.csr_matrix(sp.kron(result, sp.csr_matrix(state)))
    return sp.csr_matrix(result)


def CT(state1, state2=None):
    state2 = state1 if state2 is None else state2
    return sp.csr_matrix(state1).dot(sp.csr_matrix(state2).conj().T)


def trace(sparse_matrix):
    return sparse_matrix.diagonal().sum()


def N_dim_ket_0_or_1_density_matrix(N, ket=0):
    dim = 2**N
    rho = sp.lil_matrix((dim, dim))
    if ket == 1:
        rho[dim, dim] = 1
    else:
        rho[0, 0] = 1
    return rho


def fidelity(rho, sigma):
    rho_root = np.sqrt(rho.toarray())
    resulting_matrix = np.sqrt((rho_root * sigma.toarray() * rho_root))
    return (trace(resulting_matrix))**2
