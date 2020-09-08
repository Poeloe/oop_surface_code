from circuit_simulation.states_and_gates import *
import numpy as np
from scipy import sparse as sp
from scipy.linalg import sqrtm
import random


def state_repr(state):
    """ Returns the visual representation of the given state if known """
    if np.array_equal(state, ket_0):
        return "|0>"
    if np.array_equal(state, ket_1):
        return "|1>"
    if np.array_equal(state, ket_p):
        return "|+>"
    if np.array_equal(state, ket_m):
        return "|->"
    return "|?>"


def gate_name(gate):
    """ Returns the (visual) representation of the given gate if known """
    if np.array_equal(gate, X_gate.matrix):
        return "X"
    if np.array_equal(gate, Y_gate.matrix):
        return "Y"
    if np.array_equal(gate, Z_gate.matrix):
        return "Z"
    if np.array_equal(gate, I_gate.matrix):
        return "I"
    if np.array_equal(gate, H_gate.matrix):
        return "H"
    if np.array_equal(gate, S_gate.matrix):
        return "S"
    return "?"


def get_value_by_prob(array, p):
    """ Returns, bases on the given weights 'p', a value out of the given array """
    r = random.random()
    index = 0
    while r >= 0 and index < len(p):
        r -= p[index]
        index += 1
    return array[index - 1]


def KP(*args):
    """ Returns the Kronecker product of the given arguments in the exact order """
    result = None
    for state in args:
        if state is None:
            continue
        if type(state) == State:
            state = state.sp_vector
        if type(state) in [SingleQubitGate, TwoQubitGate]:
            state = state.sp_matrix
        if not sp.issparse(state):
            state = sp.csr_matrix(state)
        if result is None:
            result = state
            continue
        result = sp.kron(result, state, format='csr')
    return result


def CT(state1, state2=None):
    """ returns the dot prodcut of the two passed states, where the second state will be the conjugate transpose """
    if type(state1) == State:
        state1 = state1.sp_vector
    elif not sp.issparse(state1):
        state1 = sp.csr_matrix(state1)

    if state2 is not None:
        if type(state2) == State:
            state2 = state2.sp_vector
        elif not sp.issparse(state2):
            state2 = sp.csr_matrix(state2)
    else:
        state2 = state1

    result = state1.dot(state2.conj().T)
    return result


def trace(sparse_matrix):
    """ Returns the trace of a matrix"""
    if not sp.issparse(sparse_matrix):
        sp.csr_matrix(sparse_matrix)

    result = sparse_matrix.diagonal().sum()

    if isinstance(result, complex):
        result = result.real

    return result


def N_dim_ket_0_or_1_density_matrix(N, ket=0):
    """ Returns an N-qubit version of the ket_0 or ket_1 density matrix """
    dim = 2**N
    rho = sp.lil_matrix((dim, dim))
    if ket == 1:
        rho[dim, dim] = 1
    else:
        rho[0, 0] = 1
    return rho


def fidelity(rho, sigma):
    """ Calculates the fidelity of two density matrices according to the 'classical' method """
    if not sp.issparse(rho):
        rho = sp.csr_matrix(rho)
    if not sp.issparse(sigma):
        sigma = sp.csr_matrix(sigma)

    rho_root = sp.csr_matrix(sqrtm(rho.toarray()))
    resulting_matrix = sqrtm((rho_root * (sigma * rho_root)).toarray())
    return (trace(resulting_matrix))**2


def fidelity_elementwise(rho, sigma):
    """ Calculates the fidelity using the element wise multiplication method """
    if not sp.issparse(rho):
        rho = rho.toarray()
    if not sp.issparse(sigma):
        sigma = sigma.toarray()

    resulting_matrix = rho * sigma * rho
    return trace(resulting_matrix)
