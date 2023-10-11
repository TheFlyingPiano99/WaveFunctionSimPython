from numba import jit, vectorize
import numpy as np
import cupy as cp
import sources.math_utils as math_utils

@jit(nopython=True)
def init_kinetic_operator(N: int, delta_x: float, delta_time: float):
    P_kinetic = np.zeros(shape=(N, N, N), dtype=np.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                f = np.array([x, y, z]) / np.array([N, N, N])
                # Fix numpy fftn-s "negative frequency in second half issue"
                if f[0] > 0.5:
                    f[0] = 1.0 - f[0]
                if f[1] > 0.5:
                    f[1] = 1.0 - f[1]
                if f[1] > 0.5:
                    f[2] = 1.0 - f[2]
                k = 2.0 * np.pi * f / delta_x
                angle = np.dot(k, k) * delta_time / 4.0
                P_kinetic[x, y, z] = math_utils.exp_i(angle)
    return P_kinetic


@jit(nopython=True)
def init_potential_operator(V:np.ndarray, N:int, delta_time:float):
    P_potential = np.zeros(shape=(N, N, N), dtype=cp.complex_)
    for x in range(0, N):
        for y in range(0, N):
            for z in range(0, N):
                angle = -V[x, y, z] * delta_time
                P_potential[x, y, z] = math_utils.exp_i(angle)
    return P_potential
