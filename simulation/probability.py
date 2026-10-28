from __future__ import annotations
from functools import lru_cache
import time
import numpy as np


def _graph_key(problem):
    n = int(problem.num_qubits)
    edges = tuple(sorted((tuple(sorted((int(u), int(v)))) for u, v in problem.edges)))
    if n < 2 or n > 20 or (not edges) or (len(set(edges)) != len(edges)):
        raise ValueError("Require 2..20 qubits and unique nonempty unweighted edges")
    if any((u == v or u < 0 or v >= n for u, v in edges)):
        raise ValueError("Invalid graph endpoint")
    return (n, edges)


@lru_cache(maxsize=16)
def _cut_values(n, edges):
    indices = np.arange(2**n, dtype=np.int64)
    values = np.zeros(2**n, dtype=np.float64)
    for u, v in edges:
        values += (indices >> u ^ indices >> v) & 1
    values.flags.writeable = False
    return values


def split_angles(theta, depth):
    if not isinstance(depth, (int, np.integer)) or depth < 1:
        raise ValueError("Positive integer QAOA depth required")
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (2 * depth,) or not np.isfinite(theta).all():
        raise ValueError("Expected finite [gamma_0..p-1, beta_0..p-1]")
    return (theta[:depth], theta[depth:])


def project_angles(theta, depth):
    gammas, betas = split_angles(theta, depth)
    return np.concatenate((np.clip(gammas, 0, np.pi), np.clip(betas, 0, np.pi / 2)))


class NumpyStateBackend:
    name = "numpy-statevector-reshape-v1"

    def __init__(self):
        self.probability_calls = 0
        self.host_seconds = 0.0

    def probabilities(self, problem, theta, depth):
        n, edges = _graph_key(problem)
        gammas, betas = split_angles(theta, depth)
        start = time.perf_counter()
        self.probability_calls += 1
        try:
            costs = _cut_values(n, edges)
            state = np.full(2**n, 1 / np.sqrt(2**n), dtype=np.complex128)
            for gamma, beta in zip(gammas, betas):
                state *= np.exp(-1j * gamma * costs)
                cosine, sine = (float(np.cos(beta)), float(np.sin(beta)))
                for q in range(n):
                    chunks = state.reshape(-1, 2, 1 << q)
                    zero, one = (chunks[:, 0, :].copy(), chunks[:, 1, :].copy())
                    chunks[:, 0, :] = cosine * zero - 1j * sine * one
                    chunks[:, 1, :] = -1j * sine * zero + cosine * one
            p = state.real * state.real + state.imag * state.imag
            p /= p.sum()
            return p
        finally:
            self.host_seconds += time.perf_counter() - start


class AerStateBackend:
    name = "aer-ideal-probabilities-v1"

    def __init__(self, max_parallel_threads=1):
        from qiskit_aer import AerSimulator

        self._simulator = AerSimulator(
            method="statevector",
            precision="double",
            max_parallel_threads=int(max_parallel_threads),
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )
        self.probability_calls = 0
        self.host_seconds = 0.0

    def probabilities(self, problem, theta, depth):
        from qiskit import QuantumCircuit

        n, edges = _graph_key(problem)
        gammas, betas = split_angles(theta, depth)
        circuit = QuantumCircuit(n)
        circuit.h(range(n))
        for gamma, beta in zip(gammas, betas):
            for u, v in edges:
                circuit.rzz(-float(gamma), u, v)
            for q in range(n):
                circuit.rx(2 * float(beta), q)
        circuit.save_probabilities(label="pre_readout_probabilities")
        self.probability_calls += 1
        start = time.perf_counter()
        try:
            result = self._simulator.run(circuit, shots=1, seed_simulator=1).result()
            if not result.success:
                raise RuntimeError("Aer probability evaluation failed: " + str(result.status))
            p = np.asarray(result.data(0)["pre_readout_probabilities"], dtype=np.float64)
            if p.shape != (2**n,) or not np.isfinite(p).all() or np.any(p < -1e-14):
                raise RuntimeError("Invalid Aer pre-readout probability vector")
            p = np.maximum(p, 0)
            p /= p.sum()
            return p
        finally:
            self.host_seconds += time.perf_counter() - start
