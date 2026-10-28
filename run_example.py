from __future__ import annotations

import argparse
import json

import numpy as np

from simulation.graph import GraphProblem
from simulation.probability import AerStateBackend, NumpyStateBackend
from simulation.sampling import semantic_uniforms


def run_example(backend: str, angles: list[float], seed: int, shots: int) -> dict:
    if backend not in ("numpy", "aer"):
        raise ValueError("Backend must be numpy or aer.")
    if not angles or len(angles) % 2:
        raise ValueError("Provide an equal number of gamma and beta angles.")
    if shots < 1:
        raise ValueError("Shots must be positive.")
    if seed < 0:
        raise ValueError("Seed must be nonnegative.")
    problem = GraphProblem("cycle4", 4, ((0, 1), (1, 2), (2, 3), (3, 0)))
    simulator = NumpyStateBackend() if backend == "numpy" else AerStateBackend()
    depth = len(angles) // 2
    probabilities = simulator.probabilities(problem, angles, depth)
    uniforms = semantic_uniforms(f"example/{seed}", "sample", "basis", 0, shots)
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    samples = np.searchsorted(cumulative, uniforms, side="right")
    counts = np.bincount(samples, minlength=len(probabilities))
    return {
        "backend": backend,
        "graph": {"num_qubits": problem.num_qubits, "edges": problem.edges},
        "depth": depth,
        "angles": angles,
        "seed": seed,
        "shots": shots,
        "expected_cut": float(probabilities @ problem.cut_values),
        "sample_mean_cut": float(counts @ problem.cut_values / shots),
        "counts": {format(index, "04b"): int(count) for index, count in enumerate(counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate and sample a four-qubit QAOA circuit.")
    parser.add_argument("--backend", choices=("numpy", "aer"), default="numpy")
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[0.4, 0.2],
        help="Gamma angles followed by beta angles, in radians.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shots", type=int, default=1024)
    args = parser.parse_args()
    try:
        result = run_example(args.backend, args.angles, args.seed, args.shots)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
