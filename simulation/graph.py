from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class GraphProblem:
    name: str
    num_qubits: int
    edges: tuple[tuple[int, int], ...]
    split: str = "evaluation"
    family: str = "unspecified"

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("A MaxCut graph needs at least two vertices.")
        if not self.edges:
            raise ValueError("A MaxCut graph needs at least one edge.")
        for u, v in self.edges:
            if u == v or min(u, v) < 0 or max(u, v) >= self.num_qubits:
                raise ValueError(f"Invalid edge {(u, v)} for {self.num_qubits} qubits.")
        canonical_edges = {tuple(sorted((u, v))) for u, v in self.edges}
        if len(canonical_edges) != len(self.edges):
            raise ValueError("A MaxCut graph cannot contain duplicate undirected edges.")
        adjacency = {vertex: set() for vertex in range(self.num_qubits)}
        for u, v in canonical_edges:
            adjacency[u].add(v)
            adjacency[v].add(u)
        reached = {0}
        frontier = [0]
        while frontier:
            vertex = frontier.pop()
            for neighbor in adjacency[vertex] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
        if len(reached) != self.num_qubits:
            raise ValueError("A MaxCut graph must be connected.")

    @cached_property
    def cut_values(self) -> np.ndarray:
        return np.asarray(
            [cut_value(index, self.edges) for index in range(2**self.num_qubits)], dtype=float
        )

    @cached_property
    def max_cut(self) -> float:
        return float(np.max(self.cut_values))

    @property
    def objective_bound(self) -> float:
        return float(len(self.edges))


def cut_value(index: int, edges: Iterable[tuple[int, int]]) -> int:
    return sum((index >> u & 1 != index >> v & 1 for u, v in edges))
