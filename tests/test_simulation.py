import unittest

import numpy as np

from run_example import run_example
from simulation.graph import GraphProblem, cut_value
from simulation.probability import AerStateBackend, NumpyStateBackend
from simulation.sampling import semantic_uniforms


class SimulationTests(unittest.TestCase):
    def test_single_edge_analytic_expectation(self):
        problem = GraphProblem("edge", 2, ((0, 1),))
        backend = NumpyStateBackend()
        for gamma, beta in ((0.3, 0.2), (np.pi / 2, np.pi / 8), (-0.7, 0.4)):
            with self.subTest(gamma=gamma, beta=beta):
                probabilities = backend.probabilities(problem, [gamma, beta], 1)
                expected = 0.5 * (1 + np.sin(4 * beta) * np.sin(gamma))
                self.assertAlmostEqual(float(probabilities @ problem.cut_values), expected, 13)

    def test_zero_angles_uniform_distribution(self):
        problem = GraphProblem("cycle", 4, ((0, 1), (1, 2), (2, 3), (3, 0)))
        probabilities = NumpyStateBackend().probabilities(problem, [0, 0, 0, 0], 2)
        np.testing.assert_allclose(probabilities, np.full(16, 1 / 16), atol=1e-14)
        self.assertAlmostEqual(float(probabilities @ problem.cut_values), 2.0, 13)
        self.assertEqual(cut_value(0b0101, problem.edges), 4)

    def test_numpy_and_aer_agree(self):
        problem = GraphProblem("triangle", 3, ((0, 1), (1, 2), (0, 2)))
        numpy_backend = NumpyStateBackend()
        aer_backend = AerStateBackend()
        for angles, depth in (([0.3, 0.2], 1), ([0.1, 0.8, -0.2, 0.4], 2)):
            with self.subTest(depth=depth):
                expected = numpy_backend.probabilities(problem, angles, depth)
                actual = aer_backend.probabilities(problem, angles, depth)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_sampler_chunk_invariance(self):
        whole = semantic_uniforms("test/7", "sample", "basis", 100, 257)
        parts = np.concatenate(
            [
                semantic_uniforms("test/7", "sample", "basis", 100, 80),
                semantic_uniforms("test/7", "sample", "basis", 180, 177),
            ]
        )
        np.testing.assert_array_equal(whole, parts)
        self.assertTrue(np.all((whole >= 0) & (whole < 1)))
        other = semantic_uniforms("test/8", "sample", "basis", 100, 257)
        self.assertFalse(np.array_equal(whole, other))

    def test_invalid_graphs(self):
        for n, edges in (
            (1, ((0, 0),)),
            (2, ()),
            (2, ((0, 0),)),
            (2, ((0, 2),)),
            (2, ((0, 1), (1, 0))),
            (3, ((0, 1),)),
        ):
            with self.subTest(n=n, edges=edges), self.assertRaises(ValueError):
                GraphProblem("invalid", n, edges)

    def test_invalid_angles_and_depth(self):
        problem = GraphProblem("edge", 2, ((0, 1),))
        for angles, depth in (([0, 0], 0), ([0, 0], 1.5), ([0], 1), ([0, np.nan], 1)):
            with self.subTest(angles=angles, depth=depth), self.assertRaises(ValueError):
                NumpyStateBackend().probabilities(problem, angles, depth)

    def test_invalid_sampling_intervals(self):
        for start, shots in ((-1, 2), (0, -1), (2**63 - 1, 1)):
            with self.subTest(start=start, shots=shots), self.assertRaises(ValueError):
                semantic_uniforms("test", "sample", "basis", start, shots)

    def test_example_reproducibility_and_counts(self):
        first = run_example("numpy", [0.4, 0.2], 7, 1000)
        self.assertEqual(first, run_example("numpy", [0.4, 0.2], 7, 1000))
        self.assertEqual(sum(first["counts"].values()), 1000)
        self.assertGreaterEqual(first["sample_mean_cut"], 0)
        self.assertLessEqual(first["sample_mean_cut"], 4)

    def test_invalid_example_arguments(self):
        for backend, angles, seed, shots in (
            ("unknown", [0, 0], 0, 10),
            ("numpy", [], 0, 10),
            ("numpy", [0], 0, 10),
            ("numpy", [0, 0], -1, 10),
            ("numpy", [0, 0], 0, 0),
        ):
            with self.subTest(backend=backend, angles=angles, seed=seed, shots=shots):
                with self.assertRaises(ValueError):
                    run_example(backend, angles, seed, shots)


if __name__ == "__main__":
    unittest.main()
