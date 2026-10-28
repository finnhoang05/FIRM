# QAOA simulation utilities

Small Python utilities for simulating the Quantum Approximate Optimization Algorithm on unweighted MaxCut graphs.

The package includes NumPy and Qiskit Aer statevector backends, graph validation, and deterministic random streams that support reproducible sampling in separate batches.

## Setup

Use Python 3.12 on macOS or Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

## Example

```bash
python run_example.py --backend numpy
python run_example.py --backend aer
```

The example evaluates a four-vertex cycle and samples its bitstring distribution. Run `python run_example.py --help` for options.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Files

| File | Purpose |
| --- | --- |
| `simulation/graph.py` | Connected, unweighted graph representation and MaxCut values |
| `simulation/probability.py` | NumPy and Qiskit Aer statevector simulation |
| `simulation/sampling.py` | Deterministic random streams with offset-based sampling |
| `run_example.py` | Small command-line example |
| `tests/test_simulation.py` | Numerical and input-validation checks |
| `requirements.txt` | Direct dependencies |
| `requirements-lock.txt` | Pinned environment |
