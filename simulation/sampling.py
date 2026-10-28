from __future__ import annotations
import hashlib
import json
import numpy as np

SAMPLER_VERSION = "absolute64-counter-PCG64-v1"


def semantic_uniforms(
    namespace: str, event: str, stream: str, start: int, shots: int
) -> np.ndarray:
    if start < 0 or shots < 0 or start + shots >= 2**63:
        raise ValueError("Invalid absolute shot interval")
    token = json.dumps([namespace, event, stream, SAMPLER_VERSION], separators=(",", ":")).encode()
    key = np.frombuffer(hashlib.blake2b(token, digest_size=32).digest(), dtype="<u4")
    bitgen = np.random.PCG64(np.random.SeedSequence(key.tolist()))
    bitgen.advance(int(start))
    x = bitgen.random_raw(shots)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)
