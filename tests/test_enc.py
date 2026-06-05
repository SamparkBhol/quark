import torch
from quark.enc import CircuitEncoder, embed
from quark.pairs import random_circuit


def test_forward():
    enc = CircuitEncoder(d=64, h=4, layers=2)
    qcs = [random_circuit(3, 10, seed=i) for i in range(3)]
    e = embed(enc, qcs)
    assert e.shape == (3, 64)
    norms = e.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_param_count():
    enc = CircuitEncoder()
    n = enc.n_params()
    assert 100_000 < n < 30_000_000


def test_backward():
    enc = CircuitEncoder(d=32, h=2, layers=1)
    qcs = [random_circuit(2, 8, seed=i) for i in range(2)]
    from quark.feat import encode_batch
    g, q1, q2, p = encode_batch(qcs)
    e = enc(g, q1, q2, p)
    e.sum().backward()
    grads = [x.grad for x in enc.parameters() if x.grad is not None]
    assert grads
