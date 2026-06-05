import random
from quark.pairs import make_pair, random_circuit, rewrite_chain, insert_id, cancel_consecutive
from quark.verify import equiv


def test_random_circuit_qubits():
    qc = random_circuit(3, 12, seed=0)
    assert qc.num_qubits == 3


def test_pair_same_qubits():
    a, b = make_pair(3, 10, k=3, seed=1)
    assert a.num_qubits == b.num_qubits


def test_pair_equivalent_small():
    failures = []
    for s in range(8):
        a, b = make_pair(2, 8, k=4, seed=s)
        if not equiv(a, b):
            failures.append(s)
    assert not failures, f"rewrite changed semantics on seeds: {failures}"


def test_insert_id_lengthens():
    a = random_circuit(2, 6, seed=0)
    rng = random.Random(0)
    b = insert_id(a, rng)
    assert len(b.data) == len(a.data) + 2


def test_cancel_does_not_break():
    a = random_circuit(3, 10, seed=2)
    rng = random.Random(0)
    b = cancel_consecutive(a, rng)
    assert b.num_qubits == a.num_qubits
