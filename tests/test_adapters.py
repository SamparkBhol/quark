import os
import tempfile
from qiskit.qasm2 import dumps
from quark.pairs import random_circuit
from quark.adapters import from_qasm, load


def test_from_qasm_roundtrip():
    qc = random_circuit(2, 8, seed=0)
    qasm = dumps(qc)
    qc2 = from_qasm(qasm)
    assert qc.num_qubits == qc2.num_qubits
    assert len(qc.data) == len(qc2.data)


def test_load_from_file():
    qc = random_circuit(3, 10, seed=1)
    with tempfile.NamedTemporaryFile('w', suffix='.qasm', delete=False) as f:
        f.write(dumps(qc))
        path = f.name
    qc2 = load(path)
    os.unlink(path)
    assert qc2.num_qubits == 3
