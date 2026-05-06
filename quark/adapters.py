def from_pennylane(tape):
    from qiskit import QuantumCircuit
    n = max((max(w) for w in (op.wires for op in tape.operations) if w), default=0) + 1
    qc = QuantumCircuit(n)
    name_map = {
        'Hadamard': 'h', 'PauliX': 'x', 'PauliY': 'y', 'PauliZ': 'z',
        'S': 's', 'T': 't', 'SX': 'sx',
        'RX': 'rx', 'RY': 'ry', 'RZ': 'rz',
        'CNOT': 'cx', 'CZ': 'cz', 'SWAP': 'swap',
    }
    for op in tape.operations:
        nm = name_map.get(op.name)
        ws = list(op.wires)
        if nm == 'h': qc.h(ws[0])
        elif nm == 'x': qc.x(ws[0])
        elif nm == 'y': qc.y(ws[0])
        elif nm == 'z': qc.z(ws[0])
        elif nm == 's': qc.s(ws[0])
        elif nm == 't': qc.t(ws[0])
        elif nm == 'sx': qc.sx(ws[0])
        elif nm == 'rx': qc.rx(float(op.parameters[0]), ws[0])
        elif nm == 'ry': qc.ry(float(op.parameters[0]), ws[0])
        elif nm == 'rz': qc.rz(float(op.parameters[0]), ws[0])
        elif nm == 'cx': qc.cx(ws[0], ws[1])
        elif nm == 'cz': qc.cz(ws[0], ws[1])
        elif nm == 'swap': qc.swap(ws[0], ws[1])
    return qc


def from_cirq(circuit):
    from qiskit import QuantumCircuit
    qubits = sorted(circuit.all_qubits(), key=lambda q: (q.row, q.col) if hasattr(q, 'row') else q.x if hasattr(q, 'x') else 0)
    qmap = {q: i for i, q in enumerate(qubits)}
    qc = QuantumCircuit(len(qubits))
    for moment in circuit:
        for op in moment.operations:
            g = op.gate
            ws = [qmap[q] for q in op.qubits]
            n = type(g).__name__
            if n == 'HPowGate' and abs(getattr(g, 'exponent', 1) - 1) < 1e-9: qc.h(ws[0])
            elif n == 'XPowGate': qc.rx(float(g.exponent) * 3.141592653589793, ws[0])
            elif n == 'YPowGate': qc.ry(float(g.exponent) * 3.141592653589793, ws[0])
            elif n == 'ZPowGate': qc.rz(float(g.exponent) * 3.141592653589793, ws[0])
            elif n == 'CNotPowGate' or n == 'CXPowGate': qc.cx(ws[0], ws[1])
            elif n == 'CZPowGate': qc.cz(ws[0], ws[1])
            elif n == 'SwapPowGate': qc.swap(ws[0], ws[1])
    return qc


def from_qasm(text):
    from qiskit import QuantumCircuit
    return QuantumCircuit.from_qasm_str(text)


def load(path):
    with open(path) as f:
        return from_qasm(f.read())
