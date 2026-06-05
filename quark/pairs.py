import math
import random
from qiskit import QuantumCircuit


def random_circuit(n, depth, seed=None):
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(depth):
        r = rng.random()
        if r < 0.4 and n >= 2:
            a, b = rng.sample(range(n), 2)
            qc.cx(a, b)
        elif r < 0.6:
            qc.h(rng.randrange(n))
        elif r < 0.7:
            qc.x(rng.randrange(n))
        elif r < 0.8:
            qc.s(rng.randrange(n))
        elif r < 0.9:
            qc.t(rng.randrange(n))
        else:
            qc.rz(rng.uniform(0, 2 * math.pi), rng.randrange(n))
    return qc


def _qidx(qc, q):
    return qc.find_bit(q).index


def insert_id(qc, rng):
    n = qc.num_qubits
    pos = rng.randrange(len(qc.data) + 1) if qc.data else 0
    out = QuantumCircuit(n)

    def emit_pair():
        kind = rng.choice(['h', 'x', 'cx']) if n >= 2 else rng.choice(['h', 'x'])
        if kind == 'h':
            q = rng.randrange(n)
            out.h(q); out.h(q)
        elif kind == 'x':
            q = rng.randrange(n)
            out.x(q); out.x(q)
        else:
            a, b = rng.sample(range(n), 2)
            out.cx(a, b); out.cx(a, b)

    for i, inst in enumerate(qc.data):
        if i == pos:
            emit_pair()
        out.append(inst.operation, inst.qubits, inst.clbits)
    if pos == len(qc.data):
        emit_pair()
    return out


def cancel_consecutive(qc, rng):
    n = qc.num_qubits
    out = QuantumCircuit(n)
    instrs = list(qc.data)
    skip = -1
    for i, inst in enumerate(instrs):
        if i == skip:
            continue
        if i + 1 < len(instrs):
            nxt = instrs[i + 1]
            if (inst.operation.name == nxt.operation.name
                    and inst.operation.name in ('h', 'x', 'y', 'z', 'cx', 'cz')
                    and tuple(inst.qubits) == tuple(nxt.qubits)):
                skip = i + 1
                continue
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def commute_swap(qc, rng):
    n = qc.num_qubits
    instrs = list(qc.data)
    if len(instrs) < 2:
        return qc.copy()
    cands = []
    for i in range(len(instrs) - 1):
        a = instrs[i]; b = instrs[i + 1]
        qa = {_qidx(qc, q) for q in a.qubits}
        qb = {_qidx(qc, q) for q in b.qubits}
        if not (qa & qb):
            cands.append(i)
    if not cands:
        return qc.copy()
    i = rng.choice(cands)
    instrs[i], instrs[i + 1] = instrs[i + 1], instrs[i]
    out = QuantumCircuit(n)
    for inst in instrs:
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def merge_rz(qc, rng):
    n = qc.num_qubits
    out = QuantumCircuit(n)
    instrs = list(qc.data)
    i = 0
    while i < len(instrs):
        cur = instrs[i]
        if (i + 1 < len(instrs)
                and cur.operation.name == 'rz'
                and instrs[i + 1].operation.name == 'rz'
                and tuple(cur.qubits) == tuple(instrs[i + 1].qubits)):
            theta = float(cur.operation.params[0]) + float(instrs[i + 1].operation.params[0])
            out.rz(theta, _qidx(qc, cur.qubits[0]))
            i += 2
        else:
            out.append(cur.operation, cur.qubits, cur.clbits)
            i += 1
    return out


def split_rz(qc, rng):
    instrs = list(qc.data)
    rz_idx = [i for i, inst in enumerate(instrs) if inst.operation.name == 'rz']
    if not rz_idx:
        return qc.copy()
    pick = rng.choice(rz_idx)
    n = qc.num_qubits
    out = QuantumCircuit(n)
    for j, inst in enumerate(instrs):
        if j == pick:
            theta = float(inst.operation.params[0])
            q = _qidx(qc, inst.qubits[0])
            out.rz(theta / 2, q)
            out.rz(theta / 2, q)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


REWRITES = [insert_id, cancel_consecutive, commute_swap, merge_rz, split_rz]
REWRITE_NAMES = {f.__name__: f for f in REWRITES}


def rewrite_chain(qc, k, rng, rewrites=None):
    cur = qc
    pool = rewrites if rewrites is not None else REWRITES
    for _ in range(k):
        op = rng.choice(pool)
        cur = op(cur, rng)
    return cur


def make_pair(n, depth, k=4, seed=None, rewrites=None):
    rng = random.Random(seed)
    a = random_circuit(n, depth, seed=seed)
    b = rewrite_chain(a, k, rng, rewrites=rewrites)
    return a, b
