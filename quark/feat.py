import math
import warnings
import torch

GATE_VOCAB = {
    'h': 1, 'x': 2, 'y': 3, 'z': 4,
    's': 5, 'sdg': 5,
    't': 6, 'tdg': 6,
    'sx': 7, 'sxdg': 7,
    'rx': 8, 'ry': 9, 'rz': 10,
    'cx': 11, 'cy': 11, 'cz': 12,
    'swap': 13,
}
PAD = 0
OTHER_1Q = 14
OTHER_2Q = 15
VOCAB = 16
MAX_QUBITS = 64
MAX_LEN = 128


def encode(qc, max_len=MAX_LEN):
    g = []
    q1 = []
    q2 = []
    p = []
    for inst in qc.data:
        op = inst.operation
        gid = GATE_VOCAB.get(op.name)
        if gid is None:
            gid = OTHER_2Q if len(inst.qubits) == 2 else OTHER_1Q
        if len(inst.qubits) == 2:
            a = qc.find_bit(inst.qubits[0]).index
            b = qc.find_bit(inst.qubits[1]).index
            q1.append(min(a + 1, MAX_QUBITS))
            q2.append(min(b + 1, MAX_QUBITS))
        else:
            a = qc.find_bit(inst.qubits[0]).index
            q1.append(min(a + 1, MAX_QUBITS))
            q2.append(0)
        try:
            theta = float(op.params[0]) if op.params else 0.0
            p.append([math.cos(theta), math.sin(theta)])
        except (TypeError, ValueError, IndexError):
            p.append([1.0, 0.0])
        g.append(gid)

    if len(g) > max_len:
        warnings.warn(
            f"circuit exceeds the {max_len}-gate context window and was truncated; "
            "embeddings lose information for circuits this long.",
            stacklevel=2,
        )
    g = g[:max_len]; q1 = q1[:max_len]; q2 = q2[:max_len]; p = p[:max_len]
    while len(g) < max_len:
        g.append(PAD); q1.append(0); q2.append(0); p.append([0.0, 0.0])

    return (
        torch.tensor(g, dtype=torch.long),
        torch.tensor(q1, dtype=torch.long),
        torch.tensor(q2, dtype=torch.long),
        torch.tensor(p, dtype=torch.float32),
    )


def encode_batch(qcs, max_len=MAX_LEN):
    G, Q1, Q2, P = [], [], [], []
    for qc in qcs:
        g, q1, q2, p = encode(qc, max_len)
        G.append(g); Q1.append(q1); Q2.append(q2); P.append(p)
    return torch.stack(G), torch.stack(Q1), torch.stack(Q2), torch.stack(P)


def pad_mask(g):
    return g == PAD
