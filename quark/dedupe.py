import os
import json
import torch
from collections import defaultdict

from .enc import CircuitEncoder, embed
from .adapters import load


def _walk(root):
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith('.qasm'):
                yield os.path.join(dp, f)


def cluster(paths, weights, threshold=0.9, device='cpu', verify=False):
    enc = CircuitEncoder().to(device)
    sd = torch.load(weights, map_location=device, weights_only=True)
    enc.load_state_dict(sd)

    qcs = []
    kept_paths = []
    failed = []
    for p in paths:
        try:
            qcs.append(load(p))
            kept_paths.append(p)
        except Exception as e:
            failed.append((p, str(e)))

    if not qcs:
        return {'groups': [], 'failed': failed}

    embs = embed(enc, qcs, device=device)
    sims = embs @ embs.t()

    n = len(qcs)
    visited = [False] * n
    groups = []
    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(i + 1, n):
            if visited[j]:
                continue
            if sims[i, j].item() >= threshold:
                if verify:
                    from .verify import equiv
                    if qcs[i].num_qubits == qcs[j].num_qubits and qcs[i].num_qubits <= 8:
                        if not equiv(qcs[i], qcs[j]):
                            continue
                group.append(j)
                visited[j] = True
        if len(group) > 1:
            groups.append([kept_paths[k] for k in group])

    return {'groups': groups, 'failed': failed, 'n_files': n}


def fmt_report(result):
    out = []
    g = result['groups']
    if not g:
        out.append(f"no duplicate groups found in {result.get('n_files', 0)} files.")
    else:
        out.append(f"found {len(g)} duplicate group(s) in {result['n_files']} files:\n")
        for i, group in enumerate(g):
            out.append(f"  group {i+1} ({len(group)} files):")
            for p in group:
                out.append(f"    - {p}")
            out.append("")
    if result['failed']:
        out.append(f"\n{len(result['failed'])} file(s) failed to parse:")
        for p, e in result['failed'][:10]:
            out.append(f"  - {p}: {e}")
    return "\n".join(out)
