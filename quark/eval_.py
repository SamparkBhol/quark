import json
import random
import torch

from .enc import CircuitEncoder, embed
from .pairs import random_circuit, rewrite_chain
from .search import hash_repr, sorted_repr, count_vec, collect_vocab, cosine_topk, hash_topk


def build_pool(n_queries, lib_size, equiv_per_query, qmax=4, depth_range=(8, 24), k=4, seed=42, rewrites=None):
    rng = random.Random(seed)
    queries = []
    library = []
    truth = []
    fillers = lib_size - n_queries * equiv_per_query

    for i in range(n_queries):
        n = rng.randint(2, qmax)
        d = rng.randint(*depth_range)
        q = random_circuit(n, d, seed=seed * 999 + i)
        queries.append(q)
        idxs = []
        for j in range(equiv_per_query):
            sub_rng = random.Random(seed * 13 + i * 100 + j)
            qe = rewrite_chain(q, k, sub_rng, rewrites=rewrites)
            idxs.append(len(library))
            library.append(qe)
        truth.append(idxs)

    for i in range(fillers):
        n = rng.randint(2, qmax)
        d = rng.randint(*depth_range)
        library.append(random_circuit(n, d, seed=seed * 11 + i + 99999))

    perm = list(range(len(library)))
    rng.shuffle(perm)
    inv = [0] * len(perm)
    for new, old in enumerate(perm):
        inv[old] = new
    library = [library[old] for old in perm]
    truth = [[inv[idx] for idx in t] for t in truth]

    return queries, library, truth


def recall_at_k(retrieved, truth, k):
    hits = 0
    total = 0
    for r, t in zip(retrieved, truth):
        rk = r[:k]
        for g in t:
            total += 1
            if g in rk:
                hits += 1
    return hits / max(total, 1)


def run(weights=None, n_queries=100, lib_size=500, equiv_per_query=2,
        qmax=4, depth_range=(8, 24), k=4, seed=42, out=None, device='cpu', rewrites=None):
    queries, library, truth = build_pool(n_queries, lib_size, equiv_per_query, qmax, depth_range, k, seed, rewrites=rewrites)

    enc = CircuitEncoder().to(device)
    if weights:
        sd = torch.load(weights, map_location=device, weights_only=True)
        enc.load_state_dict(sd)
    q_emb = embed(enc, queries, device=device)
    l_emb = embed(enc, library, device=device)
    quark_top = cosine_topk(q_emb, l_emb, k=20)

    lib_h = [hash_repr(c) for c in library]
    q_h = [hash_repr(c) for c in queries]
    hash_top = hash_topk(q_h, lib_h, k=20)

    lib_s = [sorted_repr(c) for c in library]
    q_s = [sorted_repr(c) for c in queries]
    sorted_top = hash_topk(q_s, lib_s, k=20)

    vocab = collect_vocab(library + queries)
    lib_cv = torch.tensor([count_vec(c, vocab) for c in library], dtype=torch.float32)
    q_cv = torch.tensor([count_vec(c, vocab) for c in queries], dtype=torch.float32)
    cv_top = cosine_topk(q_cv, lib_cv, k=20)

    res = {}
    for K in (1, 5, 10):
        res[f'quark_r@{K}'] = recall_at_k(quark_top, truth, K)
        res[f'count_r@{K}'] = recall_at_k(cv_top, truth, K)
        res[f'hash_r@{K}'] = recall_at_k(hash_top, truth, K)
        res[f'sorted_r@{K}'] = recall_at_k(sorted_top, truth, K)

    print()
    print(f"  pool: {n_queries} queries, {lib_size} library, {equiv_per_query} equivalents per query")
    print()
    print(f"  method   |  R@1   |  R@5   |  R@10")
    print(f"  ---------|--------|--------|-------")
    for m in ('hash', 'sorted', 'count', 'quark'):
        r1, r5, r10 = res[f'{m}_r@1'], res[f'{m}_r@5'], res[f'{m}_r@10']
        print(f"  {m:8s} | {r1:.3f}  | {r5:.3f}  | {r10:.3f}")
    print()

    if out:
        with open(out, 'w') as f:
            json.dump(res, f, indent=2)

    return res


def smoke_bench(n_circuits=50, qmax=8, seed=0, weights=None, device='cpu'):
    # fast end-to-end sanity check (generate -> encode -> embed); the hard
    # guarantee is unit-norm embeddings. used by CI and runs without weights.
    import time

    if n_circuits < 1:
        raise ValueError("n_circuits must be >= 1")

    rng = random.Random(seed)
    circuits = [
        random_circuit(rng.randint(2, max(2, qmax)), rng.randint(8, 24), seed=seed * 1000 + i)
        for i in range(n_circuits)
    ]

    enc = CircuitEncoder().to(device)
    tag = 'random init'
    if weights:
        enc.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        tag = weights

    t0 = time.perf_counter()
    emb = embed(enc, circuits, device=device)
    dt = time.perf_counter() - t0

    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "embeddings are not unit-norm"

    eq_sims, rand_sims = [], []
    for i in range(min(n_circuits, 20)):
        eqc = rewrite_chain(circuits[i], 4, random.Random(seed * 7 + i))
        pair = embed(enc, [circuits[i], eqc], device=device)
        eq_sims.append(float((pair[0] * pair[1]).sum()))
        if n_circuits > 1:
            rand_sims.append(float((emb[i] * emb[(i + 1) % n_circuits]).sum()))

    eq_mean = sum(eq_sims) / len(eq_sims)
    rand_mean = sum(rand_sims) / len(rand_sims) if rand_sims else None

    print()
    print(f"  weights:        {tag}")
    print(f"  circuits:       {n_circuits} (up to {qmax} qubits), seed {seed}")
    print(f"  embed time:     {dt * 1000:.1f} ms ({n_circuits / dt:.0f} circuits/s)")
    print(f"  equivalent sim: {eq_mean:.3f} (mean)")
    if rand_mean is not None:
        print(f"  random sim:     {rand_mean:.3f} (mean)")
        print(f"  separation:     {eq_mean - rand_mean:+.3f}")
    else:
        print("  random sim:     n/a (needs >= 2 circuits)")
    print()

    return {'embed_s': dt, 'equiv_sim': eq_mean, 'random_sim': rand_mean}
