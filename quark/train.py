import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from .enc import CircuitEncoder
from .feat import encode_batch
from .pairs import make_pair, random_circuit, rewrite_chain


class TripletDS(Dataset):
    def __init__(self, n_samples=1500, qmax=4, depth_range=(8, 24), k=4, seed=0, rewrites=None):
        self.items = []
        rng = random.Random(seed)
        for s in range(n_samples):
            n = rng.randint(2, qmax)
            d = rng.randint(*depth_range)
            a, p = make_pair(n, d, k=k, seed=seed * 1000 + s, rewrites=rewrites)
            n_neg = rng.randint(2, qmax)
            d_neg = rng.randint(*depth_range)
            neg = random_circuit(n_neg, d_neg, seed=seed * 7777 + s)
            self.items.append((a, p, neg))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _collate(batch):
    a, p, n = zip(*batch)
    A = encode_batch(list(a))
    P = encode_batch(list(p))
    N = encode_batch(list(n))
    return A, P, N


def triplet(a, p, n, m=0.2):
    pd = (a - p).pow(2).sum(-1)
    nd = (a - n).pow(2).sum(-1)
    return torch.clamp(pd - nd + m, min=0).mean()


def info_nce(a, p, temp=0.07):
    logits = (a @ p.t()) / temp
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def train(samples=1500, epochs=10, batch=32, lr=3e-4, qmax=4,
          depth_range=(8, 24), k=4, save_to=None, device='cpu', seed=0,
          rewrites=None, loss='triplet'):
    ds = TripletDS(n_samples=samples, qmax=qmax, depth_range=depth_range, k=k, seed=seed, rewrites=rewrites)
    n_val = max(50, len(ds) // 10)
    n_tr = len(ds) - n_val
    tr, val = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(seed))
    dl = DataLoader(tr, batch_size=batch, collate_fn=_collate, shuffle=True)
    vdl = DataLoader(val, batch_size=batch, collate_fn=_collate)

    enc = CircuitEncoder().to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"params: {enc.n_params():,} loss={loss}")

    def _step(batch, train_step):
        A, P, N = batch
        A = [t.to(device) for t in A]
        P = [t.to(device) for t in P]
        N = [t.to(device) for t in N]
        ea = enc(*A); ep_ = enc(*P)
        if loss == 'infonce':
            return info_nce(ea, ep_)
        en = enc(*N)
        return triplet(ea, ep_, en)

    for ep in range(epochs):
        enc.train()
        ttot = 0; tn = 0
        for batch in dl:
            l = _step(batch, True)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            ttot += l.item(); tn += 1

        enc.train(False)
        vtot = 0; vn = 0
        with torch.no_grad():
            for batch in vdl:
                vtot += _step(batch, False).item(); vn += 1
        sched.step()
        print(f"ep {ep}: train={ttot/max(tn,1):.4f} val={vtot/max(vn,1):.4f}")

    if save_to:
        torch.save(enc.state_dict(), save_to)
        print(f"saved -> {save_to}")
    return enc
