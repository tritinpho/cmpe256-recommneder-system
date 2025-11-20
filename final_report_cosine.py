import numpy as np
from scipy.sparse import csr_matrix as csr, diags
import streamlit as st 

# ---------- 1) Read interactions ----------
user2id, item2id = dict(), dict()
user_order = list()
rows, cols = list(), list()
uid = iid = 0

# ---------- 1) Read -----------
with open("train-1.txt", "r", encoding="utf-8") as g:
    for line in g:
        parts = line.strip().split()
        u_raw, items = parts[0], parts[1:]
        if u_raw not in user2id:
            user2id[u_raw] = uid
            user_order.append(u_raw)
            uid += 1
        u = user2id[u_raw]
        observed = set()
        for it in items:
            if it in observed:
                continue
            observed.add(it)
            if it not in item2id:
                item2id[it] = iid
                iid += 1
            j = item2id[it]
            rows.append(u); cols.append(j)

n_users, n_items = uid, iid
data = np.ones(len(rows), dtype=np.float32)
R = csr((data, (np.array(rows), np.array(cols))), shape=(n_users, n_items), dtype=np.float32)

# ---------- 2) Build cosine item-item similarity ----------
# Co-occurrence
C = (R.T @ R).astype(np.float32)   # (items x items)
# Zero diagonal (no self-sim)
C.setdiag(0.0)
C.eliminate_zeros()

# Norms
freq = R.sum(axis=0).A.ravel().astype(np.float32)  # item supports
denom = np.sqrt(np.clip(freq, 1e-12, None)).astype(np.float32)
inv = 1.0 / denom
Dinv = diags(inv)

S = (Dinv @ C @ Dinv).tocsc()
S.setdiag(0.0)
S.eliminate_zeros()

# ---------- 3) Keep top-K neighbors per column ----------
Sa = S.copy()
for j in range(Sa.shape[1]):
    start, end = Sa.indptr[j], Sa.indptr[j+1]
    col = Sa.data[start:end]
    if col.size > 10:
        keep = np.argpartition(np.abs(col), -10)[-10:]
        mask = np.zeros_like(col, dtype=bool)
        mask[keep] = True
        col[~mask] = 0.0
        Sa.data[start:end] = col
Sa.eliminate_zeros()

# ---------- 4) Normalize neighbor lists (column sums = 1) ----------
col_sums = np.array(Sa.sum(axis=0)).ravel().astype(np.float32)
col_sums[col_sums == 0] = 1e-12
Dinv = diags(1.0 / col_sums)

# ---------- 5) Score & recommend ----------
S_norm = Sa @ Dinv
scores = (R @ S_norm).tocsr()  # sparse user x item
recs = list()
for u in range(R.shape[0]):
    # candidate indices and scores from sparse row
    row = scores.getrow(u)
    idx, val = row.indices, row.data
    if idx.size == 0:
        recs.append([]) 
        continue

    # mask seen
    seen = set(R.indices[R.indptr[u]:R.indptr[u+1]])
    keep = [(j, s) for j, s in zip(idx, val) if j not in seen and np.isfinite(s)]
    if not keep:
        recs.append([]) 
        continue

    if len(keep) > 20:
        vals = np.fromiter((s for _, s in keep), dtype=np.float32)
        part = np.argpartition(vals, -20)[-20:]
        chosen = [keep[k] for k in part]
        chosen.sort(key=lambda x: -x[1])
        recs.append([j for j, _ in chosen])
    else:
        keep.sort(key=lambda x: -x[1])
        recs.append([j for j, _ in keep[:20]])

# ---------- 6) Popularity backfill ----------
pop = R.sum(axis=0).A.ravel()
pop_order = np.argsort(-pop)
out = []
for u, top in enumerate(recs):
    seen = set(R.indices[R.indptr[u]:R.indptr[u+1]])
    top = list(top)
    if len(top) < 20:
        have = set(top)
        for j in pop_order:
            if j not in have and j not in seen:
                top.append(j)
                if len(top) == 20: 
                    break
    out.append(top[:20])

# ---------- 7) Write output ----------
# reverse item map
n_items = len(item2id)
id2item = np.empty(n_items, dtype=object)
for it, j in item2id.items():
    id2item[j] = it

with open("recommendations_cosine10_new7.txt", "w", newline="", encoding="utf-8") as g:
    for u_idx, u_raw in enumerate(user_order):
        items = [int(id2item[j]) for j in out[u_idx]]
        items.sort()
        g.write(u_raw + " " + " ".join(map(str, items)) + "\n")