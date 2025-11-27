def cosine_similarity():
    import numpy as np
    from scipy.sparse import csr_matrix as csr, diags
    import streamlit as st 

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

    with open("recommendations_cosine10.txt", "w", newline="", encoding="utf-8") as g:
        for u_idx, u_raw in enumerate(user_order):
            items = [int(id2item[j]) for j in out[u_idx]]
            items.sort()
            g.write(u_raw + " " + " ".join(map(str, items)) + "\n")

def jaccard_similarity():
    import collections
    import heapq
    from pathlib import Path

    # 1) Read
    user_items = collections.defaultdict(set)
    with open("train-1.txt", "r", encoding="utf-8") as g:
        for ln, line in enumerate(g, 1):
            # Split on any whitespace so tabs/spaces both work
            toks = line.strip().split()
            u = toks[0]
            items = toks[1:]
            # Use a set to deduplicate within a line
            if len(items) > 0:
                user_items[u].update(items)

    # 2) Invert and prune
    item_users = collections.defaultdict(set)
    for u, items in user_items.items():
        for it in items:
            item_users[it].add(u)

    # 3) Co-occurrence counts
    co_counts = collections.Counter()
    for u, items in user_items.items():
        if len(items) < 2:
            continue
        # Sorted for deterministic (i<j) ordering
        sitems = sorted(items)
        for i_idx in range(len(sitems)):
            i = sitems[i_idx]
            for j in sitems[i_idx+1:]:
                co_counts[(i, j)] += 1

    # 4) Item neighbors by Jaccard
    # Precompute item user counts
    item_deg = {it: len(users) for it, users in item_users.items()}
    # For each pair with co>0, compute Jaccard and push to both sides' heaps.
    heaps = collections.defaultdict(list)  # item -> min-heap of (sim, neighbor)
    for (i, j), inter in co_counts.items():
        ui, uj = item_deg.get(i, 0), item_deg.get(j, 0)
        if ui == 0 or uj == 0:
            continue
        union = ui + uj - inter
        if union <= 0:
            continue
        sim = inter / union
        if sim <= 0:
            continue
        # Push for i
        heapq.heappush(heaps[i], (sim, j))
        if len(heaps[i]) > 10:
            heapq.heappop(heaps[i])
        # Push for j
        heapq.heappush(heaps[j], (sim, i))
        if len(heaps[j]) > 10:
            heapq.heappop(heaps[j])
    # Convert heaps to sorted lists (descending by sim, tie-break by neighbor id)
    neighbors = dict()
    for it, h in heaps.items():
        # h is min-heap; get all and sort
        lst = [heapq.heappop(h) for _ in range(len(h))]
        lst.sort(key=lambda t: (-t[0], t[1]))
        neighbors[it] = lst

    # 5) Popularity fallback (global)
    popular = sorted(item_deg.items(), key=lambda kv: (-kv[1], kv[0]))
    popular_items = [it for it, _ in popular]

    # 6) Recommend per user
    out_path = Path("recommendations_jaccard10.txt")
    with open(out_path, "w", newline="", encoding="utf-8") as w:
        middle = []
        for u in sorted(user_items.keys()):  # deterministic user order
            scores = collections.defaultdict(float)
            seen = user_items[u]
            for it in user_items[u]:
                for sim, nb in neighbors.get(it, []):
                    if nb in seen:
                        continue
                    scores[nb] += sim
            if not scores:
                # cold-start: backfill by popularity
                recs = [int(it) for it in popular_items if it not in seen][:20]
            # Rank by score desc, then item id for determinism
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            recs = [int(it) for it, _ in ranked[:20]]
            if len(recs) < 20:
                # backfill with popularity
                for it in popular_items:
                    if len(recs) >= 20:
                        break
                    if it not in seen and it not in scores:
                        recs.append(int(it))
            recs.sort()
            # space-separated list in one cell
            middle.append([int(u), ' '.join(map(str, recs))])

        middle.sort(key=lambda x: x[0])
        for i in middle:
            j = str(i[0])
            w.write(f"{j} {i[1]}\n")

def pearson_similarity():
    import math
    from collections import defaultdict

    user_items = dict()
    with open("train-1.txt", "r", encoding="utf-8") as g:
        for line in g:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = parts[0]
            items = list(dict.fromkeys(parts[1:]))  # unique, preserve order
            user_items[user] = set(items)

    N = len(user_items)  # number of users
    co_counts = defaultdict(lambda: defaultdict(int))
    item_freq = defaultdict(int)
    items_all = set()

    for u, items in user_items.items():
        items_list = sorted(items)  # stable ordering to enforce i<j
        L = len(items_list)
        for it in items_list:
            item_freq[it] += 1
            items_all.add(it)
        for idx in range(L):
            i = items_list[idx]
            for jdx in range(idx + 1, L):
                j = items_list[jdx]
                co_counts[i][j] += 1

    # Build symmetric similarities
    neighbors = defaultdict(list)  # item -> list[(other_item, sim)]
    # Gather potential pairs from co_counts (only pairs with c>0)
    for i, row in co_counts.items():
        a = item_freq[i]
        for j, c in row.items():
            b = item_freq[j]
            # Edge cases: if an item is interacted by no user or all users, variance is zero
            if a == 0 or b == 0 or a == N or b == N:
                sim = 0.0
            num = c - (a * b) / float(N)
            denom = math.sqrt(a * (1.0 - a / float(N)) * b * (1.0 - b / float(N)))
            if denom == 0.0:
                sim = 0.0
            sim = num / denom
            if sim != 0.0:
                neighbors[i].append((j, sim))
                neighbors[j].append((i, sim))

    # For items that never co-occurred with others, make sure they exist in neighbors
    for it in items_all:
        neighbors[it] = neighbors[it]  # touch to ensure key exists

    # Keep only top-k by |similarity| for stability
    for it, nbrs in neighbors.items():
        nbrs.sort(key=lambda x: abs(x[1]), reverse=True)
        neighbors[it] = nbrs[:10]

    recs = dict()
    for u, items_u in user_items.items():
        scores = defaultdict(float)
        norm = defaultdict(float)

        for i in items_u:
            for j, sim in neighbors.get(i, []):
                if j in items_u:
                    continue
                scores[j] += sim
                norm[j] += abs(sim)

        # Normalized score
        ranked = list()
        for j, s in scores.items():
            denom = norm[j] if norm[j] > 0 else 1.0
            ranked.append((j, s / denom))

        ranked.sort(key=lambda x: x[1], reverse=True)
        recs[u] = [it for it, _ in ranked[:20]]

    with open("recommendations_pearson10.txt", "w", encoding="utf-8") as g:
        for u, items in recs.items():
            it = [int(i) for i in items]
            it.sort()
            g.write(f"{u} {' '.join(map(str, it))}\n")

def loo_evaluation():
    import math, random
    import numpy as np
    from collections import defaultdict
    from math import log2
    import torch
    import matplotlib.pyplot as plt

    random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    user_items = dict()
    with open("train-1.txt", "r", encoding="utf-8") as g:
        for line in g:
            tokens = line.strip().split()
            # data cleaning: skip lines with only users.
            if len(tokens) < 2:
                continue
            u = tokens[0] 
            # remove dupliacate items for every user.
            items = list(dict.fromkeys(tokens[1:]))
            user_items[u] = set(items)

    train_ui, test_ui = dict(), dict()
    for u, items in user_items.items():
        items = list(items)
        if len(items) <= 1:
            train_ui[u] = set(items)
            continue
        n_test = max(1, int(len(items)*0.2))
        tset = set(random.sample(items, n_test))
        tr = set(items) - tset
        if tr:
            train_ui[u] = tr
            if tset:
                test_ui[u] = tset

    # ---------- metrics ----------
    def average_ndcg(recs, test, k=20):
        vals = list()
        for u, rlist in recs.items():
            gt = test.get(u, set())
            if gt:
                dcg = 0.0
                for i in range(min(k, len(rlist))):
                    if rlist[i] in gt:
                        dcg += 1.0 / log2(i+2)
                ideal = min(len(gt), k)
                if ideal == 0: 
                    return 0.0
                idcg = sum(1.0 / log2(i+2) for i in range(ideal))
                middle = dcg / idcg
                vals.append(middle)
        return round(float(np.mean(vals)), 4) if vals else 0.0

    import random

    def split_leave_one_out(user_items, seed=42):
        """
        For each user with >=2 items, randomly hold out exactly one item for test.
        Returns:
        train_ui: dict[user] -> set(train_items)
        test_ui : dict[user] -> set({held_out_item})
        Users with <2 items are kept entirely in train (no test).
        """
        rng = random.Random(seed)
        train_ui, test_ui = {}, {}
        for u, items in user_items.items():
            items = list(items)
            if len(items) < 2:
                # not enough to hold out one -> all go to train, no test
                train_ui[u] = set(items)
                continue
            t = rng.choice(items)
            tr = set(items); tr.remove(t)
            train_ui[u] = tr
            test_ui[u]  = {t}
        return train_ui, test_ui

    import numpy as np

    def run_leave_one_out(
        user_items,
        k_values=(5,10,20,30,40,50,75,100),
        sim_types=("cosine","jaccard","pearson"),
        topn=20,
        seed=42,
        repeats=1,
        build_neighbors_fn=None,
        recommend_all_fn=None,   # recommend_all or recommend_all_gpu_wrapper
    ):
        """
        Leave-one-out evaluation with HR@K and MRR@K (K = topn).
        Returns:
        results[sim] = {
            "k_values": list(k_values),
            "HR":  [mean over repeats],
            "MRR": [mean over repeats],
            "per_repeat_HR":  np.ndarray shape (repeats, len(k_values)),
            "per_repeat_MRR": np.ndarray shape (repeats, len(k_values)),
        }
        """

        R_HR  = {s: np.zeros((repeats, len(k_values)), dtype=np.float64) for s in sim_types}
        R_MRR = {s: np.zeros((repeats, len(k_values)), dtype=np.float64) for s in sim_types}

        for r in range(repeats):
            train_ui, test_ui = split_leave_one_out(user_items, seed=seed + r)

            # Users that actually have a held-out item
            eval_users = list(test_ui.keys())
            if not eval_users:
                print(f"[LOO] Repeat {r+1}: no evaluable users (all had <2 items).")
                continue

            for k_idx, k in enumerate(k_values):
                for sim in sim_types:
                    # 1) Build neighbors on train interactions
                    neighbors, popular = build_neighbors_fn(train_ui, k, sim_type=sim)
                    # 2) Recommend for users in train_ui (your recommender uses their train subset)
                    recs = recommend_all_fn(train_ui, neighbors, popular, topn=topn)

                    # 3) LOO metrics
                    hits = 0.0
                    rr_sum = 0.0
                    for u in eval_users:
                        held_out = next(iter(test_ui[u]))      # exactly one item
                        rec_u = recs.get(u, [])
                        if not rec_u:
                            continue
                        try:
                            rank = rec_u.index(held_out) + 1    # 1-based rank if present
                            if rank <= topn:
                                hits += 1.0
                                rr_sum += 1.0 / rank
                        except ValueError:
                            # not present -> no hit, no RR contribution
                            pass

                    n_eval = float(len(eval_users))
                    hr  = hits / n_eval
                    mrr = rr_sum / n_eval

                    R_HR[sim][r, k_idx]  = hr
                    R_MRR[sim][r, k_idx] = mrr

            print(f"[LOO] Completed repeat {r+1}/{repeats}")

        # Aggregate
        results = {}
        for sim in sim_types:
            HR_mat  = R_HR[sim]
            MRR_mat = R_MRR[sim]
            results[sim] = {
                "k_values": list(k_values),
                "HR":  HR_mat.mean(axis=0).tolist(),
                "MRR": MRR_mat.mean(axis=0).tolist(),
                "per_repeat_HR":  HR_mat,
                "per_repeat_MRR": MRR_mat,
            }
        return results

    def cosine(a,b,c):
        return c / math.sqrt(a*b) if a and b else 0.0

    def jaccard(a,b,c):
        u = a + b - c
        return c / u if u else 0.0
    
    def pearson(N,a,b,c):
        if a == 0 or b == 0 or a == N or b == N:
            return 0.0
        num = c - (a*b)/N
        den = math.sqrt(a*(1 - a/N) * b*(1 - b/N))
        return num/den if den else 0.0

    def build_neighbors(train_ui, k, sim_type):
        N = len(train_ui)
        co = defaultdict(lambda: defaultdict(int))
        deg = defaultdict(int)
        items_all = set()
        for _, items in train_ui.items():
            s = sorted(items)
            items_all.update(s)
            for it in s:
                deg[it] += 1
            for i in range(len(s)):
                a = s[i]
                for j in range(i+1, len(s)):
                    b = s[j]
                    co[a][b] += 1
        if sim_type == "cosine":
            sim_fn = lambda a,b,c: cosine(a,b,c)
        elif sim_type == "jaccard":
            sim_fn = lambda a,b,c: jaccard(a,b,c)
        else:
            sim_fn = lambda a,b,c: pearson(N,a,b,c)

        nbrs = defaultdict(list)
        for i, row in co.items():
            ai = deg[i]
            for j, c in row.items():
                aj = deg[j]
                s = sim_fn(ai, aj, c)
                if s != 0.0:
                    nbrs[i].append((j, s))
                    nbrs[j].append((i, s))
        for it in list(nbrs.keys()):
            nbrs[it].sort(key=lambda t: abs(t[1]), reverse=True)
            nbrs[it] = nbrs[it][:k]
        popular = [it for it,_ in sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))]
        return nbrs, popular

    def utility(idx_list, items_u, idx2item, topn, pop_idx):
        ids = list()
        seen = items_u
        for j in idx_list:
            ids.append(idx2item[j])
            if len(ids) == topn:
                break
        if len(ids) < topn:
            have = set(ids) | seen
            for j in pop_idx:
                it = idx2item[j]
                if it not in have:
                    ids.append(it)
                    if len(ids) == topn:
                        break
        return ids

    def recommend_all_gpu(train_user_items, item_ids, item2idx, nbr_idx, nbr_val, popular_items, topn=20):
        idx2item = item_ids

        # Make a popularity backfill list in indices (faster on GPU/CPU mix)
        pop_idx = [item2idx[it] for it in popular_items if it in item2idx]
        all_recs = dict()
        for u, items_u in train_user_items.items():
            # indices first
            I, K = nbr_idx.shape

            if not items_u:
                # cold start: backfill by popularity
                idx_list = (popular_items or [])[:topn]
                all_recs[u] = utility(idx_list, items_u, idx2item, topn, pop_idx)

            # Map seen items -> indices
            seen_idx = [item2idx[it] for it in items_u if it in item2idx]
            if not seen_idx:
                idx_list = (popular_items or [])[:topn]
                all_recs[u] = utility(idx_list, items_u, idx2item, topn, pop_idx)

            seen_idx_t = torch.tensor(seen_idx, dtype=torch.long, device=device)  # (m,)

            # Gather neighbor rows for all seen items -> (m, K)
            nb_j   = nbr_idx.index_select(0, seen_idx_t)   # neighbor indices (padded with -1)
            nb_sim = nbr_val.index_select(0, seen_idx_t)   # neighbor weights (padded with 0)

            # Flatten contributions
            flat_j   = nb_j.reshape(-1)                    # (m*K,)
            flat_sim = nb_sim.reshape(-1)

            # Drop pads (-1)
            valid = flat_j >= 0
            flat_j   = flat_j[valid]                       # candidate indices
            flat_sim = flat_sim[valid]                     # their contributions

            # Mask out already-seen candidates
            seen_mask = torch.zeros(I, dtype=torch.bool, device=device)
            seen_mask[seen_idx_t] = True
            not_seen = ~seen_mask[flat_j]
            flat_j   = flat_j[not_seen]
            flat_sim = flat_sim[not_seen]

            if flat_j.numel() == 0:
                # no candidates via neighbors -> popularity backfill
                idx_list = [it for it in (popular_items or []) if it not in items_u][:topn]
                all_recs[u] = utility(idx_list, items_u, idx2item, topn, pop_idx)

            # Aggregate: sum(sim) and sum(|sim|) per candidate (for normalization)
            scores = torch.zeros(I, dtype=torch.float32, device=device)
            norms  = torch.zeros(I, dtype=torch.float32, device=device)
            scores.scatter_add_(0, flat_j, flat_sim)
            norms.scatter_add_(0, flat_j, flat_sim.abs())

            # Normalize; set seen to -inf so they're never recommended
            denom = torch.where(norms > 0, norms, torch.ones_like(norms))
            final_scores = scores / denom
            final_scores[seen_idx_t] = -float("inf")

            # Top-N
            n_eff = min(topn, I)
            top_scores, top_idx = torch.topk(final_scores, n_eff, largest=True)
            # Filter any -inf (in case user interacted with almost everything)
            mask_finite = torch.isfinite(top_scores) & (top_scores > -1e30)
            top_idx = top_idx[mask_finite].tolist()

            # Backfill with popularity if needed
            if popular_items:
                # convert indices -> ids
                # We need the id list from prepare_torch_neighbors
                # We'll pass it as a closure, or return it from the caller.
                pass

            idx_list = top_idx
            all_recs[u] = utility(idx_list, items_u, idx2item, topn, pop_idx)
        return all_recs

    def prepare_torch_neighbors(neighbors, all_items=None, k=None):
        # Build a stable item index (0..I-1)
        if all_items is None:
            item_ids = sorted(neighbors.keys())
        else:
            item_ids = sorted(all_items)
        item2idx = {it: i for i, it in enumerate(item_ids)}
        I = len(item_ids)

        # Determine K from data if not provided
        if k is None:
            k = max((len(v) for v in neighbors.values()), default=0)

        # Allocate (-1 / 0 padding)
        nbr_idx = torch.full((I, k), -1, dtype=torch.long, device=device)
        nbr_val = torch.zeros((I, k), dtype=torch.float32, device=device)

        for it, lst in neighbors.items():
            i = item2idx[it]
            # lst: list[(nbr_item, sim)] — we will keep up to k
            upto = min(k, len(lst))
            if upto == 0: 
                continue
            idxs = [item2idx[nbr] for (nbr, _) in lst[:upto] if nbr in item2idx]
            vals = [float(sim) for (nbr, sim) in lst[:upto] if nbr in item2idx]
            if not idxs:
                continue
            # Put into tensors
            t_idx = torch.tensor(idxs, dtype=torch.long, device=device)
            t_val = torch.tensor(vals, dtype=torch.float32, device=device)
            nbr_idx[i, :len(idxs)] = t_idx
            nbr_val[i, :len(vals)] = t_val

        return item_ids, item2idx, nbr_idx, nbr_val

    # ---- Wrapper that matches the CV driver's expected signature ----
    def recommend_all_gpu_wrapper(train_user_items_subset, neighbors, popular, topn=20):
        # Pack neighbors to GPU tensors
        item_ids, item2idx, nbr_idx, nbr_val = prepare_torch_neighbors(
            neighbors, all_items=None, k=None)

        # Popularity filtered to indexed items (keep order)
        popular_items = [it for it in popular if it in item2idx]

        # Call your existing GPU recommender
        recs = recommend_all_gpu(
            train_user_items=train_user_items_subset,
            item_ids=item_ids,
            item2idx=item2idx,
            nbr_idx=nbr_idx,
            nbr_val=nbr_val,
            popular_items=popular_items,
            topn=topn
        )
        return recs

    import random
    import numpy as np

    def kfold_users(user_items, n_folds=5, seed=42):
        rng = random.Random(seed)
        users = list(user_items.keys())
        rng.shuffle(users)
        fold_size = int(np.ceil(len(users) / n_folds))
        folds = [users[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]
        return [f for f in folds if f]  # drop any empty tail

    def split_user_items_for_cv(items_set, test_frac=0.2, rng=None):
        """Per-user item split: returns (train_items, test_items), ensuring at least 1 test when possible."""
        if rng is None:
            rng = random
        items = list(items_set)
        if len(items) <= 1:
            return set(items), set()
        n_test = max(1, int(round(len(items) * test_frac)))
        test = set(rng.sample(items, n_test))
        train = set(items_set) - test
        if not train:
            # move one back to train to keep both non-empty
            x = next(iter(test))
            test.remove(x)
            train.add(x)
        return train, test

    def run_nfold_cv(user_items,
                    n_folds=5,
                    k_values=(5,10,20,30,40,50,75,100),
                    sim_types=("cosine","jaccard","pearson"),
                    test_frac=0.2,
                    topn=20,
                    seed=42,
                    build_neighbors_fn=None,
                    recommend_all_fn=None,
                    average_ndcg_fn=None):
        
        folds = kfold_users(user_items, n_folds=n_folds, seed=seed)
        rng = random.Random(seed)

        per_sim = {sim: np.zeros((len(folds), len(k_values)), dtype=np.float64) for sim in sim_types}

        for f_idx, test_users in enumerate(folds):
            # Train users (all items contribute to training signal)
            train_users = [u for u in user_items if u not in test_users]
            train_ui = {u: set(user_items[u]) for u in train_users if user_items[u]}

            # For users in this fold, split their items into personal train/test
            per_user_train = dict()
            test_ui = dict()
            for u in test_users:
                tr, te = split_user_items_for_cv(user_items[u], test_frac=test_frac, rng=rng)
                if tr:
                    per_user_train[u] = tr
                if te:
                    test_ui[u] = te

            # Include the train part of test-fold users so their own history can be used
            train_ui.update(per_user_train)

            if not test_ui:
                print(f"[CV] Fold {f_idx+1}: no test ground truth—skipping.")
                continue

            for k_idx, k in enumerate(k_values):
                for sim in sim_types:
                    # 1) Build neighbors on current training interactions
                    neighbors, popular = build_neighbors_fn(train_ui, k, sim_type=sim)
                    # 2) Recommend only for the users we will evaluate (the test fold users), using their train part
                    recs = recommend_all_fn(per_user_train, neighbors, popular, topn=topn)
                    # 3) Evaluate against held-out items
                    score = average_ndcg_fn(recs, test_ui, k=topn)
                    per_sim[sim][f_idx, k_idx] = score

            print(f"[CV] Completed fold {f_idx+1}/{len(folds)}")

        # Aggregate across folds
        results = dict()
        for sim in sim_types:
            mat = per_sim[sim]  # (F, K)
            means = [float(np.mean(mat[:, k_idx])) for k_idx in range(len(k_values))]
            results[sim] = {
                "k_values": list(k_values),
                "mean": means,
                "per_fold": mat
            }
        return results

    results = run_leave_one_out(
        user_items=user_items,
        k_values = [5,10,20,30,40,50,75,100],
        sim_types = ("cosine", "jaccard", "pearson"),
        topn=20,
        seed=42,
        repeats=3,
        build_neighbors_fn=build_neighbors,
        recommend_all_fn=recommend_all_gpu_wrapper
    )
        
    print("Cosine HR@20:",  [f"{v:.4f}" for v in results["cosine"]["HR"]])
    print("Cosine MRR@20:", [f"{v:.4f}" for v in results["cosine"]["MRR"]])
    print("Jaccard HR@20:",  [f"{v:.4f}" for v in results["jaccard"]["HR"]])
    print("Jaccard MRR@20:", [f"{v:.4f}" for v in results["jaccard"]["MRR"]])
    print("Pearson HR@20:",  [f"{v:.4f}" for v in results["pearson"]["HR"]])
    print("Pearson MRR@20:", [f"{v:.4f}" for v in results["pearson"]["MRR"]])

    K_VALUES = [5, 10, 20, 30, 40, 50, 75, 100]

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(K_VALUES, results["cosine"]["HR"],  marker="o", color="green", label="Cosine HR")
    ax.plot(K_VALUES, results["jaccard"]["HR"], marker="s", color="red", label="Jaccard HR")
    ax.plot(K_VALUES, results["pearson"]["HR"], marker="+", color="orange", label="Pearson HR")
    ax.plot(K_VALUES, results["cosine"]["MRR"],  marker="o", color="brown", label="Cosine MRR")
    ax.plot(K_VALUES, results["jaccard"]["MRR"], marker="s", color="pink", label="Jaccard MRR")
    ax.plot(K_VALUES, results["pearson"]["MRR"], marker="+", color="blue", label="Pearson MRR")
    ax.set_xlabel("Neighbor list size (k)")
    ax.set_ylabel("Leave one out")
    ax.set_title("LOO vs. k-size")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig("item-based_knn_entire_program_v3.jpg")
    
def main():
    # ---- training ----
    cosine_similarity()
    jaccard_similarity()
    pearson_similarity()
    # ---- evaluation ----
    loo_evaluation()

if __name__ == "__main__":
    main()