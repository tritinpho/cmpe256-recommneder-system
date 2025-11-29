import collections
import heapq
import math
import random
import re
from typing import Dict, Set, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ============================================================
# Config
# ============================================================

INPUT_PATH = "train-1.txt"   # original data file
TRAIN_RATIO = 0.8            # per-user train ratio
TOP_N = 20                   # recommendation list length
USER_NORMALIZE = True
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Neighbor sizes for x-axis (you can change these)
NEIGHBOR_SIZES = [20, 50, 100, 150, 200, 250, 300]

# Jaccard hyperparams
MIN_INTER = 3
ALPHA = 20.0  # shrinkage: sim = inter / (union + ALPHA)

# SVD hyperparam (fixed embedding dim; we only sweep neighbors)
EMB_DIM = 64

# ============================================================
# Metric helpers
# ============================================================

def hit_at_k(preds: List[str], gt_set: Set[str], k: int = 20) -> float:
    return 1.0 if any(item in gt_set for item in preds[:k]) else 0.0

def ndcg_at_k(preds: List[str], gt_set: Set[str], k: int = 20) -> float:
    dcg = 0.0
    for rank, it in enumerate(preds[:k]):
        if it in gt_set:
            dcg += 1.0 / math.log2(rank + 2)
    ideal = min(len(gt_set), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0

# ============================================================
# 1. Load ALL items per user from train-1.txt
# ============================================================

def load_all_user_items(path: str) -> Dict[str, List[str]]:
    all_user_items = collections.defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            nums = re.findall(r"-?\d+", line)
            if len(nums) < 2:
                continue
            u = nums[0]
            items = nums[1:]
            all_user_items[u].extend(items)
    return all_user_items

all_user_items = load_all_user_items(INPUT_PATH)
print(f"Loaded users from original file: {len(all_user_items)}")

# ============================================================
# 2. Per-user train/val split (same users in both)
# ============================================================

def split_per_user(
    all_user_items: Dict[str, List[str]],
    train_ratio: float,
    seed: int
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    random.seed(seed)
    train_user_items: Dict[str, Set[str]] = collections.defaultdict(set)
    val_user_items: Dict[str, Set[str]] = collections.defaultdict(set)

    for u, items in all_user_items.items():
        uniq_items = list(set(items))
        if len(uniq_items) == 1:
            train_user_items[u].add(uniq_items[0])
            continue

        items_copy = uniq_items[:]
        random.shuffle(items_copy)

        k_val = max(1, int((1.0 - train_ratio) * len(items_copy)))
        val_items = items_copy[:k_val]
        train_items = items_copy[k_val:]

        for it in train_items:
            train_user_items[u].add(it)
        for it in val_items:
            val_user_items[u].add(it)

    return train_user_items, val_user_items

train_user_items, val_user_items = split_per_user(
    all_user_items, TRAIN_RATIO, seed=RANDOM_SEED
)
print(f"Train users: {len(train_user_items)}, Val users: {len(val_user_items)}")

all_users_sorted = sorted(all_user_items.keys())

# ============================================================
# 3. Precompute for Jaccard: item_users, item_deg, popularity, co_counts
# ============================================================

item_users = collections.defaultdict(set)
for u, items in train_user_items.items():
    for it in items:
        item_users[it].add(u)

item_deg = {it: len(us) for it, us in item_users.items()}
popular = sorted(item_deg.items(), key=lambda kv: (-kv[1], kv[0]))
popular_items = [it for it, _ in popular]

co_counts = collections.Counter()
for u, items in train_user_items.items():
    if len(items) < 2:
        continue
    sitems = sorted(items)
    for i_idx in range(len(sitems)):
        i = sitems[i_idx]
        for j in sitems[i_idx + 1:]:
            co_counts[(i, j)] += 1

print(f"Num items in TRAIN (Jaccard): {len(item_users)}")
print(f"Co-occurrence pairs: {len(co_counts)}")

# ============================================================
# 4. Jaccard: build neighbors and evaluate for a given MAX_K
# ============================================================

def jaccard_eval_for_max_k(MAX_K: int) -> float:
    # --- build neighbors ---
    heaps = collections.defaultdict(list)

    for (i, j), inter in co_counts.items():
        if inter < MIN_INTER:
            continue
        ui = item_deg.get(i, 0)
        uj = item_deg.get(j, 0)
        if ui == 0 or uj == 0:
            continue
        union = ui + uj - inter
        if union <= 0:
            continue

        sim = inter / (union + ALPHA)
        if sim <= 0:
            continue

        heapq.heappush(heaps[i], (sim, j))
        if len(heaps[i]) > MAX_K:
            heapq.heappop(heaps[i])

        heapq.heappush(heaps[j], (sim, i))
        if len(heaps[j]) > MAX_K:
            heapq.heappop(heaps[j])

    neighbors = {}
    for it, h in heaps.items():
        lst = [heapq.heappop(h) for _ in range(len(h))]
        lst.sort(key=lambda t: (-t[0], t[1]))
        neighbors[it] = lst

    # --- recommend for all users ---
    user_recs = {}

    for u in all_users_sorted:
        seen = train_user_items.get(u, set())
        scores = collections.defaultdict(float)

        if seen:
            if USER_NORMALIZE:
                w = 1.0 / len(seen)
            else:
                w = 1.0
            for it in seen:
                for sim, nb in neighbors.get(it, []):
                    if nb in seen:
                        continue
                    scores[nb] += sim * w

        if not scores:
            recs = [it for it in popular_items if it not in seen][:TOP_N]
        else:
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            recs = [it for it, _ in ranked[:TOP_N]]
            if len(recs) < TOP_N:
                for it in popular_items:
                    if len(recs) >= TOP_N:
                        break
                    if it not in seen and it not in scores:
                        recs.append(it)

        if len(recs) > TOP_N:
            recs = recs[:TOP_N]

        user_recs[u] = recs

    # --- validation NDCG ---
    val_ndcg_sum = 0.0
    val_users_eval = 0

    for u, gt_items in val_user_items.items():
        if len(gt_items) == 0:
            continue
        recs = user_recs.get(u)
        if recs is None:
            continue

        n = ndcg_at_k(recs, gt_items, k=TOP_N)
        val_ndcg_sum += n
        val_users_eval += 1

    return val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

# ============================================================
# 5. Build user–item matrix for SVD (TRAIN only)
# ============================================================

def build_user_item_matrix(
    user_items: Dict[str, Set[str]]
) -> Tuple[sparse.csr_matrix, List[str], List[str]]:
    users = sorted(user_items.keys())
    user2idx = {u: idx for idx, u in enumerate(users)}

    items_set = {it for its in user_items.values() for it in its}
    items = sorted(items_set)
    item2idx = {it: idx for idx, it in enumerate(items)}

    rows, cols, data = [], [], []

    for u, its in user_items.items():
        ui = user2idx[u]
        for it in its:
            ii = item2idx[it]
            rows.append(ui)
            cols.append(ii)
            data.append(1.0)

    X = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    return X, users, items

X, train_users_order, items_list = build_user_item_matrix(train_user_items)
print("User–item matrix (TRAIN) shape:", X.shape)

# Compute SVD item embeddings once
svd = TruncatedSVD(n_components=EMB_DIM, random_state=RANDOM_SEED)
item_emb = svd.fit_transform(X.T)   # [num_items, EMB_DIM]
item_emb = normalize(item_emb, norm="l2", axis=1)
num_items = item_emb.shape[0]
idx2item = {idx: it for idx, it in enumerate(items_list)}

# Popularity for SVD model (can reuse same popularity idea)
item_deg_svd = collections.Counter()
for u, items in train_user_items.items():
    for it in items:
        item_deg_svd[it] += 1
popular_svd = sorted(item_deg_svd.items(), key=lambda kv: (-kv[1], kv[0]))
popular_items_svd = [it for it, _ in popular_svd]

# ============================================================
# 6. SVD+cosine: build neighbors and evaluate for a given MAX_K
# ============================================================

def svd_eval_for_max_k(MAX_K: int) -> float:
    # --- build neighbors from embeddings ---
    neighbors = {it: [] for it in items_list}

    sims = item_emb @ item_emb.T  # cosine, since emb is L2-normalized

    for i in range(num_items):
        row = sims[i].copy()
        row[i] = -1.0  # exclude self
        if MAX_K < num_items:
            topk_idx = np.argpartition(-row, MAX_K)[:MAX_K]
        else:
            topk_idx = np.arange(num_items)
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]
        it = idx2item[i]
        neighbors[it] = [(float(row[j]), idx2item[j]) for j in topk_idx]

    # --- recommend for all users ---
    user_recs = {}

    for u in all_users_sorted:
        seen = train_user_items.get(u, set())
        scores = collections.defaultdict(float)

        if seen:
            if USER_NORMALIZE:
                w = 1.0 / len(seen)
            else:
                w = 1.0
            for it in seen:
                for sim, nb in neighbors.get(it, []):
                    if nb in seen:
                        continue
                    scores[nb] += sim * w

        if not scores:
            recs = [it for it in popular_items_svd if it not in seen][:TOP_N]
        else:
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            recs = [it for it, _ in ranked[:TOP_N]]
            if len(recs) < TOP_N:
                for it in popular_items_svd:
                    if len(recs) >= TOP_N:
                        break
                    if it not in seen and it not in scores:
                        recs.append(it)

        if len(recs) > TOP_N:
            recs = recs[:TOP_N]

        user_recs[u] = recs

    # --- validation NDCG ---
    val_ndcg_sum = 0.0
    val_users_eval = 0

    for u, gt_items in val_user_items.items():
        if len(gt_items) == 0:
            continue
        recs = user_recs.get(u)
        if recs is None:
            continue

        n = ndcg_at_k(recs, gt_items, k=TOP_N)
        val_ndcg_sum += n
        val_users_eval += 1

    return val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

# ============================================================
# 7. Functions to generate & save recommendations as text files
# ============================================================

def generate_and_save_recommendations_jaccard(MAX_K: int, output_path: str) -> None:
    """
    Build Jaccard neighbors for the given MAX_K and save top-20
    recommendations for every user to a text file.

    Each line: 20 item IDs (no user ID).
    Users are sorted by numeric ID when possible.
    """
    # --- build neighbors (same as in jaccard_eval_for_max_k) ---
    heaps = collections.defaultdict(list)

    for (i, j), inter in co_counts.items():
        if inter < MIN_INTER:
            continue
        ui = item_deg.get(i, 0)
        uj = item_deg.get(j, 0)
        if ui == 0 or uj == 0:
            continue
        union = ui + uj - inter
        if union <= 0:
            continue

        sim = inter / (union + ALPHA)
        if sim <= 0:
            continue

        heapq.heappush(heaps[i], (sim, j))
        if len(heaps[i]) > MAX_K:
            heapq.heappop(heaps[i])

        heapq.heappush(heaps[j], (sim, i))
        if len(heaps[j]) > MAX_K:
            heapq.heappop(heaps[j])

    neighbors = {}
    for it, h in heaps.items():
        lst = [heapq.heappop(h) for _ in range(len(h))]
        lst.sort(key=lambda t: (-t[0], t[1]))
        neighbors[it] = lst

    # --- recommend for all users ---
    def user_sort_key(u: str):
        return int(u) if u.isdigit() else u

    users_sorted_for_file = sorted(all_users_sorted, key=user_sort_key)

    with open(output_path, "w", encoding="utf-8") as f:
        for u in users_sorted_for_file:
            seen = train_user_items.get(u, set())
            scores = collections.defaultdict(float)

            if seen:
                if USER_NORMALIZE:
                    w = 1.0 / len(seen)
                else:
                    w = 1.0
                for it in seen:
                    for sim, nb in neighbors.get(it, []):
                        if nb in seen:
                            continue
                        scores[nb] += sim * w

            if not scores:
                recs = [it for it in popular_items if it not in seen][:TOP_N]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:TOP_N]]
                if len(recs) < TOP_N:
                    for it in popular_items:
                        if len(recs) >= TOP_N:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)

            if len(recs) > TOP_N:
                recs = recs[:TOP_N]

            f.write(" ".join(recs) + "\n")

    print(f"\n[Jaccard] Saved top-{TOP_N} recommendations to {output_path} (MAX_K={MAX_K})")


def generate_and_save_recommendations_svd(MAX_K: int, output_path: str) -> None:
    """
    Build SVD+cosine neighbors for the given MAX_K and save top-20
    recommendations for every user to a text file.

    Each line: 20 item IDs (no user ID).
    Users are sorted by numeric ID when possible.
    """
    # --- build neighbors from embeddings (same pattern as svd_eval_for_max_k) ---
    neighbors = {it: [] for it in items_list}

    sims = item_emb @ item_emb.T  # cosine

    for i in range(num_items):
        row = sims[i].copy()
        row[i] = -1.0
        if MAX_K < num_items:
            topk_idx = np.argpartition(-row, MAX_K)[:MAX_K]
        else:
            topk_idx = np.arange(num_items)
        topk_idx = topk_idx[np.argsort(-row[topk_idx])]
        it = idx2item[i]
        neighbors[it] = [(float(row[j]), idx2item[j]) for j in topk_idx]

    def user_sort_key(u: str):
        return int(u) if u.isdigit() else u

    users_sorted_for_file = sorted(all_users_sorted, key=user_sort_key)

    with open(output_path, "w", encoding="utf-8") as f:
        for u in users_sorted_for_file:
            seen = train_user_items.get(u, set())
            scores = collections.defaultdict(float)

            if seen:
                if USER_NORMALIZE:
                    w = 1.0 / len(seen)
                else:
                    w = 1.0
                for it in seen:
                    for sim, nb in neighbors.get(it, []):
                        if nb in seen:
                            continue
                        scores[nb] += sim * w

            if not scores:
                recs = [it for it in popular_items_svd if it not in seen][:TOP_N]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:TOP_N]]
                if len(recs) < TOP_N:
                    for it in popular_items_svd:
                        if len(recs) >= TOP_N:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)

            if len(recs) > TOP_N:
                recs = recs[:TOP_N]

            f.write(" ".join(recs) + "\n")

    print(f"\n[SVD] Saved top-{TOP_N} recommendations to {output_path} (MAX_K={MAX_K}, EMB_DIM={EMB_DIM})")


# ============================================================
# 8. Sweep neighbor sizes and collect NDCG@20
# ============================================================

ndcg_jaccard = []
ndcg_svd = []

print("\nRunning neighbor-size sweep (Jaccard vs SVD)...\n")

for K in NEIGHBOR_SIZES:
    ndcg_j = jaccard_eval_for_max_k(K)
    ndcg_s = svd_eval_for_max_k(K)
    ndcg_jaccard.append(ndcg_j)
    ndcg_svd.append(ndcg_s)
    print(f"MAX_K={K:3d} → Jaccard NDCG@20={ndcg_j:.6f}, SVD NDCG@20={ndcg_s:.6f}")

# ============================================================
# 9. Plot Jaccard vs SVD: Neighbor Size vs NDCG@20
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(NEIGHBOR_SIZES, ndcg_jaccard, marker='o', label="Jaccard item-kNN")
plt.plot(NEIGHBOR_SIZES, ndcg_svd, marker='s', linestyle='--', label=f"SVD+cosine item-kNN (dim={EMB_DIM})")

plt.title("Item-kNN: Neighbor Size vs NDCG@20\nJaccard vs SVD+Cosine")
plt.xlabel("Number of neighbors (MAX_K)")
plt.ylabel("Validation NDCG@20")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("neighbors_vs_ndcg20_jaccard_vs_svd.png")
plt.show()

print("\nSaved plot as neighbors_vs_ndcg20_jaccard_vs_svd.png")

# ============================================================
# 10. Example: choose best MAX_K and save rec files
# ============================================================

# Best Jaccard K
best_j_idx = max(range(len(NEIGHBOR_SIZES)), key=lambda i: ndcg_jaccard[i])
best_j_k = NEIGHBOR_SIZES[best_j_idx]
print(f"\nBest Jaccard MAX_K={best_j_k} with Val NDCG@20={ndcg_jaccard[best_j_idx]:.6f}")

# Best SVD K
best_s_idx = max(range(len(NEIGHBOR_SIZES)), key=lambda i: ndcg_svd[i])
best_s_k = NEIGHBOR_SIZES[best_s_idx]
print(f"Best SVD MAX_K={best_s_k} with Val NDCG@20={ndcg_svd[best_s_idx]:.6f}")

# Generate files (you can comment out one if you only need one model)
generate_and_save_recommendations_jaccard(
    MAX_K=best_j_k,
    output_path="recommendations_jaccard_best.txt",
)

generate_and_save_recommendations_svd(
    MAX_K=best_s_k,
    output_path="recommendations_svd_best.txt",
)
