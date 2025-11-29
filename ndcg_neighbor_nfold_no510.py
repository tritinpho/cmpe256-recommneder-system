import collections
import heapq
import math
import random
import re
from typing import Dict, Set, List, Tuple

import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================

INPUT_PATH = "train-1.txt"   # original data file
TRAIN_RATIO = 0.8            # per-user train ratio
TOP_N = 20                   # recommendation list length
USER_NORMALIZE = True
RANDOM_SEED = 42

# neighbor sizes for x-axis
NEIGHBOR_SIZES = [20, 50, 100, 150, 200, 250, 300]

# fixed hyperparameters
MIN_INTER = 3
ALPHA = 20.0

# we will compare these CV settings
CV_FOLDS_LIST = [5, 10]

random.seed(RANDOM_SEED)

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

# ============================================================
# 3. Precompute stats from TRAIN: item_users, item_deg, popular, co_counts
# ============================================================

def build_stats(
    train_user_items: Dict[str, Set[str]]
) -> Tuple[Dict[str, Set[str]], Dict[str, int], List[str], collections.Counter]:
    item_users: Dict[str, Set[str]] = collections.defaultdict(set)
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

    return item_users, item_deg, popular_items, co_counts

# ============================================================
# 4. Evaluate a given MAX_K on a given split
# ============================================================

def eval_for_max_k(
    MAX_K: int,
    all_user_items: Dict[str, List[str]],
    train_user_items: Dict[str, Set[str]],
    val_user_items: Dict[str, Set[str]],
    item_deg: Dict[str, int],
    popular_items: List[str],
    co_counts: collections.Counter,
) -> float:
    # ---------- build neighbors for this MAX_K ----------
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

    # ---------- recommend for all users ----------
    all_users_sorted = sorted(all_user_items.keys())
    user_recs: Dict[str, List[str]] = {}

    for u in all_users_sorted:
        seen = train_user_items.get(u, set())
        scores = collections.defaultdict(float)

        if seen:
            if USER_NORMALIZE:
                user_weight = 1.0 / len(seen)
            else:
                user_weight = 1.0

            for it in seen:
                for sim, nb in neighbors.get(it, []):
                    if nb in seen:
                        continue
                    scores[nb] += sim * user_weight

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

    # ---------- validation NDCG ----------
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
# 4b. Function to generate and save recommendations to a file
# ============================================================

def generate_and_save_recommendations(
    MAX_K: int,
    all_user_items: Dict[str, List[str]],
    train_user_items: Dict[str, Set[str]],
    item_deg: Dict[str, int],
    popular_items: List[str],
    co_counts: collections.Counter,
    output_path: str,
):
    """
    Build neighbors for the given MAX_K using Jaccard + shrinkage,
    recommend top-20 items for every user, and save to a text file.

    Format: each line has 20 item IDs separated by spaces (no user ID).
    Users are ordered by numeric ID if possible, otherwise lexicographically.
    """
    # ---------- build neighbors ----------
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

    # ---------- recommend for all users ----------
    all_users_sorted = sorted(all_user_items.keys(), key=lambda x: (int(x) if x.isdigit() else x))
    user_recs: Dict[str, List[str]] = {}

    for u in all_users_sorted:
        seen = train_user_items.get(u, set())
        scores = collections.defaultdict(float)

        if seen:
            if USER_NORMALIZE:
                user_weight = 1.0 / len(seen)
            else:
                user_weight = 1.0

            for it in seen:
                for sim, nb in neighbors.get(it, []):
                    if nb in seen:
                        continue
                    scores[nb] += sim * user_weight

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

    # ---------- save to file ----------
    with open(output_path, "w", encoding="utf-8") as f:
        for u in all_users_sorted:
            recs = user_recs[u]
            f.write(" ".join(recs) + "\n")

    print(f"\nSaved top-{TOP_N} recommendations for {len(all_users_sorted)} users to {output_path}")


# ============================================================
# 5. Single split (no CV) curve
# ============================================================

print("\n=== Single split (no CV) ===")
train_user_items_single, val_user_items_single = split_per_user(
    all_user_items, TRAIN_RATIO, seed=RANDOM_SEED
)
item_users_s, item_deg_s, popular_items_s, co_counts_s = build_stats(train_user_items_single)

ndcg_no_cv = []
for K in NEIGHBOR_SIZES:
    ndcg = eval_for_max_k(
        K,
        all_user_items,
        train_user_items_single,
        val_user_items_single,
        item_deg_s,
        popular_items_s,
        co_counts_s,
    )
    ndcg_no_cv.append(ndcg)
    print(f"MAX_K={K} → [no CV] Val NDCG@20 = {ndcg:.6f}")

# ============================================================
# 6. n-fold cross-validation curves for 5-fold and 10-fold
# ============================================================

cv_results = {}  # folds -> list of ndcg for each K

for n_folds in CV_FOLDS_LIST:
    print(f"\n=== {n_folds}-fold cross-validation ===")
    ndcg_cv_sum = [0.0 for _ in NEIGHBOR_SIZES]

    for fold in range(n_folds):
        seed = RANDOM_SEED + (n_folds * 100) + fold + 1  # different seeds per scheme
        train_user_items_f, val_user_items_f = split_per_user(
            all_user_items, TRAIN_RATIO, seed=seed
        )
        item_users_f, item_deg_f, popular_items_f, co_counts_f = build_stats(train_user_items_f)

        print(f"\nFold {fold+1}/{n_folds}, seed={seed}")
        for idx, K in enumerate(NEIGHBOR_SIZES):
            ndcg = eval_for_max_k(
                K,
                all_user_items,
                train_user_items_f,
                val_user_items_f,
                item_deg_f,
                popular_items_f,
                co_counts_f,
            )
            ndcg_cv_sum[idx] += ndcg
            print(f"MAX_K={K} → fold {fold+1} Val NDCG@20 = {ndcg:.6f}")

    ndcg_cv_avg = [v / n_folds for v in ndcg_cv_sum]
    cv_results[n_folds] = ndcg_cv_avg

    print(f"\n=== {n_folds}-fold CV-averaged NDCG@20 ===")
    for K, ndcg in zip(NEIGHBOR_SIZES, ndcg_cv_avg):
        print(f"MAX_K={K} → [{n_folds}-fold CV avg] Val NDCG@20 = {ndcg:.6f}")

# ============================================================
# 7. Plot comparison: no CV vs 5-fold vs 10-fold
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(NEIGHBOR_SIZES, ndcg_no_cv, marker='o', label="Single split (no CV)")

if 5 in cv_results:
    plt.plot(NEIGHBOR_SIZES, cv_results[5], marker='s', linestyle='--', label="5-fold CV (avg)")
if 10 in cv_results:
    plt.plot(NEIGHBOR_SIZES, cv_results[10], marker='^', linestyle='-.', label="10-fold CV (avg)")

plt.title("Item-kNN: Neighbor Size vs NDCG@20\nNo CV vs 5-fold CV vs 10-fold CV")
plt.xlabel("Number of neighbors (MAX_K)")
plt.ylabel("Validation NDCG@20")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("neighbors_vs_ndcg20_cv_5_10.png")
plt.show()

print("\nSaved plot as neighbors_vs_ndcg20_cv_5_10.png")

# ============================================================
# 8. Use best MAX_K (e.g., from 10-fold CV if available) to
#    generate and save 20 recommendations per user
# ============================================================

# choose best MAX_K from one of the curves (here: 10-fold CV if present, else 5-fold, else no-CV)
def choose_best_max_k():
    if 10 in cv_results:
        ndcgs = cv_results[10]
        label = "10-fold CV"
    elif 5 in cv_results:
        ndcgs = cv_results[5]
        label = "5-fold CV"
    else:
        ndcgs = ndcg_no_cv
        label = "no CV"

    best_idx = max(range(len(NEIGHBOR_SIZES)), key=lambda i: ndcgs[i])
    best_k = NEIGHBOR_SIZES[best_idx]
    best_ndcg = ndcgs[best_idx]
    print(f"\nBest MAX_K based on {label}: MAX_K={best_k}, Val NDCG@20={best_ndcg:.6f}")
    return best_k

best_max_k = choose_best_max_k()

# For the final model, we can reuse the single-split train/val stats
# (or you could rebuild stats with a fresh seed if you prefer)
generate_and_save_recommendations(
    MAX_K=best_max_k,
    all_user_items=all_user_items,
    train_user_items=train_user_items_single,
    item_deg=item_deg_s,
    popular_items=popular_items_s,
    co_counts=co_counts_s,
    output_path="recommendations_item_knn_best.txt",
)