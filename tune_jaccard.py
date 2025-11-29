import collections
import heapq
import math
import random
import re

# ===================== GLOBAL SETTINGS =====================

INPUT_PATH = "train-1.txt"
TOP_N = 20
TRAIN_RATIO = 0.8
USER_NORMALIZE = True
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Grids to try
GRID_MAX_K = [100, 200, 300]
GRID_MIN_INTER = [1, 2, 3]
GRID_ALPHA = [0.0, 5.0, 10.0, 20.0]

# ===================== METRIC HELPERS ======================

def hit_at_k(preds, gt_set, k=20):
    return 1.0 if any(item in gt_set for item in preds[:k]) else 0.0

def ndcg_at_k(preds, gt_set, k=20):
    dcg = 0.0
    for rank, it in enumerate(preds[:k]):
        if it in gt_set:
            dcg += 1.0 / math.log2(rank + 2)
    ideal = min(len(gt_set), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0

# ===================== 1. LOAD ORIGINAL DATA ===============

all_user_items = collections.defaultdict(list)

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        nums = re.findall(r"-?\d+", line)
        if len(nums) < 2:
            continue
        u = nums[0]
        items = nums[1:]
        all_user_items[u].extend(items)

print(f"Loaded users from original file: {len(all_user_items)}")

# ===================== 2. PER-USER TRAIN/VAL SPLIT =========

train_user_items = collections.defaultdict(set)
val_user_items = collections.defaultdict(set)

for u, items in all_user_items.items():
    uniq_items = list(set(items))
    if len(uniq_items) == 1:
        train_user_items[u].add(uniq_items[0])
        continue

    items_copy = uniq_items[:]
    random.shuffle(items_copy)

    k_val = max(1, int((1.0 - TRAIN_RATIO) * len(items_copy)))
    val_items = items_copy[:k_val]
    train_items = items_copy[k_val:]

    for it in train_items:
        train_user_items[u].add(it)
    for it in val_items:
        val_user_items[u].add(it)

print(f"Train users: {len(train_user_items)}, Val users: {len(val_user_items)}")

# ===================== 3. PRECOMPUTE ITEM STATS ============

item_users = collections.defaultdict(set)
for u, items in train_user_items.items():
    for it in items:
        item_users[it].add(u)

print(f"Num items in TRAIN: {len(item_users)}")

item_deg = {it: len(us) for it, us in item_users.items()}
popular = sorted(item_deg.items(), key=lambda kv: (-kv[1], kv[0]))
popular_items = [it for it, _ in popular]

# co-occurrence counts; independent of hyperparams
co_counts = collections.Counter()
for u, items in train_user_items.items():
    if len(items) < 2:
        continue
    sitems = sorted(items)
    for i_idx in range(len(sitems)):
        i = sitems[i_idx]
        for j in sitems[i_idx + 1:]:
            co_counts[(i, j)] += 1

print(f"Co-occurrence pairs: {len(co_counts)}")

all_users = sorted(all_user_items.keys())

# ===================== 4. CORE RUN FUNCTION =================

def run_model(MAX_K, MIN_INTER, ALPHA):
    # ---- build neighbors for this hyperparam setting ----
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

        sim = inter / (union + ALPHA) if ALPHA > 0 else inter / union
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

    # ---- make recommendations for all users ----
    user_recs = {}

    for u in all_users:
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

    # ---- training LOO ----
    train_hit_sum = 0.0
    train_ndcg_sum = 0.0
    train_users_eval = 0

    for u, items in train_user_items.items():
        if len(items) < 2:
            continue
        items_list = list(items)
        heldout = random.choice(items_list)
        seen = set(items_list)
        seen.remove(heldout)

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
            continue

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

        gt_set = {heldout}
        h = hit_at_k(recs, gt_set, k=TOP_N)
        n = ndcg_at_k(recs, gt_set, k=TOP_N)

        train_hit_sum += h
        train_ndcg_sum += n
        train_users_eval += 1

    train_hit = train_hit_sum / train_users_eval if train_users_eval > 0 else 0.0
    train_ndcg = train_ndcg_sum / train_users_eval if train_users_eval > 0 else 0.0

    # ---- validation ----
    val_hit_sum = 0.0
    val_ndcg_sum = 0.0
    val_users_eval = 0

    for u, gt_items in val_user_items.items():
        if len(gt_items) == 0:
            continue
        recs = user_recs.get(u)
        if recs is None:
            continue

        h = hit_at_k(recs, gt_items, k=TOP_N)
        n = ndcg_at_k(recs, gt_items, k=TOP_N)

        val_hit_sum += h
        val_ndcg_sum += n
        val_users_eval += 1

    val_hit = val_hit_sum / val_users_eval if val_users_eval > 0 else 0.0
    val_ndcg = val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

    print(f"MAX_K={MAX_K:3d}, MIN_INTER={MIN_INTER}, ALPHA={ALPHA:4.1f} "
          f"| TrainNDCG@20={train_ndcg:.4f}, ValNDCG@20={val_ndcg:.4f}, ValHit@20={val_hit:.4f}")

    return train_ndcg, val_ndcg, train_hit, val_hit

# ===================== 5. GRID SEARCH ======================

best_cfg = None
best_val_ndcg = -1.0

for max_k in GRID_MAX_K:
    for min_inter in GRID_MIN_INTER:
        for alpha in GRID_ALPHA:
            tn, vn, th, vh = run_model(max_k, min_inter, alpha)
            if vn > best_val_ndcg:
                best_val_ndcg = vn
                best_cfg = (max_k, min_inter, alpha, tn, th)

print("\nBest config by ValNDCG@20:")
print(f"MAX_K={best_cfg[0]}, MIN_INTER={best_cfg[1]}, ALPHA={best_cfg[2]} "
      f"| TrainNDCG@20={best_cfg[3]:.4f}, ValNDCG@20={best_val_ndcg:.4f}")