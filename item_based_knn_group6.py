import collections
import heapq
import math
import random
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# All of the experiments were performed at train/test = 8/2.
# The number of recommended items is always 20.
USER_NORMALIZE = True
RANDOM_SEED = 42

# A tried neighbor list is the same like this for all of the experiments.
NEIGHBOR_SIZES = [20, 50, 100, 150, 200, 250, 300]

# Fixed hyperparameters
MIN_INTER = 3 # The lowest allowed co-occurrences for items between users. If co-occurrences are less than three, discard the item.
ALPHA = 20.0 # Jaccard shrinkage, make Jaccard similarity more reliable.

# Top 20 items are recommended using no CV, 5 fold CV, and 10 fold CV and evaluated using NDCG@20 with number of neighbors increases.
def n_fold_variation():
    random.seed(42)

    # Helper function: NDCG@20 value is obtained at the size of neighbor k.
    def ndcg_at_k(preds, gt_set, k = 20):
        dcg = 0.0
        for rank, it in enumerate(preds[:k]):
            if it in gt_set:
                dcg += 1.0 / math.log2(rank + 2)
        ideal = min(len(gt_set), k)
        if ideal == 0:
            return 0.0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    # 1. Load ALL items per user from train-1.txt
    def load_user_items(path):
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

    all_user_items = load_user_items("train-1.txt")
    print(f"Loaded users from original file: {len(all_user_items)}")

    # 2. train/val split is performed per user
    def split_per_user(all_user_items, seed):
        random.seed(seed)
        train_user_items = collections.defaultdict(set)
        val_user_items = collections.defaultdict(set)

        for u, items in all_user_items.items():
            uniq_items = list(set(items))
            if len(uniq_items) == 1:
                train_user_items[u].add(uniq_items[0])
                continue

            items_copy = uniq_items[:]
            random.shuffle(items_copy)

            k_val = max(1, int(0.2 * len(items_copy)))
            val_items = items_copy[:k_val]
            train_items = items_copy[k_val:]

            for it in train_items:
                train_user_items[u].add(it)
            for it in val_items:
                val_user_items[u].add(it)

        return train_user_items, val_user_items

    # 3. Precompute stats like item_users, item_deg, popular, and co_counts from train
    def stats_build(train_user_items):
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

        return item_users, item_deg, popular_items, co_counts

    # 4a. Evaluate a given MAX_K on a given split
    def evaluation_for_max_k(MAX_K, all_user_items, train_user_items, val_user_items, item_deg, popular_items, co_counts):
        # build neighbors for this MAX_K
        heaps = collections.defaultdict(list)

        for (i, j), inter in co_counts.items():
            if inter < 3:
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

        neighbors = dict()
        for it, h in heaps.items():
            lst = [heapq.heappop(h) for _ in range(len(h))]
            lst.sort(key=lambda t: (-t[0], t[1]))
            neighbors[it] = lst

        # recommend for entire users
        all_users_sorted = sorted(all_user_items.keys())
        user_recs = dict()

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
                recs = [it for it in popular_items if it not in seen][:20]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:20]]
                if len(recs) < 20:
                    for it in popular_items:
                        if len(recs) >= 20:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)

            if len(recs) > 20:
                recs = recs[:20]

            user_recs[u] = recs

        # validation using NDCG@20
        val_ndcg_sum = 0.0
        val_users_eval = 0

        for u, gt_items in val_user_items.items():
            if len(gt_items) == 0:
                continue
            recs = user_recs.get(u)
            if recs is None:
                continue

            n = ndcg_at_k(recs, gt_items, k=20)
            val_ndcg_sum += n
            val_users_eval += 1

        return val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

    # 4b. generate and save 20 recommendations to a file
    def generate_and_save_rec(MAX_K, all_user_items, train_user_items, item_deg, popular_items, co_counts, output_path):
        # build neighbors
        heaps = collections.defaultdict(list)

        for (i, j), inter in co_counts.items():
            if inter < 3:
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

        neighbors = dict()
        for it, h in heaps.items():
            lst = [heapq.heappop(h) for _ in range(len(h))]
            lst.sort(key=lambda t: (-t[0], t[1]))
            neighbors[it] = lst

        # recommend for entire users
        all_users_sorted = sorted(all_user_items.keys(), key=lambda x: (int(x) if x.isdigit() else x))
        user_recs = dict()
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
                recs = [it for it in popular_items if it not in seen][:20]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:20]]
                if len(recs) < 20:
                    for it in popular_items:
                        if len(recs) >= 20:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)
            if len(recs) > 20:
                recs = recs[:20]
            user_recs[u] = recs

        # save to file
        with open(output_path, "w", encoding="utf-8") as f:
            for u in all_users_sorted:
                recs = user_recs[u]
                f.write(" ".join(recs) + "\n")

        print(f"\nSaved top-20 recommendations for {len(all_users_sorted)} users to {output_path}")

    # 5. NDCG@20 curve is drawn at no CV.
    print("\n=== Single split ===")
    train_user_items_single, val_user_items_single = split_per_user(
        all_user_items, seed=42
    )
    _, item_deg_s, popular_items_s, co_counts_s = stats_build(train_user_items_single)

    ndcg_no_cv = list()
    for K in [20, 50, 100, 150, 200, 250, 300]:
        ndcg = evaluation_for_max_k(
            K, all_user_items, train_user_items_single, val_user_items_single, item_deg_s, popular_items_s, co_counts_s)
        ndcg_no_cv.append(ndcg)
        print(f"MAX_K={K} → [no CV] Val NDCG@20 = {ndcg:.4f}")

    # 6. NDCG@20 curves are drawn at 5 and 10 folds CV.
    cv_results = dict()
    for n_folds in [5, 10]:
        print(f"\n=== {n_folds}-fold cross-validation ===")
        ndcg_cv_sum = [0.0 for _ in [20, 50, 100, 150, 200, 250, 300]]

        for fold in range(n_folds):
            seed = 42 + (n_folds * 100) + fold + 1  # different seeds per scheme
            train_user_items_f, val_user_items_f = split_per_user(
                all_user_items, seed=seed
            )
            _, item_deg_f, popular_items_f, co_counts_f = stats_build(train_user_items_f)

            print(f"\nFold {fold+1}/{n_folds}, seed={seed}")
            for idx, K in enumerate([20, 50, 100, 150, 200, 250, 300]):
                ndcg = evaluation_for_max_k(K, all_user_items, train_user_items_f, val_user_items_f, item_deg_f, popular_items_f, co_counts_f)
                ndcg_cv_sum[idx] += ndcg
                print(f"MAX_K={K} → fold {fold+1} Val NDCG@20 = {ndcg:.4f}")

        ndcg_cv_avg = [v / n_folds for v in ndcg_cv_sum]
        cv_results[n_folds] = ndcg_cv_avg

        print(f"\n {n_folds}-fold CV-averaged NDCG@20")
        for K, ndcg in zip([20, 50, 100, 150, 200, 250, 300], ndcg_cv_avg):
            print(f"MAX_K={K} : [{n_folds}-fold CV avg] Val NDCG@20 = {ndcg:.4f}")

    # 7. Plot comparison: no CV vs. 5-fold vs. 10-fold
    plt.figure(figsize=(8, 5))
    plt.plot([20, 50, 100, 150, 200, 250, 300], ndcg_no_cv, marker='o', label="Single split (no CV)")

    if 5 in cv_results:
        plt.plot([20, 50, 100, 150, 200, 250, 300], cv_results[5], marker='s', linestyle='--', label="5-fold CV (avg)")
    if 10 in cv_results:
        plt.plot([20, 50, 100, 150, 200, 250, 300], cv_results[10], marker='^', linestyle='-.', label="10-fold CV (avg)")

    plt.title("Item-based-kNN: Neighbor Size vs NDCG@20\nNo CV vs. 5-fold CV vs. 10-fold CV")
    plt.xlabel("Number of neighbors (MAX_K)")
    plt.ylabel("Validation NDCG@20")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("neighbors_vs_ndcg20_cv_5_10.png")
    plt.show()

    print("\nSaved plot as neighbors_vs_ndcg20_cv_5_10.png")

    # 8. Use best MAX_K to generate and save 20 recommendations per user
    # Choose best MAX_K from one of the curves with 10-fold CV, 5-fold CV, and no-CV
    def best_value_k():
        if 10 in cv_results:
            ndcgs = cv_results[10]
            label = "10-fold CV"
        elif 5 in cv_results:
            ndcgs = cv_results[5]
            label = "5-fold CV"
        else:
            ndcgs = ndcg_no_cv
            label = "no CV"

        best_idx = max(range(len([20, 50, 100, 150, 200, 250, 300])), key=lambda i: ndcgs[i])
        best_k = NEIGHBOR_SIZES[best_idx]
        best_ndcg = ndcgs[best_idx]
        print(f"\nBest MAX_K based on {label}: MAX_K={best_k}, Val NDCG@20={best_ndcg:.4f}")
        return best_k

    best_k = best_value_k()

    # We can use the single-split train/val stats again for the final model.
    generate_and_save_rec(
        MAX_K=best_k,
        all_user_items=all_user_items,
        train_user_items=train_user_items_single,
        item_deg=item_deg_s,
        popular_items=popular_items_s,
        co_counts=co_counts_s,
        output_path="recommendations_item_knn_best.txt",
    )

# Jaccard similarity is compared with SVD+cosine similarity in terms of NDCG@20.
def compare_jaccard_svd():
    random.seed(42)

    # Helper function: NDCG@20 is computed at the number of neighbor k.
    def ndcg_at_k(preds, gt_set, k = 20):
        dcg = 0.0
        for rank, it in enumerate(preds[:k]):
            if it in gt_set:
                dcg += 1.0 / math.log2(rank + 2)
        ideal = min(len(gt_set), k)
        if ideal == 0:
            return 0.0
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
        return dcg / idcg if idcg > 0 else 0.0

    # 1. Load ALL items per user from train-1.txt
    def load_user_items(path):
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

    all_user_items = load_user_items("train-1.txt")
    print(f"Loaded users from original file: {len(all_user_items)}")

    # 2. train/val split is performed per user 
    def splits_per_user(all_user_items, seed):
        random.seed(seed)
        train_user_items = collections.defaultdict(set)
        val_user_items = collections.defaultdict(set)

        for u, items in all_user_items.items():
            uniq_items = list(set(items))
            if len(uniq_items) == 1:
                train_user_items[u].add(uniq_items[0])
                continue

            items_copy = uniq_items[:]
            random.shuffle(items_copy)

            k_val = max(1, int(0.2 * len(items_copy)))
            val_items = items_copy[:k_val]
            train_items = items_copy[k_val:]

            for it in train_items:
                train_user_items[u].add(it)
            for it in val_items:
                val_user_items[u].add(it)

        return train_user_items, val_user_items

    train_user_items, val_user_items = splits_per_user(all_user_items, seed=42)
    print(f"Train users: {len(train_user_items)}, Val users: {len(val_user_items)}")
    all_users_sorted = sorted(all_user_items.keys())

    # 3. Precompute item_users, item_deg, popularity, and co_counts for Jaccard
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

    # 4. Jaccard: build neighbors and evaluate.
    def jaccard_evaluation_for_max_k(MAX_K):
        # build neighbors
        heaps = collections.defaultdict(list)

        for (i, j), inter in co_counts.items():
            if inter < 3:
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

        neighbors = dict()
        for it, h in heaps.items():
            lst = [heapq.heappop(h) for _ in range(len(h))]
            lst.sort(key=lambda t: (-t[0], t[1]))
            neighbors[it] = lst

        # recommend for entire users
        user_recs = dict()

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
                recs = [it for it in popular_items if it not in seen][:20]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:20]]
                if len(recs) < 20:
                    for it in popular_items:
                        if len(recs) >= 20:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)

            if len(recs) > 20:
                recs = recs[:20]

            user_recs[u] = recs

        # evaluation using NDCG@20
        val_ndcg_sum = 0.0
        val_users_eval = 0

        for u, gt_items in val_user_items.items():
            if len(gt_items) == 0:
                continue
            recs = user_recs.get(u)
            if recs is None:
                continue

            n = ndcg_at_k(recs, gt_items, k=20)
            val_ndcg_sum += n
            val_users_eval += 1

        return val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

    # 5. Build user–item matrix for SVD+cosine for train.
    def user_item_matrix(user_items):
        users = sorted(user_items.keys())
        user2idx = {u: idx for idx, u in enumerate(users)}

        items_set = {it for its in user_items.values() for it in its}
        items = sorted(items_set)
        item2idx = {it: idx for idx, it in enumerate(items)}

        rows, cols, data = list(), list(), list()

        for u, its in user_items.items():
            ui = user2idx[u]
            for it in its:
                ii = item2idx[it]
                rows.append(ui)
                cols.append(ii)
                data.append(1.0)

        X = sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))
        return X, users, items

    X, _, items_list = user_item_matrix(train_user_items)
    print("User–item matrix shape:", X.shape)

    # Compute SVD item embeddings once
    svd = TruncatedSVD(n_components=64, random_state=42)
    item_emb = svd.fit_transform(X.T)
    item_emb = normalize(item_emb, norm="l2", axis=1)
    num_items = item_emb.shape[0]
    idx2item = {idx: it for idx, it in enumerate(items_list)}

    # Popularity for SVD model is calculated.
    item_deg_svd = collections.Counter()
    for u, items in train_user_items.items():
        for it in items:
            item_deg_svd[it] += 1
    popular_svd = sorted(item_deg_svd.items(), key=lambda kv: (-kv[1], kv[0]))
    popular_items_svd = [it for it, _ in popular_svd]

    # 6. SVD+cosine evaluation function.
    def svd_evaluation_for_max_k(MAX_K):
        # build neighbors from embeddings
        neighbors = {it: list() for it in items_list}

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

        # Recommend for all users
        user_recs = dict()
        for u in all_users_sorted:
            seen = train_user_items.get(u, set())
            scores = collections.defaultdict(float)
            if seen:
                if USER_NORMALIZE:
                    w = 1.0 / len(seen)
                else:
                    w = 1.0
                for it in seen:
                    for sim, nb in neighbors.get(it, list()):
                        if nb in seen:
                            continue
                        scores[nb] += sim * w

            if not scores:
                recs = [it for it in popular_items_svd if it not in seen][:20]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:20]]
                if len(recs) < 20:
                    for it in popular_items_svd:
                        if len(recs) >= 20:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)

            if len(recs) > 20:
                recs = recs[:20]

            user_recs[u] = recs

        # validation NDCG
        val_ndcg_sum = 0.0
        val_users_eval = 0

        for u, gt_items in val_user_items.items():
            if len(gt_items) == 0:
                continue
            recs = user_recs.get(u)
            if recs is None:
                continue

            n = ndcg_at_k(recs, gt_items, k=20)
            val_ndcg_sum += n
            val_users_eval += 1

        return val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

    # 7a. A function to generate and save recommendations using Jaccard similarity.
    def save_jaccard(MAX_K, output_path):
        # build neighbors
        heaps = collections.defaultdict(list)

        for (i, j), inter in co_counts.items():
            if inter < 3:
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

        neighbors = dict()
        for it, h in heaps.items():
            lst = [heapq.heappop(h) for _ in range(len(h))]
            lst.sort(key=lambda t: (-t[0], t[1]))
            neighbors[it] = lst

        # Entire users are sorted.
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
                        for sim, nb in neighbors.get(it, list()):
                            if nb in seen:
                                continue
                            scores[nb] += sim * w

                if not scores:
                    recs = [it for it in popular_items if it not in seen][:20]
                else:
                    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                    recs = [it for it, _ in ranked[:20]]
                    if len(recs) < 20:
                        for it in popular_items:
                            if len(recs) >= 20:
                                break
                            if it not in seen and it not in scores:
                                recs.append(it)
                if len(recs) > 20:
                    recs = recs[:20]
                f.write(" ".join(recs) + "\n")

        print(f"\n[Jaccard] Saved top-20 recommendations to {output_path} (MAX_K={MAX_K})")

    # 7b. A function to generate and save recommendations using SVD+cosine similarity.
    def save_svd(MAX_K, output_path):
        # build neighbors from embeddings
        neighbors = {it: list() for it in items_list}

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

        # Entire users are sorted.
        def user_sort_key(u):
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
                        for sim, nb in neighbors.get(it, list()):
                            if nb in seen:
                                continue
                            scores[nb] += sim * w

                if not scores:
                    recs = [it for it in popular_items_svd if it not in seen][:20]
                else:
                    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                    recs = [it for it, _ in ranked[:20]]
                    if len(recs) < 20:
                        for it in popular_items_svd:
                            if len(recs) >= 20:
                                break
                            if it not in seen and it not in scores:
                                recs.append(it)

                if len(recs) > 20:
                    recs = recs[:20]

                f.write(" ".join(recs) + "\n")

        print(f"\n[SVD] Saved top-20 recommendations to {output_path} (MAX_K={MAX_K}, EMB_DIM=64)")

    # 8. Sweep neighbor sizes and collect NDCG@20
    ndcg_jaccard, ndcg_svd = list(), list()

    print("\nRunning neighbor-size sweep: Jaccard vs. SVD\n")

    for K in [20, 50, 100, 150, 200, 250, 300]:
        ndcg_j = jaccard_evaluation_for_max_k(K)
        ndcg_s = svd_evaluation_for_max_k(K)
        ndcg_jaccard.append(ndcg_j)
        ndcg_svd.append(ndcg_s)
        print(f"MAX_K={K:3d} → Jaccard NDCG@20={ndcg_j:.6f}, SVD NDCG@20={ndcg_s:.6f}")

    # 9. Plot Jaccard vs SVD: Neighbor Size vs NDCG@20
    plt.figure(figsize=(8, 5))
    plt.plot([20, 50, 100, 150, 200, 250, 300], ndcg_jaccard, marker='o', label="Jaccard item-kNN")
    plt.plot([20, 50, 100, 150, 200, 250, 300], ndcg_svd, marker='s', linestyle='--', label=f"SVD+cosine item-kNN (dim=64)")

    plt.title("Item-kNN: Neighbor Size vs NDCG@20\nJaccard vs SVD+Cosine")
    plt.xlabel("Number of neighbors (MAX_K)")
    plt.ylabel("Validation NDCG@20")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("neighbors_vs_ndcg20_jaccard_vs_svd.png")
    plt.show()

    print("\nSaved plot as neighbors_vs_ndcg20_jaccard_vs_svd.png")

    # 10. choose best MAX_K and save rec files
    # Best Jaccard K
    best_j_idx = max(range(len([20, 50, 100, 150, 200, 250, 300])), key=lambda i: ndcg_jaccard[i])
    best_j_k = NEIGHBOR_SIZES[best_j_idx]
    print(f"\nBest Jaccard MAX_K={best_j_k} with Val NDCG@20={ndcg_jaccard[best_j_idx]:.4f}")

    # Best SVD K
    best_s_idx = max(range(len([20, 50, 100, 150, 200, 250, 300])), key=lambda i: ndcg_svd[i])
    best_s_k = NEIGHBOR_SIZES[best_s_idx]
    print(f"Best SVD MAX_K={best_s_k} with Val NDCG@20={ndcg_svd[best_s_idx]:.4f}")

    # Generate files for jaccard and svd
    save_jaccard(MAX_K=best_j_k, output_path="recommendations_jaccard_best.txt")
    save_svd(MAX_K=best_s_k, output_path="recommendations_svd_best.txt")

# NDCG@20 values are measured varying neighbor size, co-occurrence of items, and Jaccard shrinkage.
def hyperparameter_sweep():
    random.seed(42)

    # HELPER FUNCTIONS: Hit evaluation metric is used at the size of neighbor k.
    def hit_at_k(preds, gt_set, k=20):
        return 1.0 if any(item in gt_set for item in preds[:k]) else 0.0

    # NDCG values are measured at the size of neighbor k.
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

    # 1. LOAD ORIGINAL DATA
    all_user_items = collections.defaultdict(list)

    with open("train-1.txt", "r", encoding="utf-8") as f:
        for line in f:
            nums = re.findall(r"-?\d+", line)
            if len(nums) < 2:
                continue
            u = nums[0]
            items = nums[1:]
            all_user_items[u].extend(items)

    print(f"Loaded users from original file: {len(all_user_items)}")

    # 2. Datasets are splitted into train/val per user.
    train_user_items = collections.defaultdict(set)
    val_user_items = collections.defaultdict(set)

    for u, items in all_user_items.items():
        uniq_items = list(set(items))
        if len(uniq_items) == 1:
            train_user_items[u].add(uniq_items[0])
            continue

        items_copy = uniq_items[:]
        random.shuffle(items_copy)

        k_val = max(1, int(0.2 * len(items_copy)))
        val_items = items_copy[:k_val]
        train_items = items_copy[k_val:]

        for it in train_items:
            train_user_items[u].add(it)
        for it in val_items:
            val_user_items[u].add(it)

    print(f"Train users: {len(train_user_items)}, Val users: {len(val_user_items)}")

    # 3. Precompute item stats
    item_users = collections.defaultdict(set)
    for u, items in train_user_items.items():
        for it in items:
            item_users[it].add(u)

    print(f"Num items in TRAIN: {len(item_users)}")
    item_deg = {it: len(us) for it, us in item_users.items()}
    popular = sorted(item_deg.items(), key=lambda kv: (-kv[1], kv[0]))
    popular_items = [it for it, _ in popular]

    # Co-occurrences of items are counted between users.
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

    # 4. Training is performed in this function.
    def run_model(MAX_K, ALPHA):
        # build neighbors for this hyperparam setting
        heaps = collections.defaultdict(list)

        for (i, j), inter in co_counts.items():
            if inter < 3:
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

        neighbors = dict()
        for it, h in heaps.items():
            lst = [heapq.heappop(h) for _ in range(len(h))]
            lst.sort(key=lambda t: (-t[0], t[1]))
            neighbors[it] = lst

        # Make recommendations for entire users
        user_recs = dict()
        for u in all_users:
            seen = train_user_items.get(u, set())
            scores = collections.defaultdict(float)

            if seen:
                if USER_NORMALIZE:
                    user_weight = 1.0 / len(seen)
                else:
                    user_weight = 1.0

                for it in seen:
                    for sim, nb in neighbors.get(it, list()):
                        if nb in seen:
                            continue
                        scores[nb] += sim * user_weight

            if not scores:
                recs = [it for it in popular_items if it not in seen][:20]
            else:
                ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                recs = [it for it, _ in ranked[:20]]
                if len(recs) < 20:
                    for it in popular_items:
                        if len(recs) >= 20:
                            break
                        if it not in seen and it not in scores:
                            recs.append(it)

            if len(recs) > 20:
                recs = recs[:20]

            user_recs[u] = recs

        # training Leave One Out
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
                    for sim, nb in neighbors.get(it, list()):
                        if nb in seen:
                            continue
                        scores[nb] += sim * user_weight
            if not scores:
                continue

            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            recs = [it for it, _ in ranked[:20]]
            if len(recs) < 20:
                for it in popular_items:
                    if len(recs) >= 20:
                        break
                    if it not in seen and it not in scores:
                        recs.append(it)
            if len(recs) > 20:
                recs = recs[:20]

            gt_set = {heldout}
            h = hit_at_k(recs, gt_set, k=20)
            n = ndcg_at_k(recs, gt_set, k=20)

            train_hit_sum += h
            train_ndcg_sum += n
            train_users_eval += 1

        train_hit = train_hit_sum / train_users_eval if train_users_eval > 0 else 0.0
        train_ndcg = train_ndcg_sum / train_users_eval if train_users_eval > 0 else 0.0

        val_hit_sum = 0.0
        val_ndcg_sum = 0.0
        val_users_eval = 0

        for u, gt_items in val_user_items.items():
            if len(gt_items) == 0:
                continue
            recs = user_recs.get(u)
            if recs is None:
                continue

            h = hit_at_k(recs, gt_items, k=20)
            n = ndcg_at_k(recs, gt_items, k=20)

            val_hit_sum += h
            val_ndcg_sum += n
            val_users_eval += 1

        val_hit = val_hit_sum / val_users_eval if val_users_eval > 0 else 0.0
        val_ndcg = val_ndcg_sum / val_users_eval if val_users_eval > 0 else 0.0

        print(f"MAX_K={MAX_K:3d}, MIN_INTER=3, ALPHA=20.0 "
            f"| TrainNDCG@20={train_ndcg:.4f}, ValNDCG@20={val_ndcg:.4f}, ValHit@20={val_hit:.4f}")

        return train_ndcg, val_ndcg, train_hit, val_hit

    # 5. GRID SEARCH
    best_cfg = None
    best_val_ndcg = -1.0

    for max_k in [100, 200, 300]:
        for min_inter in [1, 2, 3]:
            for alpha in [0.0, 5.0, 10.0, 20.0]:
                tn, vn, th, _ = run_model(max_k, alpha)
                if vn > best_val_ndcg:
                    best_val_ndcg = vn
                    best_cfg = (max_k, min_inter, alpha, tn, th)

    print("\nBest config by ValNDCG@20:")
    print(f"MAX_K={best_cfg[0]}, MIN_INTER={best_cfg[1]}, ALPHA={best_cfg[2]} "
        f"| TrainNDCG@20={best_cfg[3]:.4f}, ValNDCG@20={best_val_ndcg:.4f}")
    
def item_based_knn():
    # change n-fold: no CV, 5 fold, and 10 fold
    n_fold_variation()
    # compare Jaccard and svd+cosine
    compare_jaccard_svd()
    # hyperparameter sweep (MAX_K, MIN_INTER, and ALPHA)
    hyperparameter_sweep()

if __name__ == "__main__":
    item_based_knn()