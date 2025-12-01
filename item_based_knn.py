import collections
import heapq
import pickle
import re
import math
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

USER_NORMALIZE = True
TRAIN_FILE = "train-1.txt"

# hyperparameters for the model
MAX_K = 200       # number of neighbors per item
MIN_INTER = 3     # minimum co-occurrence to keep
ALPHA = 20.0      # Jaccard shrinkage

# function for loading user items
def _load_user_items_as_sets(path):
    all_user_items = collections.defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            nums = re.findall(r"-?\d+", line)
            if len(nums) < 2:
                continue
            u = nums[0]
            items = nums[1:]
            for it in items:
                all_user_items[u].add(it)
    return all_user_items

# function for building stats: number of users per item, popularity list, and co-occurrence count
def _build_stats(train_user_items):
    item_users = collections.defaultdict(set)
    for u, items in train_user_items.items():
        for it in items:
            item_users[it].add(u)

    # degree = number of users per item
    item_deg = {it: len(us) for it, us in item_users.items()}

    # popularity list
    popular = sorted(item_deg.items(), key=lambda kv: (-kv[1], kv[0]))
    popular_items = [it for it, _ in popular]

    # co-occurrence counts
    co_counts = collections.Counter()
    for u, items in train_user_items.items():
        if len(items) < 2:
            continue
        sitems = sorted(items)
        for i_idx in range(len(sitems)):
            i = sitems[i_idx]
            for j in sitems[i_idx + 1:]:
                co_counts[(i, j)] += 1

    return item_deg, popular_items, co_counts

# function for building item-item neighbors using Jaccard with shrinkage
def _build_neighbors_jaccard(item_deg, co_counts, max_k=MAX_K, min_inter=MIN_INTER, alpha=ALPHA):
    heaps = collections.defaultdict(list)

    for (i, j), inter in co_counts.items():
        if inter < min_inter:
            continue
        ui = item_deg.get(i, 0)
        uj = item_deg.get(j, 0)
        if ui == 0 or uj == 0:
            continue
        union = ui + uj - inter
        if union <= 0:
            continue

        sim = inter / (union + alpha)
        if sim <= 0:
            continue

        # push symmetric neighbors
        heapq.heappush(heaps[i], (sim, j))
        if len(heaps[i]) > max_k:
            heapq.heappop(heaps[i])

        heapq.heappush(heaps[j], (sim, i))
        if len(heaps[j]) > max_k:
            heapq.heappop(heaps[j])

    neighbors = dict()
    for it, h in heaps.items():
        lst = [heapq.heappop(h) for _ in range(len(h))]
        # sort by descending similarity, then by item id
        lst.sort(key=lambda t: (-t[0], t[1]))
        neighbors[it] = lst

    return neighbors

# function for including svd factor to cosine metric
def build_item_factors_svd(train_user_items, n_components=100):
    # Map items to indices
    all_items = sorted({it for items in train_user_items.values() for it in items})
    item_index = {it: idx for idx, it in enumerate(all_items)}
    n_items = len(all_items)

    # Map users to indices
    all_users = sorted(train_user_items.keys())
    user_index = {u: idx for idx, u in enumerate(all_users)}
    n_users = len(all_users)

    # Build item-user matrix (rows=items, cols=users)
    rows, cols, data = list(), list(), list()
    for u, items in train_user_items.items():
        u_idx = user_index[u]
        for it in items:
            i_idx = item_index[it]
            rows.append(i_idx)
            cols.append(u_idx)
            data.append(1.0)

    X = csr_matrix((data, (rows, cols)), shape=(n_items, n_users), dtype=np.float32)

    # Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    V = svd.fit_transform(X)             # shape: (n_items, n_components)

    # L2-normalize item factors so dot == cosine
    V = normalize(V, axis=1)

    item_factors = {
        it: V[item_index[it]] for it in all_items
    }
    return item_factors, item_index

# function for building neighbors using cosine metric
def _build_neighbors_cosine(
    item_deg,
    co_counts,
    max_k=MAX_K,
    min_inter=MIN_INTER,
    item_factors=None,   # dict: item_id -> np.array([...]) or None
    alpha=0.5            # weight for classic cosine; (1-alpha) for SVD-cosine
):
    heaps = collections.defaultdict(list)  # item -> min-heap of (sim, neighbor)

    use_svd = item_factors is not None and (1.0 - alpha) > 0.0

    for (i, j), inter in co_counts.items():
        if inter < min_inter:
            continue

        ui = item_deg.get(i, 0)  # |U_i|
        uj = item_deg.get(j, 0)  # |U_j|
        if ui == 0 or uj == 0:
            continue

        # ---- classic cosine on counts ----
        denom = math.sqrt(ui * uj)
        if denom <= 0:
            continue
        sim_cf = inter / denom

        # ---- SVD-based cosine (if available) ----
        if use_svd and (i in item_factors) and (j in item_factors):
            vi = item_factors[i]
            vj = item_factors[j]
            # assume vi, vj already L2-normalized; if not, normalize here
            sim_svd = float(np.dot(vi, vj))
        else:
            sim_svd = 0.0

        # ---- combine similarities ----
        sim = alpha * sim_cf + (1.0 - alpha) * sim_svd
        if sim <= 0:
            continue

        # push symmetric neighbors, keeping only top max_k by similarity
        heapq.heappush(heaps[i], (sim, j))
        if len(heaps[i]) > max_k:
            heapq.heappop(heaps[i])

        heapq.heappush(heaps[j], (sim, i))
        if len(heaps[j]) > max_k:
            heapq.heappop(heaps[j])

    # Convert heaps to sorted lists (descending sim)
    neighbors = dict()
    for it, h in heaps.items():
        lst = [heapq.heappop(h) for _ in range(len(h))]
        lst.sort(key=lambda t: (-t[0], t[1]))  # sort by sim desc, then item id
        neighbors[it] = lst

    return neighbors

# function for scoring user items
def _score_user_items(user_items, neighbors, popular_items, top_k=20):
    seen = set(user_items)
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
        # cold-start or no neighbor contributions -> popularity backfill
        recs = [it for it in popular_items if it not in seen][:top_k]
    else:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        recs = [it for it, _ in ranked[:top_k]]
        if len(recs) < top_k:
            for it in popular_items:
                if len(recs) >= top_k:
                    break
                if it not in seen and it not in scores:
                    recs.append(it)

    if len(recs) > top_k:
        recs = recs[:top_k]
    return recs

def item_based_knn():
    # function for training and saving the model
    def train_and_save_model(model_name):
        
        train_user_items = _load_user_items_as_sets(TRAIN_FILE)
        item_deg, popular_items, co_counts = _build_stats(train_user_items)

        if "jaccard" in model_name:
            print(f"[train_and_save_model] Building Jaccard neighbors (MAX_K={MAX_K}, "
            f"MIN_INTER={MIN_INTER}, ALPHA={ALPHA})...")
            neighbors = _build_neighbors_jaccard(item_deg, co_counts, max_k=MAX_K,
                                    min_inter=MIN_INTER, alpha=ALPHA)
        else:
            item_factors, _ = build_item_factors_svd(train_user_items, n_components=100)
            print(f"[train_and_save_model] Building cosine neighbors (MAX_K={MAX_K}, "
            f"MIN_INTER={MIN_INTER}, ALPHA={ALPHA})...")
            neighbors = _build_neighbors_cosine(item_deg, co_counts, max_k=MAX_K,
                                    min_inter=MIN_INTER, item_factors=item_factors, alpha=ALPHA)

        model = {
            "neighbors": neighbors,
            "popular_items": popular_items,
            "train_user_items": train_user_items
        }

        path = f"{model_name}.h5"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[train_and_save_model] Saved model to {path}")
    
    # function for loading model
    def load_model(model_name):
        path = f"{model_name}.h5"
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[load_model] Loaded model from {path}")
        return model

    # recommend top-20 items for entire users and save to a file
    def recommend_all_users(model, out_path, top_k=20, include_user_id=True):
        neighbors = model["neighbors"]
        popular_items = model["popular_items"]
        train_user_items = model["train_user_items"]

        with open(out_path, "w", encoding="utf-8") as f:
            for u in sorted(train_user_items.keys(), key=lambda x: int(x)):
                user_items = train_user_items[u]
                recs = _score_user_items(user_items, neighbors, popular_items, top_k=top_k)
                if include_user_id:
                    line = " ".join([u] + recs)
                else:
                    line = " ".join(recs)
                f.write(line + "\n")
        print(f"[recommend_all_users] Wrote recommendations to {out_path}")

    # 1.1) Train once and save for jaccard metric
    train_and_save_model("itemknn_jaccard")
    # 1.2) Later: load and predict for jaccard metric
    jaccard_model = load_model("itemknn_jaccard")
    # 1.3) Recommend 20 items for ALL users (Jaccard)
    recommend_all_users(
        jaccard_model,
        out_path="recommendations_itemknn_jaccard.txt",
        top_k=20,
        include_user_id=False
    )

    # 2.1) Train once and save for cosine metric
    train_and_save_model("itemknn_cosine")
    # 2.2) Later: load for cosine metric
    cosine_model = load_model("itemknn_cosine")
    # 2.3) Recommend 20 items for ALL users (Cosine)
    recommend_all_users(
        cosine_model,
        out_path="recommendations_itemknn_cosine.txt",
        top_k=20,
        include_user_id=False
    )

if __name__ == "__main__":
    item_based_knn()