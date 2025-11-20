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
out_path = Path("recommendations_jaccard10_new.txt")
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