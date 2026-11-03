"""
Convert the Gaza Random Forest into Google Earth Engine decision-tree strings.

The Syria run classifies inside Earth Engine instead of exporting 28-band feature
rasters (roughly 1 TB per window at national scale). Earth Engine can run a forest
written as text (ee.Classifier.decisionTreeEnsemble), so the trained scikit-learn
forest is written out in that text format and stored as a table asset.

Usage, from the repository root:
    python -m src.syria.forest_to_gee selftest      Offline checks of the converter (no Earth Engine)
    python -m src.syria.forest_to_gee report        Size and shape of the forest and of its text form
    python -m src.syria.forest_to_gee convert       Write the chunked CSV for the table upload
    python -m src.syria.forest_to_gee toy-check     Small test forests sent to Earth Engine (needs login)
    python -m src.syria.forest_to_gee verify-asset  After the upload: compare it with the conversion, then run the
                                                    uploaded forest in Earth Engine on ~100 probe vectors and
                                                    compare with the local forest

The text format is the one written by geemap.ml.tree_to_string (rpart style, node ids
2k and 2k+1, probability of the damaged class at each leaf). This module reproduces it
byte for byte on depth-first trees (checked by `selftest`). geemap itself writes malformed
trees for forests grown with max_leaf_nodes, because it assumes depth-first node numbering,
so it is not used here. Trees can also be pruned to shrink them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_PROJECT = "gaza-damage-mapping"
DEFAULT_ASSET = f"projects/{DEFAULT_PROJECT}/assets/gaza_rf_trees"
MODEL_RELPATH = Path("runs") / "rf_s1_2months_50trees_1x1_all7reducers" / "model.pkl"

# Feature order used by get_features_names(): band, period, window, reducer.
BANDS = ["VV", "VH"]
PERIODS = ["pre", "post"]
WINDOW = "1x1"
REDUCERS = ["mean", "stdDev", "median", "min", "max", "skew", "kurtosis"]
DEFAULT_FEATURES = [f"{b}_{p}_{WINDOW}_{r}" for b in BANDS for p in PERIODS for r in REDUCERS]

CHUNK_MARKER = "|"  # Prefix on every stored chunk so leading spaces cannot be trimmed on upload.
NEWLINE_CODE = "#"  # Newlines are stored as '#', as in geemap.ml.export_trees_to_fc.
BYTES_PER_NODE = 115  # Rough size of one line of tree text, used only for the size guard.
EXACT_NODE_LIMIT = 3_000_000  # Above this many nodes, report converts only a sample of trees.


# Tree container


@dataclass
class Tree:
    """One decision tree as flat arrays (children are -1 at leaves)."""

    left: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    p1: np.ndarray  # Fraction of the damaged class at each node (unrounded).
    n: np.ndarray  # Training rows reaching each node.
    w: np.ndarray  # Weighted training rows reaching each node.
    imp: np.ndarray  # Node impurity.

    @classmethod
    def from_sklearn(cls, est, positive: int = 1) -> "Tree":
        t = est.tree_
        v = t.value[:, 0, :]
        p1 = v[:, positive] / v.sum(axis=1)
        return cls(
            t.children_left.astype(np.int64),
            t.children_right.astype(np.int64),
            t.feature.astype(np.int64),
            t.threshold.astype(np.float64),
            p1,
            t.n_node_samples.astype(np.int64),
            t.weighted_n_node_samples.astype(np.float64),
            t.impurity.astype(np.float64),
        )

    @property
    def n_nodes(self) -> int:
        return len(self.left)

    @property
    def n_leaves(self) -> int:
        return int((self.left == -1).sum())

    def depths(self) -> np.ndarray:
        """Depth of every node (root = 0)."""
        depth = np.zeros(self.n_nodes, dtype=np.int64)
        stack = [0]
        while stack:
            i = stack.pop()
            if self.left[i] != -1:
                for c in (self.left[i], self.right[i]):
                    depth[c] = depth[i] + 1
                    stack.append(c)
        return depth

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Index of the leaf each row ends in (same rule as scikit-learn)."""
        X = np.asarray(X, dtype=np.float32)
        node = np.zeros(len(X), dtype=np.int64)
        active = np.arange(len(X))
        while len(active):
            nd = node[active]
            internal = self.left[nd] != -1
            active, nd = active[internal], nd[internal]
            if not len(active):
                break
            go_left = X[active, self.feature[nd]] <= self.threshold[nd]
            node[active] = np.where(go_left, self.left[nd], self.right[nd])
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probability of the damaged class for each row."""
        return self.p1[self.apply(X)]


def predict_forest(trees: list[Tree], X: np.ndarray) -> np.ndarray:
    return np.mean([t.predict(X) for t in trees], axis=0)


def iter_trees(clf, max_depth: int | None = None, max_leaves: int | None = None):
    """Yield the forest's trees one at a time, pruned if requested (keeps memory low)."""
    for est in clf.estimators_:
        yield prune(Tree.from_sklearn(est), max_depth, max_leaves)


# Pruning


def _compact(tree: Tree, left: np.ndarray, right: np.ndarray) -> Tree:
    """Re-index a tree after nodes have been collapsed into leaves."""
    order, stack = [], [0]
    while stack:
        i = stack.pop()
        order.append(i)
        if left[i] != -1:
            stack.append(right[i])
            stack.append(left[i])
    order = np.array(order, dtype=np.int64)
    new = np.full(len(left), -1, dtype=np.int64)
    new[order] = np.arange(len(order))
    lo, ro = left[order], right[order]
    return Tree(
        np.where(lo != -1, new[lo], -1),
        np.where(lo != -1, new[ro], -1),
        tree.feature[order],
        tree.threshold[order],
        tree.p1[order],
        tree.n[order],
        tree.w[order],
        tree.imp[order],
    )


def prune(tree: Tree, max_depth: int | None = None, max_leaves: int | None = None) -> Tree:
    """
    Shrink a tree by collapsing nodes into leaves.

    max_depth turns every node at that depth into a leaf. max_leaves then removes the
    splits that reduced impurity least, one at a time, until the tree has that many
    leaves. A collapsed node predicts its own class fraction, so no data are needed.
    """
    if max_depth is None and max_leaves is None:
        return tree
    left, right = tree.left.copy(), tree.right.copy()
    parent = np.full(tree.n_nodes, -1, dtype=np.int64)
    internal = np.nonzero(left != -1)[0]
    parent[left[internal]] = internal
    parent[right[internal]] = internal

    if max_depth is not None:
        cut = (tree.depths() >= max_depth) & (left != -1)
        left[cut] = -1
        right[cut] = -1

    if max_leaves is not None:
        reach = np.zeros(tree.n_nodes, dtype=bool)
        stack = [0]
        while stack:
            i = stack.pop()
            reach[i] = True
            if left[i] != -1:
                stack.extend((left[i], right[i]))
        n_leaves = int(((left == -1) & reach).sum())
        if n_leaves > max_leaves:
            w, imp = tree.w, tree.imp

            def gain(i: int) -> float:
                a, b = left[i], right[i]
                return float(w[i] * imp[i] - w[a] * imp[a] - w[b] * imp[b])

            heap = [
                (gain(i), int(i))
                for i in np.nonzero(reach & (left != -1))[0]
                if left[left[i]] == -1 and left[right[i]] == -1
            ]
            heapq.heapify(heap)
            while n_leaves > max_leaves and heap:
                _, i = heapq.heappop(heap)
                left[i] = right[i] = -1
                n_leaves -= 1
                p = parent[i]
                if p >= 0:
                    sibling = right[p] if left[p] == i else left[p]
                    if left[sibling] == -1:
                        heapq.heappush(heap, (gain(int(p)), int(p)))
    return _compact(tree, left, right)


# Text format


def tree_to_rpart(tree: Tree, names: list[str], indent: bool = True, thr_decimals: int = 6) -> str:
    """
    Write one tree in the text format read by ee.Classifier.decisionTree.

    Every node except the root gets one line: its id (2k for a left child, 2k+1 for a
    right child), the split of its parent, and a value. Leaves end in ' *' and carry the
    probability of the damaged class. This matches geemap.ml.tree_to_string in its
    PROBABILITY mode, including the statistics printed on non-leaf lines.
    """
    left, right = tree.left.tolist(), tree.right.tolist()
    fname = [names[f] if f >= 0 else "" for f in tree.feature.tolist()]
    thr = [f"{t:.{thr_decimals}f}" for t in tree.threshold.tolist()]
    val = [str(float(v)) for v in np.around(tree.p1, 6)]
    n = tree.n.tolist()
    imp = [f"{v:.4f}" for v in tree.imp.tolist()]

    out = [f"1) root {n[0]} 9999 9999 ({tree.imp.sum()})\n"]
    # Actions are pushed in reverse so that they pop in order: left line, left subtree,
    # right line, right subtree.
    stack = [(0, 0, 1, 0, "", -1)]  # (kind, node, id, depth, sign, child); kind 0 = visit, 1 = line.
    while stack:
        kind, x, cid, d, sign, child = stack.pop()
        if kind == 0:
            if left[x] == -1:
                continue
            stack.append((0, right[x], 2 * cid + 1, d + 1, "", -1))
            stack.append((1, x, 2 * cid + 1, d, ">", right[x]))
            stack.append((0, left[x], 2 * cid, d + 1, "", -1))
            stack.append((1, x, 2 * cid, d, "<=", left[x]))
        else:
            is_leaf = left[child] == -1
            s = child if is_leaf else x  # Non-leaf lines print the parent's statistics.
            pad = "  " * (d + 1) if indent else ""
            out.append(f"{pad}{cid}) {fname[x]} {sign} {thr[x]} {n[s]} {imp[s]} {val[s]}{' *' if is_leaf else ''}\n")
    return "".join(out)


def parse_rpart(text: str) -> dict:
    """Read a tree string back into {id: (feature, sign, threshold, value, is_leaf)}."""
    nodes = {}
    for line in text.splitlines()[1:]:
        s = line.strip()
        is_leaf = s.endswith("*")
        if is_leaf:
            s = s[:-1].rstrip()
        node_id, rest = s.split(") ", 1)
        feature, sign, thr, _n, _dev, yval = rest.split(" ")
        nodes[int(node_id)] = (feature, sign, float(thr), float(yval), is_leaf)
    return nodes


def eval_rpart(nodes: dict, X: np.ndarray, names: list[str]) -> np.ndarray:
    """Evaluate a parsed tree row by row, following node ids only."""
    col = {nm: i for i, nm in enumerate(names)}
    X = np.asarray(X, dtype=np.float32)
    out = np.empty(len(X))
    for r in range(len(X)):
        x, k = X[r], 1
        while True:
            if k != 1 and nodes[k][4]:
                out[r] = nodes[k][3]
                break
            feature, _, thr, _, _ = nodes[2 * k]
            k = 2 * k if float(x[col[feature]]) <= thr else 2 * k + 1
    return out


def indent_matches_ids(text: str) -> bool:
    """True if the indentation of every line agrees with the parent implied by its id."""
    last = {0: 1}  # Indentation level -> id of the latest line at that level.
    for line in text.splitlines()[1:]:
        level = (len(line) - len(line.lstrip(" "))) // 2
        node_id = int(line.strip().split(")", 1)[0])
        if last.get(level - 1) != node_id // 2:
            return False
        last[level] = node_id
    return True


def to_chunks(text: str, max_bytes: int) -> list[str]:
    """Split a tree string at line boundaries into pieces of at most max_bytes."""
    chunks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        if cur and size + len(line) > max_bytes:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks


def encode_chunk(chunk: str) -> str:
    return CHUNK_MARKER + chunk.replace("\n", NEWLINE_CODE)


def decode_chunks(encoded: list[str]) -> str:
    return "".join(c[len(CHUNK_MARKER):] for c in encoded).replace(NEWLINE_CODE, "\n")


# Model loading


def default_model_path() -> Path | None:
    try:
        from src.constants import DATA_PATH
    except Exception:
        return None
    return Path(DATA_PATH) / MODEL_RELPATH


def load_model(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not hasattr(obj, "estimators_"):
        items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, (list, tuple)) else []
        obj = next((item for item in items if hasattr(item, "estimators_")), None)
    if obj is None:
        raise SystemExit(f"{path} does not contain a fitted scikit-learn forest.")
    if getattr(obj, "n_classes_", 2) != 2:
        raise SystemExit("Only binary forests are supported.")
    return obj


def resolve_features(clf, override: str | None) -> list[str]:
    if override:
        names = [s.strip() for s in override.split(",") if s.strip()]
    elif hasattr(clf, "feature_names_in_"):
        names = list(clf.feature_names_in_)
    else:
        names = list(DEFAULT_FEATURES)
    if len(names) != clf.n_features_in_:
        raise SystemExit(f"Model expects {clf.n_features_in_} features but {len(names)} names were given.")
    return names


def model_path_from(args) -> Path:
    path = Path(args.model) if args.model else default_model_path()
    if path is None:
        raise SystemExit("Pass --model path/to/model.pkl")
    return path


def load_sample_rows(path: Path, names: list[str], n: int = 100_000, seed: int = 0) -> np.ndarray:
    """Valid pixels from a 28-band feature GeoTIFF, in model feature order."""
    import rasterio

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        desc = list(src.descriptions)
    index = {d: i for i, d in enumerate(desc)}
    missing = [nm for nm in names if nm not in index]
    if missing:
        raise SystemExit(f"{path.name} lacks bands such as {missing[:3]}; its bands are {desc[:3]}...")
    X = data[[index[nm] for nm in names]].reshape(len(names), -1).T
    X = X[~np.isnan(X).any(axis=1)]
    if len(X) > n:
        X = X[np.random.default_rng(seed).choice(len(X), n, replace=False)]
    return X


# Synthetic data for tests


def make_synthetic(rng: np.random.Generator, n: int, d: int = 28):
    X = rng.normal(size=(n, d)).astype(np.float32)
    score = 1.2 * X[:, 0] + 0.8 * X[:, 3] * X[:, 5] - 0.6 * X[:, 9] + rng.normal(scale=1.5, size=n)
    return X, (score > 0.3).astype(int)


# Command: selftest


def cmd_selftest(args) -> int:
    from sklearn.ensemble import RandomForestClassifier

    results = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")

    rng = np.random.default_rng(1)
    names = DEFAULT_FEATURES
    X, y = make_synthetic(rng, 20000)
    rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=3, max_leaf_nodes=250, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    Xt, _ = make_synthetic(rng, 3000)
    trees = [Tree.from_sklearn(e) for e in rf.estimators_]
    strings = [tree_to_rpart(t, names) for t in trees]
    truth = rf.predict_proba(Xt)[:, 1]
    print("Converter checks")

    mine = predict_forest(trees, Xt)
    check("array evaluator equals predict_proba", float(np.abs(mine - truth).max()) < 1e-9, f"max diff {np.abs(mine - truth).max():.2e}")

    parsed = [parse_rpart(s) for s in strings]
    from_text = np.mean([eval_rpart(p, Xt, names) for p in parsed], axis=0)
    diff = np.abs(from_text - truth)
    check("tree strings reproduce predict_proba", float(diff.max()) < 1e-4 and float((diff > 1e-6).mean()) < 0.002,
          f"max diff {diff.max():.2e}, rows off by >1e-6: {(diff > 1e-6).mean():.3%}")
    check("indentation agrees with node ids", all(indent_matches_ids(s) for s in strings))

    try:
        from geemap import ml

        # geemap assumes scikit-learn numbers nodes depth first, which is only true without a leaf cap.
        dfs = RandomForestClassifier(n_estimators=4, max_depth=9, min_samples_leaf=20, random_state=0, n_jobs=-1).fit(X, y)
        theirs = [ml.tree_to_string(e, names, output_mode="PROBABILITY") for e in dfs.estimators_]
        ours = [tree_to_rpart(Tree.from_sklearn(e), names) for e in dfs.estimators_]
        check("byte-identical to geemap on depth-first trees (no leaf cap)", theirs == ours)
        capped = [ml.tree_to_string(e, names, output_mode="PROBABILITY") for e in rf.estimators_[:2]]
        try:
            bad = np.mean([eval_rpart(parse_rpart(s), Xt[:500], names) for s in capped], axis=0)
            wrong = float(np.abs(bad - np.mean([t.predict(Xt[:500]) for t in trees[:2]], axis=0)).mean())
            print(f"  [info] geemap on leaf-capped trees (like this forest): mean error {wrong:.3f}, so do not use it for them")
        except Exception:
            print("  [info] geemap on leaf-capped trees (like this forest) writes malformed trees, so do not use it for them")
    except ImportError:
        print("  [skip] geemap not installed, comparison with geemap skipped")

    flat = tree_to_rpart(trees[0], names, indent=False)
    check("no-indent variant parses to the same tree", parse_rpart(flat) == parsed[0])

    pruned = [prune(t, max_depth=6, max_leaves=40) for t in trees]
    check("pruning respects max_leaves and max_depth",
          all(t.n_leaves <= 40 and t.depths().max() <= 6 for t in pruned),
          f"leaves {min(t.n_leaves for t in pruned)}-{max(t.n_leaves for t in pruned)}")
    p_text = np.mean([eval_rpart(parse_rpart(tree_to_rpart(t, names)), Xt, names) for t in pruned], axis=0)
    check("pruned strings match pruned arrays", float(np.abs(p_text - predict_forest(pruned, Xt)).max()) < 1e-4)
    agree = float(((predict_forest(pruned, Xt) >= 0.5) == (truth >= 0.5)).mean())
    print(f"         (pruned to 40 leaves and depth 6 on a toy forest: {agree:.1%} of rows keep the same side of 0.5)")

    chunks = [to_chunks(s, 3000) for s in strings]
    check("chunks round-trip", all(decode_chunks([encode_chunk(c) for c in ch]) == s for ch, s in zip(chunks, strings)),
          f"{sum(len(c) for c in chunks)} chunks of at most 3000 bytes")

    stump = RandomForestClassifier(n_estimators=1, max_depth=1, random_state=0).fit(X, y)
    t1 = Tree.from_sklearn(stump.estimators_[0])
    s1 = tree_to_rpart(t1, names)
    check("depth-1 tree round-trips", float(np.abs(eval_rpart(parse_rpart(s1), Xt, names) - t1.predict(Xt)).max()) < 1e-6)

    ok = all(results)
    print("All converter checks passed." if ok else "Some converter checks FAILED.")
    return 0 if ok else 1


# Command: report


def _verdict(total_mb: float) -> str:
    # Rules of thumb only. Earth Engine does not publish a limit for classifier strings.
    if total_mb <= 30:
        return "small enough that Earth Engine should take it"
    if total_mb <= 150:
        return "large, worth trying but may be slow or rejected; the toy check and one tile will tell"
    return "too large to expect Earth Engine to run; prune with --max-leaves (and --max-depth)"


def _extrapolate(measured: list[tuple[int, int]], nodes: list[int]) -> float:
    """Total bytes, given (nodes, bytes) for some trees and node counts for all of them."""
    per_node = sum(b for _, b in measured) / sum(n for n, _ in measured)
    return sum(b for _, b in measured) + per_node * (sum(nodes) - sum(n for n, _ in measured))


def cmd_report(args) -> int:
    import sklearn

    path = model_path_from(args)
    clf = load_model(path)
    names = resolve_features(clf, args.features)
    params = clf.get_params()

    print(f"Model file      : {path}")
    print(f"scikit-learn    : {sklearn.__version__}")
    for key in ("n_estimators", "min_samples_leaf", "max_leaf_nodes", "max_depth", "max_features", "class_weight", "bootstrap"):
        print(f"  {key:<17}: {params.get(key)}")
    guessed = not (hasattr(clf, "feature_names_in_") or args.features)
    print(f"Features ({len(names)})   : {names[0]} ... {names[-1]}" + ("  (default order, check against get_features_names)" if guessed else ""))
    if params.get("max_leaf_nodes") is None:
        print("  NOTE: max_leaf_nodes is None, so these trees were grown without a leaf cap.")

    node_counts = [e.tree_.node_count for e in clf.estimators_]
    exact_all = sum(node_counts) <= EXACT_NODE_LIMIT
    pruning = bool(args.max_depth or args.max_leaves)
    nodes, leaves, depth, indented, compact, pruned = [], [], [], [], [], []
    t0 = time.time()
    for k, est in enumerate(clf.estimators_):
        tree = Tree.from_sklearn(est)
        nodes.append(tree.n_nodes)
        leaves.append(tree.n_leaves)
        depth.append(int(tree.depths().max()))
        if exact_all or k < args.sample_trees:
            indented.append((tree.n_nodes, len(tree_to_rpart(tree, names))))
            compact.append((tree.n_nodes, len(tree_to_rpart(tree, names, indent=False))))
        if pruning:
            pruned.append(prune(tree, args.max_depth, args.max_leaves))

    print(f"\nTrees: {len(nodes)}")
    print(f"  nodes per tree : min {min(nodes):,}  median {int(np.median(nodes)):,}  max {max(nodes):,}  (total {sum(nodes):,})")
    print(f"  leaves per tree: median {int(np.median(leaves)):,}")
    print(f"  depth          : min {min(depth)}  median {int(np.median(depth))}  max {max(depth)}")
    print("  (node ids double with every level, so the deepest id needs about "
          f"{max(depth) + 1} bits; R's rpart, whose format this is, documents 30 as its maximum depth)")
    total = _extrapolate(indented, nodes) / 1e6
    total_flat = _extrapolate(compact, nodes) / 1e6
    how = "exact" if exact_all else f"estimated from the first {args.sample_trees} trees"
    print(f"\nEarth Engine text form ({how}, {time.time() - t0:.0f} s):")
    print(f"  with indentation (geemap format): {total:,.1f} MB total, largest tree {max(b for _, b in indented) / 1e6:,.2f} MB")
    print(f"  without indentation             : {total_flat:,.1f} MB total")
    print(f"  -> {_verdict(total)}")

    if pruning:
        sizes = [len(tree_to_rpart(p, names)) for p in pruned]
        print(f"\nAfter pruning (max_depth={args.max_depth}, max_leaves={args.max_leaves}):")
        print(f"  leaves per tree: median {int(np.median([p.n_leaves for p in pruned])):,}, max depth {max(int(p.depths().max()) for p in pruned)}")
        print(f"  text size: {sum(sizes) / 1e6:,.1f} MB total -> {_verdict(sum(sizes) / 1e6)}")
        if args.sample_raster:
            X = load_sample_rows(Path(args.sample_raster), names)
            full = clf.predict_proba(X)[:, 1]
            cut = predict_forest(pruned, X)
            flips = float(((full >= args.threshold) != (cut >= args.threshold)).mean())
            print(f"  on {len(X):,} pixels from {Path(args.sample_raster).name}: mean |diff| {np.abs(full - cut).mean():.4f}, "
                  f"correlation {np.corrcoef(full, cut)[0, 1]:.4f}, pixels changing side of t={args.threshold}: {flips:.2%}")
        else:
            print("  (add --sample-raster <a Gaza feature GeoTIFF> to measure how much pruning changes the probabilities)")
    return 0


# Command: convert


def cmd_convert(args) -> int:
    path = model_path_from(args)
    clf = load_model(path)
    names = resolve_features(clf, args.features)
    cap = 2 * args.max_leaves - 1 if args.max_leaves else None
    kept = [min(e.tree_.node_count, cap) if cap else e.tree_.node_count for e in clf.estimators_]
    est_mb = sum(kept) * BYTES_PER_NODE / 1e6
    if est_mb > args.limit_mb and not args.force:
        raise SystemExit(
            f"The forest has up to {sum(kept):,} nodes, roughly {est_mb:,.0f} MB of text, over the {args.limit_mb} MB guard.\n"
            "Run `report` first, then prune with --max-leaves / --max-depth, or pass --force to write it anyway."
        )
    out_dir = Path(args.out_dir) if args.out_dir else path.parent / "gee_export"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp_csv, fp_manifest = out_dir / "rf_trees_chunks.csv", out_dir / "manifest.json"

    chars, leaves, depth, n_rows = [], [], [], 0
    t0 = time.time()
    with open(fp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["tree", "chunk", "longitude", "latitude", "text"])
        for i, tree in enumerate(iter_trees(clf, args.max_depth, args.max_leaves)):
            text = tree_to_rpart(tree, names, indent=not args.no_indent, thr_decimals=args.thr_decimals)
            chars.append(len(text))
            leaves.append(tree.n_leaves)
            depth.append(int(tree.depths().max()))
            for j, chunk in enumerate(to_chunks(text, args.chunk_kb * 1000)):
                writer.writerow([i, j, 0, 0, encode_chunk(chunk)])
                n_rows += 1
    manifest = {
        "model": str(path), "n_trees": len(chars), "features": names, "chunk_kb": args.chunk_kb,
        "rows": n_rows, "chars_per_tree": chars, "indent": not args.no_indent, "thr_decimals": args.thr_decimals,
        "max_depth": args.max_depth, "max_leaves": args.max_leaves, "max_tree_depth": max(depth),
        "leaves_per_tree": leaves, "csv_sha256": hashlib.sha256(fp_csv.read_bytes()).hexdigest(),
    }
    fp_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {fp_csv} ({fp_csv.stat().st_size / 1e6:,.1f} MB, {n_rows:,} rows) in {time.time() - t0:.0f} s")
    print(f"Wrote {fp_manifest}")
    print(f"""
Upload the CSV as a table asset (no Cloud Storage needed):
  1. Open https://code.earthengine.google.com and sign in with the account that owns the project.
  2. Assets tab -> NEW -> Table upload -> select {fp_csv.name} (it is in the folder above, on your P: drive).
  3. Asset ID: {args.asset}
  4. Advanced options: X column = longitude, Y column = latitude.
  5. Upload, then wait for the task to finish in the Tasks tab.
If Earth Engine rejects the file, run convert again with a smaller --chunk-kb (for example 20).
Then check that it arrived intact and that Earth Engine reproduces the forest:
  python -m src.syria.forest_to_gee verify-asset --manifest {fp_manifest} --asset {args.asset}""")
    return 0


# Commands: Earth Engine


def strings_from_table(fc, n_trees: int):
    """Server-side list of the tree strings, rebuilt from the chunk table."""
    import ee

    def one_tree(i):
        parts = ee.FeatureCollection(fc).filter(ee.Filter.eq("tree", i)).sort("chunk").aggregate_array("text")
        return ee.List(parts).map(lambda s: ee.String(s).slice(len(CHUNK_MARKER))).join("").replace(NEWLINE_CODE, "\n", "g")

    return ee.List.sequence(0, n_trees - 1).map(one_tree)


def classifier_from_table(fc, n_trees: int):
    """Rebuild the ensemble inside Earth Engine from the chunk table."""
    import ee

    return ee.Classifier.decisionTreeEnsemble(strings_from_table(fc, n_trees))


def _start_ee(project: str):
    import ee

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise SystemExit(f"Could not start Earth Engine ({exc}).\nRun `earthengine authenticate` on the login node, then try again.")
    return ee


# Parity probe: run the uploaded forest in Earth Engine and compare with the local one


def _parent_and_side(tree: Tree) -> tuple[np.ndarray, np.ndarray]:
    """Parent of every node, and whether the node is its parent's right child."""
    parent = np.full(tree.n_nodes, -1, dtype=np.int64)
    is_right = np.zeros(tree.n_nodes, dtype=bool)
    inner = np.flatnonzero(tree.left != -1)
    parent[tree.left[inner]] = inner
    parent[tree.right[inner]] = inner
    is_right[tree.right[inner]] = True
    return parent, is_right


def row_reaching(tree: Tree, leaf: int, links, base: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """A feature vector that ends in `leaf`, meeting every split on the way down with room to spare."""
    parent, is_right = links
    lo, hi = np.full(len(base), -np.inf), np.full(len(base), np.inf)
    node = leaf
    while parent[node] != -1:
        up = parent[node]
        f, t = tree.feature[up], tree.threshold[up]
        if is_right[node]:
            lo[f] = max(lo[f], t)  # Went right: value > t.
        else:
            hi[f] = min(hi[f], t)  # Went left: value <= t.
        node = up
    x = base.copy()
    both = np.isfinite(lo) & np.isfinite(hi)
    only_hi = ~np.isfinite(lo) & np.isfinite(hi)
    only_lo = np.isfinite(lo) & ~np.isfinite(hi)
    x[both] = (lo[both] + hi[both]) / 2
    x[only_hi] = hi[only_hi] - spread[only_hi]
    x[only_lo] = lo[only_lo] + spread[only_lo]
    return x


def build_probes(trees: list[Tree], depth_of: list[np.ndarray], n_features: int, n_probes: int, seed: int = 0):
    """
    Feature vectors that exercise the whole forest, as float32 rows plus the kind of each:
      deepest   the deepest leaf of each tree, trees with the longest paths first
      leaf      leaves chosen at random over the forest (deep ones are the most common)
      marginal  each feature drawn from the split values used on it, plus jitter
    """
    rng = np.random.default_rng(seed)
    thr = [np.concatenate([t.threshold[t.feature == f] for t in trees]) for f in range(n_features)]
    base = np.array([np.median(v) if len(v) else 0.0 for v in thr])
    spread = np.array([max(float(np.subtract(*np.percentile(v, [75, 25]))), 1e-3) / 2 if len(v) else 1.0 for v in thr])
    links: dict[int, tuple] = {}
    leaves_of = [np.flatnonzero(t.left == -1) for t in trees]
    rows, kinds = [], []

    def add_leaf(kind: str, i: int, leaf: int) -> None:
        if i not in links:
            links[i] = _parent_and_side(trees[i])
        x = row_reaching(trees[i], leaf, links[i], base, spread).astype(np.float32)
        if int(trees[i].apply(x[None])[0]) == leaf:  # Rounding to float32 can spoil a very narrow interval.
            rows.append(x)
            kinds.append(kind)

    n_deep = min(len(trees), n_probes // 2)
    for i in sorted(range(len(trees)), key=lambda i: -depth_of[i][leaves_of[i]].max())[:n_deep]:
        add_leaf("deepest", i, int(leaves_of[i][np.argmax(depth_of[i][leaves_of[i]])]))
    for _ in range(n_probes // 3):
        i = int(rng.integers(len(trees)))
        add_leaf("leaf", i, int(rng.choice(leaves_of[i])))
    while len(rows) < n_probes:
        rows.append(np.array([rng.choice(v) + rng.uniform(-1, 1) * s if len(v) else 0.0 for v, s in zip(thr, spread)], dtype=np.float32))
        kinds.append("marginal")
    return np.array(rows, dtype=np.float32), kinds


def classify_in_ee(ee, classifier, names: list[str], X: np.ndarray, how: str, batch: int = 25) -> np.ndarray:
    """Probability for each row of X from an Earth Engine classifier; `how` is 'collection' or 'images'."""
    if how == "collection":
        feats = [ee.Feature(None, {"probe": i, **{nm: float(v) for nm, v in zip(names, row)}}) for i, row in enumerate(X)]
        info = ee.FeatureCollection(feats).classify(classifier).select(["probe", "classification"], None, False).getInfo()
        p = np.full(len(X), np.nan)
        for f in info["features"]:
            props = f["properties"]
            value = props.get("classification")
            p[int(props["probe"])] = np.nan if value is None else value
        return p
    point = ee.Geometry.Point([0, 0])

    def one(r):
        img = ee.Image.constant(ee.List(r)).rename(names)
        return img.classify(classifier).reduceRegion(ee.Reducer.first(), point, 1000).get("classification")

    out = []
    for start in range(0, len(X), batch):
        out += ee.List([[float(v) for v in row] for row in X[start:start + batch]]).map(one).getInfo()
    return np.array([np.nan if v is None else v for v in out], dtype=float)


def read_tree_texts(path: Path, n_trees: int) -> list[str]:
    """The tree strings in a chunk CSV written by `convert`."""
    csv.field_size_limit(2**31 - 1)
    chunks: dict[int, dict[int, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            chunks.setdefault(int(row["tree"]), {})[int(row["chunk"])] = row["text"]
    return [decode_chunks([chunks[i][j] for j in sorted(chunks[i])]) for i in range(n_trees)]


def run_parity(ee, args, manifest: dict) -> int:
    path = Path(args.model) if args.model else Path(manifest["model"])
    if not path.exists():
        print(f"\nParity probe not run: {path} was not found. Pass --model path/to/model.pkl.")
        return 2
    names = manifest["features"]
    clf = load_model(path)
    if clf.n_features_in_ != len(names) or len(clf.estimators_) != manifest["n_trees"]:
        print(f"\nParity probe not run: {path.name} has {len(clf.estimators_)} trees and {clf.n_features_in_} features, "
              f"but the manifest says {manifest['n_trees']} and {len(names)}.")
        return 2
    csv_path = Path(args.csv) if args.csv else Path(args.manifest).parent / "rf_trees_chunks.csv"
    if not csv_path.exists():
        print(f"\nParity probe not run: {csv_path} was not found. It is the CSV that `convert` wrote; pass --csv if it has moved.")
        return 2
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != manifest["csv_sha256"]:
        print(f"\nParity probe not run: {csv_path.name} is not the file that `convert` wrote for this manifest (checksum differs).")
        return 2
    tree_texts = read_tree_texts(csv_path, manifest["n_trees"])
    trees = list(iter_trees(clf, manifest.get("max_depth"), manifest.get("max_leaves")))
    depth_of = [t.depths() for t in trees]
    X, kinds = build_probes(trees, depth_of, len(names), args.probes, args.seed)
    counts = ", ".join(f"{kinds.count(k)} {k}" for k in ("deepest", "leaf", "marginal"))
    print(f"\nParity probe: {len(X)} test vectors ({counts})")

    # The text that was uploaded (read from the CSV that convert wrote), evaluated here, and the forest it came from.
    p_text = np.zeros(len(X))
    for text in tree_texts:
        p_text += eval_rpart(parse_rpart(text), X, names)
    p_text /= len(trees)
    p_exact = predict_forest(trees, X)
    reached = np.max([d[t.apply(X)] for d, t in zip(depth_of, trees)], axis=0)
    print(f"  Deepest path any probe follows: {int(reached.max())} (the forest's deepest is {int(max(d.max() for d in depth_of))})")

    print("  Running the uploaded forest in Earth Engine (it reads the whole table, so this can take a minute or two)...", flush=True)
    classifier = classifier_from_table(ee.FeatureCollection(args.asset), len(trees))
    p_ee, passed, label, problems = None, False, "", []
    for how, label in (("collection", "FeatureCollection.classify"), ("images", "one constant image per vector")):
        t0 = time.time()
        try:
            p = classify_in_ee(ee, classifier, names, X, how)
        except Exception as exc:
            problems.append(f"{label}: {str(exc).splitlines()[0][:400]}")
            print(f"  {label}: Earth Engine returned an error")
            continue
        p_ee = p
        diff = float(np.abs(p - p_text).max())
        passed = bool(np.isfinite(p).all() and diff < 1e-5 and len(np.unique(np.round(p, 6))) >= 5)
        print(f"  {label}: {time.time() - t0:.0f} s, largest difference from the exported text {diff:.1e}" + ("" if passed else " (too large)"))
        if passed:
            break
        problems.append(f"{label}: answers differ from the exported text by up to {diff:.1e}")
    if p_ee is None:
        print("\n  Earth Engine could not run the forest:\n    " + "\n    ".join(problems))
        print("  If a message mentions memory, time or size, the forest is too big to run as it stands: convert it again with\n"
              "  --max-leaves 5000 (or --max-depth 30), upload, and compare accuracy with `report --sample-raster`.")
        return 1

    d_text, d_exact = np.abs(p_ee - p_text), np.abs(p_ee - p_exact)
    kind = np.array(kinds)
    print(f"\n  {'probes':<10}{'n':>4}  {'max |EE - exported text|':>26}  {'max |EE - forest before rounding|':>35}")
    for name in ("deepest", "leaf", "marginal"):
        sel = kind == name
        if sel.any():
            print(f"  {name:<10}{int(sel.sum()):>4}  {d_text[sel].max():>26.2e}  {d_exact[sel].max():>35.2e}")
    side = int(((p_ee >= args.threshold) == (p_exact >= args.threshold)).sum())
    print(f"  Masked (no answer) in Earth Engine: {int(np.isnan(p_ee).sum())}; distinct answers: {len(np.unique(np.round(p_ee[np.isfinite(p_ee)], 6)))}")
    print(f"  Same side of {args.threshold} as the forest before rounding: {side} of {len(X)}")
    if passed:
        for problem in problems:
            print(f"  Note, {problem}")
        print(f"\nEarth Engine reproduces the exported forest (largest difference {d_text.max():.1e}, using {label}), "
              f"including paths {int(reached.max())} splits deep. The asset is ready to use.")
        return 0
    print("\nEarth Engine's answers do NOT match the exported text:\n  " + "\n  ".join(problems)
          + "\nDo not use the asset yet; paste this output back.")
    return 1


def cmd_verify_asset(args) -> int:
    ee = _start_ee(args.project)
    manifest = json.loads(Path(args.manifest).read_text())
    fc = ee.FeatureCollection(args.asset)
    rows = fc.size().getInfo()
    print(f"Rows in asset: {rows} (manifest says {manifest['rows']})")

    def assemble(i):
        parts = fc.filter(ee.Filter.eq("tree", i)).sort("chunk").aggregate_array("text")
        return ee.List(parts).map(lambda s: ee.String(s).slice(len(CHUNK_MARKER))).join("")

    seq = ee.List.sequence(0, manifest["n_trees"] - 1)
    lengths = seq.map(lambda i: assemble(i).length()).getInfo()
    heads = seq.map(lambda i: assemble(i).slice(0, 7)).getInfo()
    bad = [i for i, (a, b) in enumerate(zip(lengths, manifest["chars_per_tree"])) if int(a) != b]
    print(f"Trees whose length differs from the manifest: {bad if bad else 'none'}")
    print(f"Trees starting with '1) root': {sum(h == '1) root' for h in heads)} of {len(heads)}")
    ok = rows == manifest["rows"] and not bad and all(h == "1) root" for h in heads)
    print("Asset matches the local conversion." if ok else "Asset does NOT match; do not use it. Re-upload the CSV.")
    if not ok or args.no_parity:
        return 0 if ok else 1
    return run_parity(ee, args, manifest)


def _chain_tree(depth: int) -> tuple[Tree, np.ndarray]:
    """A maximally unbalanced tree (right spine) and rows that leave it at every depth."""
    n = 2 * depth + 1
    left, right = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    feature, threshold, p1 = np.full(n, -2, dtype=np.int64), np.zeros(n), np.zeros(n)
    for i in range(depth):
        left[i], right[i] = depth + i, (i + 1 if i < depth - 1 else 2 * depth)
        feature[i], threshold[i] = i % 28, -5.0 + 0.1 * i
    p1[depth:] = (np.arange(depth + 1) + 1) / (depth + 2)
    tree = Tree(left, right, feature, threshold, p1, np.full(n, 10), np.full(n, 10.0), np.full(n, 0.5))
    rows = []
    for j in range(depth):
        row = np.full(28, 10.0)
        row[feature[j]] = threshold[j] - 0.05
        rows.append(row)
    rows.append(np.full(28, 10.0))
    return tree, np.array(rows, dtype=np.float32)


def cmd_toy_check(args) -> int:
    from sklearn.ensemble import RandomForestClassifier

    ee = _start_ee(args.project)
    names = DEFAULT_FEATURES
    point = ee.Geometry.Point([0, 0])

    # Each builder takes tree strings (a Python list or a server-side list) and returns image -> image.
    def ensemble(strings):
        return lambda img: img.classify(ee.Classifier.decisionTreeEnsemble(strings))

    def ensemble_regression(strings):
        return lambda img: img.classify(ee.Classifier.decisionTreeEnsemble(strings).setOutputMode("REGRESSION"))

    def mean_of_trees(strings):
        def make(img):
            if isinstance(strings, list):
                images = [img.classify(ee.Classifier.decisionTree(s)) for s in strings]
                return ee.ImageCollection.fromImages(images).mean()
            images = ee.List(strings).map(lambda s: img.classify(ee.Classifier.decisionTree(s)))
            return ee.ImageCollection.fromImages(images).mean()

        return make

    def run(make_image, X) -> np.ndarray:
        """Classify each row as a constant image and read the value back."""
        def one(r):
            img = ee.Image.constant(ee.List(r)).rename(names)
            return make_image(img).reduceRegion(ee.Reducer.first(), point, 1000).get("classification")

        out = ee.List([[float(v) for v in row] for row in X]).map(one).getInfo()
        return np.array([np.nan if v is None else v for v in out], dtype=float)

    def attempt(label: str, make_image, X, truth):
        try:
            got = run(make_image, X)
        except Exception as exc:
            print(f"  [ERROR] {label}: {str(exc).splitlines()[0][:230]}")
            return False
        diff = np.abs(got - truth)
        worst = float(np.nanmax(diff)) if np.isfinite(diff).any() else float("nan")
        good = bool(np.isfinite(diff).all() and worst < 1e-5)
        distinct = len(np.unique(np.round(got[np.isfinite(got)], 6)))
        print(f"  [{'MATCH' if good else 'differs'}] {label}: max |EE - sklearn| = {worst:.2e}, "
              f"{distinct} distinct values, {int(np.isnan(got).sum())} masked")
        return good

    rng = np.random.default_rng(0)
    Xtr, ytr = make_synthetic(rng, 20000)
    rf = RandomForestClassifier(n_estimators=8, min_samples_leaf=3, max_leaf_nodes=300, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    rows = make_synthetic(rng, 60)[0]
    truth = rf.predict_proba(rows)[:, 1]
    trees = [Tree.from_sklearn(e) for e in rf.estimators_]
    strings = [tree_to_rpart(t, names) for t in trees]
    flat = [tree_to_rpart(t, names, indent=False) for t in trees]

    print("1. What does Earth Engine return for trees whose leaves hold probabilities?")
    builders = [
        ("decisionTreeEnsemble, default output", ensemble),
        ("decisionTreeEnsemble + setOutputMode('REGRESSION')", ensemble_regression),
        ("average of single decisionTree classifiers", mean_of_trees),
    ]
    best_label, best = None, None
    for label, builder in builders:
        if attempt(label, builder(strings), rows, truth) and best is None:
            best_label, best = label, builder
    if best is None:
        print("\nNo variant reproduced predict_proba. Paste this output back; the forest cannot be used as it stands.")
        return 1
    print(f"\n  -> use: {best_label}")

    print("\n2. Compact strings and server-side assembly from a table of chunks")
    attempt("no indentation", best(flat), rows, truth)
    features = [
        ee.Feature(None, {"tree": i, "chunk": j, "text": encode_chunk(chunk)})
        for i, s in enumerate(strings)
        for j, chunk in enumerate(to_chunks(s, 2000))
    ]
    from_table = strings_from_table(ee.FeatureCollection(features), len(strings))
    attempt("assembled from 2 kB chunks", best(from_table), rows, truth)

    print("\n3. Deep trees (node ids grow as 2^depth)")
    depths = (12, 24, 30, 31, 32, 40, 62, 64)
    passed = {}
    for depth in depths:
        tree, chain_rows = _chain_tree(depth)
        passed[depth] = attempt(f"depth {depth}", best([tree_to_rpart(tree, names)]), chain_rows, tree.predict(chain_rows))
    safe = max([d for d in depths if all(passed[x] for x in depths if x <= d)], default=0)
    print(f"\n  -> every depth up to {safe} matched" + ("" if safe == depths[-1] else f"; trees deeper than {safe} need --max-depth {safe or 30}"))
    print("\nDone. Paste everything above back.")
    return 0


# Entry point


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def model_args(sp):
        sp.add_argument("--model", help="Path to model.pkl (default: DATA_PATH/runs/rf_s1_2months_50trees_1x1_all7reducers/model.pkl)")
        sp.add_argument("--features", help="Comma-separated feature names in model order (default: derived)")
        sp.add_argument("--max-depth", type=int, help="Collapse nodes deeper than this into leaves")
        sp.add_argument("--max-leaves", type=int, help="Prune each tree down to this many leaves")

    sub.add_parser("selftest", help="Offline checks of the converter")

    sp = sub.add_parser("report", help="Forest size and Earth Engine text size")
    model_args(sp)
    sp.add_argument("--sample-trees", type=int, default=2, help="Trees to convert exactly when the forest is very large")
    sp.add_argument("--sample-raster", help="A Gaza feature GeoTIFF, to measure the effect of pruning")
    sp.add_argument("--threshold", type=float, default=0.670)

    sp = sub.add_parser("convert", help="Write the chunked CSV and manifest")
    model_args(sp)
    sp.add_argument("--out-dir", help="Default: a gee_export folder next to model.pkl")
    sp.add_argument("--chunk-kb", type=int, default=100)
    sp.add_argument("--no-indent", action="store_true", help="Drop indentation (only after toy-check confirms Earth Engine accepts it)")
    sp.add_argument("--thr-decimals", type=int, default=6)
    sp.add_argument("--asset", default=DEFAULT_ASSET)
    sp.add_argument("--limit-mb", type=int, default=400, help="Refuse to write more text than this unless --force")
    sp.add_argument("--force", action="store_true")

    sp = sub.add_parser("toy-check", help="Send small test forests to Earth Engine")
    sp.add_argument("--project", default=DEFAULT_PROJECT)

    sp = sub.add_parser("verify-asset", help="Compare the uploaded table with the manifest, then run the forest in Earth Engine")
    sp.add_argument("--manifest", required=True)
    sp.add_argument("--asset", default=DEFAULT_ASSET)
    sp.add_argument("--project", default=DEFAULT_PROJECT)
    sp.add_argument("--model", help="model.pkl for the parity probe (default: the path recorded in the manifest)")
    sp.add_argument("--csv", help="The CSV that convert wrote (default: rf_trees_chunks.csv next to the manifest)")
    sp.add_argument("--probes", type=int, default=100, help="Test vectors for the parity probe")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--threshold", type=float, default=0.670)
    sp.add_argument("--no-parity", action="store_true", help="Only compare the table with the manifest")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "selftest": cmd_selftest, "report": cmd_report, "convert": cmd_convert,
        "toy-check": cmd_toy_check, "verify-asset": cmd_verify_asset,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
