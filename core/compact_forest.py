"""Lossless, compact re-encoding of a trained scikit-learn RandomForestClassifier.

WHY
    The project's forest (150 fully grown trees, max_depth=None) has ~13.8 M nodes:
    ~1.2 GB in RAM and a 225 MB pickle. That does not fit GitHub's 100 MB file
    limit nor Streamlit Community Cloud's ~2.7 GB RAM ceiling (loading peaks ~2.4 GB).

WHAT
    Prediction only needs, per node: split feature, threshold, right child, and (for
    leaves) the class probabilities. scikit-learn additionally stores ~60 bytes of
    training statistics per node that are never used when predicting.

    This module stores exactly the information prediction needs - same trees, same
    splits, same leaf values - in ~7 bytes per node, and walks the trees with numpy.
    Nothing is approximated: `CompactForest.from_sklearn` raises if the forest has
    any property the encoding relies on (left child == node+1, thresholds exactly
    representable in float16), and train_model.py compares predict/predict_proba
    against scikit-learn on the whole held-out test set before saving.

    Bonus: the saved file is plain numpy, so it does not depend on the scikit-learn
    version (no pickle compatibility problems on the deployment host).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FORMAT_VERSION = 1


class CompactForest:
    def __init__(self, feat, thr, right, roots, leaf_table, classes, n_features):
        self.feat = feat              # int8   (n_nodes,)  -2 => leaf
        self.thr = thr                # float16(n_nodes,)  split threshold (exact)
        self.right = right            # int32  (n_nodes,)  GLOBAL right-child index; leaf => row in leaf_table
        self.roots = roots            # int32  (n_trees,)  global index of each tree's root
        self.leaf_table = leaf_table  # float64(n_unique, n_classes) per-tree class probabilities
        self.classes_ = classes       # int64  (n_classes,)
        self.n_features = int(n_features)
        self._max_steps = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_sklearn(cls, forest) -> "CompactForest":
        feats, thrs, rights, leaf_rows = [], [], [], []
        roots, offset = [], 0

        for est in forest.estimators_:
            t = est.tree_
            left, right, feature, thr = t.children_left, t.children_right, t.feature, t.threshold
            internal = left != -1
            idx = np.nonzero(internal)[0]

            if not np.array_equal(left[idx], idx + 1):
                raise ValueError("Tree layout not depth-first (left child != node+1); cannot encode losslessly")
            thr_i = thr[internal]
            if not np.array_equal(thr_i, thr_i.astype(np.float16).astype(np.float64)):
                raise ValueError("Thresholds are not exactly representable in float16")
            if feature.max() > 127:
                raise ValueError("More than 127 features; int8 feature index not enough")

            n = t.node_count
            f = np.where(internal, feature, -2).astype(np.int8)
            th = np.zeros(n, dtype=np.float16)
            th[internal] = thr_i.astype(np.float16)

            # same normalisation scikit-learn applies in DecisionTreeClassifier.predict_proba
            val = t.value[:, 0, :].astype(np.float64)
            norm = val.sum(axis=1)
            norm[norm == 0.0] = 1.0
            proba = val / norm[:, None]

            r = np.zeros(n, dtype=np.int64)
            r[internal] = right[internal] + offset      # global child index
            leaf_rows.append(proba[~internal])
            rights.append(r)
            feats.append(f)
            thrs.append(th)
            roots.append(offset)
            offset += n

        leaf_all = np.concatenate(leaf_rows)
        table, inverse = np.unique(leaf_all, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

        feat = np.concatenate(feats)
        thr = np.concatenate(thrs)
        right = np.concatenate(rights)
        is_leaf = feat == -2
        right[is_leaf] = inverse                         # leaves store their table row
        if right.max() > np.iinfo(np.int32).max:
            raise ValueError("Forest too large for int32 indices")

        return cls(
            feat=feat,
            thr=thr,
            right=right.astype(np.int32),
            roots=np.asarray(roots, dtype=np.int32),
            leaf_table=table,
            classes=np.asarray(forest.classes_),
            n_features=forest.n_features_in_,
        )

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        # Store the right pointers as "distance from this node" (small numbers
        # compress far better); leaves keep their table index.
        idx = np.arange(len(self.feat), dtype=np.int64)
        is_leaf = self.feat == -2
        stored = self.right.astype(np.int64).copy()
        stored[~is_leaf] -= idx[~is_leaf]
        np.savez_compressed(
            path,
            version=np.int32(FORMAT_VERSION),
            feat=self.feat,
            thr=self.thr.view(np.uint16),
            right_delta=stored.astype(np.int32),
            roots=self.roots,
            leaf_table=self.leaf_table,
            classes=self.classes_,
            n_features=np.int32(self.n_features),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CompactForest":
        z = np.load(path)
        if int(z["version"]) != FORMAT_VERSION:
            raise ValueError("Unsupported compact-forest file version")
        feat = z["feat"]
        stored = z["right_delta"].astype(np.int64)
        is_leaf = feat == -2
        idx = np.arange(len(feat), dtype=np.int64)
        stored[~is_leaf] += idx[~is_leaf]
        return cls(
            feat=feat,
            thr=z["thr"].view(np.float16),
            right=stored.astype(np.int32),
            roots=z["roots"],
            leaf_table=z["leaf_table"],
            classes=z["classes"],
            n_features=int(z["n_features"]),
        )

    # ------------------------------------------------------------------ predict
    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)          # scikit-learn also casts to float32
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got shape {X.shape}")
        n, n_trees = X.shape[0], len(self.roots)
        cur = np.broadcast_to(self.roots, (n, n_trees)).astype(np.int64)
        rows = np.arange(n)[:, None]

        while True:
            f = self.feat[cur]
            active = f >= 0
            if not active.any():
                break
            xv = X[rows, np.where(active, f, 0)]
            go_left = xv <= self.thr[cur]            # same rule as scikit-learn: x <= threshold -> left
            nxt = np.where(go_left, cur + 1, self.right[cur])
            cur = np.where(active, nxt, cur)

        proba = self.leaf_table[self.right[cur]]      # (n, n_trees, n_classes)
        # sum over trees then divide: same arithmetic as RandomForestClassifier.predict_proba
        return proba.sum(axis=1) / n_trees

    def predict(self, X) -> np.ndarray:
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1), axis=0)
