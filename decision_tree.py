import numpy as np
from collections import Counter


def find_best_split(feature_vector: np.ndarray, target: np.ndarray) -> tuple:
    feature_vector = np.asarray(feature_vector, dtype=float)
    target = np.asarray(target, dtype=int)
    n = len(feature_vector)

    if n <= 1:
        p = np.bincount(target, minlength=1) / max(n, 1)
        return None, None, feature_vector[0], 1.0 - float((p ** 2).sum())

    order = np.argsort(feature_vector)
    x_sorted = feature_vector[order]
    y_sorted = target[order]
    n_classes = int(target.max()) + 1

    indicators = np.zeros((n, n_classes))
    indicators[np.arange(n), y_sorted] = 1.0
    cum_counts = np.cumsum(indicators, axis=0)

    split_mask = x_sorted[:-1] != x_sorted[1:]
    if not split_mask.any():
        p = cum_counts[-1] / n
        return None, None, x_sorted[0], 1.0 - float((p ** 2).sum())

    positions = np.where(split_mask)[0]
    left_counts = cum_counts[positions]
    left_n = (positions + 1).astype(float)
    right_n = n - left_n
    right_counts = cum_counts[-1] - left_counts

    gini_left = 1.0 - np.sum((left_counts / left_n[:, None]) ** 2, axis=1)
    gini_right = 1.0 - np.sum((right_counts / right_n[:, None]) ** 2, axis=1)
    weighted_gini = (left_n * gini_left + right_n * gini_right) / n

    best = int(np.argmin(weighted_gini))
    best_pos = positions[best]
    threshold = 0.5 * (x_sorted[best_pos] + x_sorted[best_pos + 1])

    return int(best_pos), None, threshold, float(weighted_gini[best])


class DecisionTree:
    def __init__(
        self,
        feature_types: list[str],
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        if any(ft not in ("real", "categorical") for ft in feature_types):
            raise ValueError("Unknown feature type")
        self._tree: dict = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._feature_importances = np.zeros(len(feature_types))

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = np.bincount(y) / len(y)
        return 1.0 - float((p ** 2).sum())

    def _fit_node(self, sub_X: np.ndarray, sub_y: np.ndarray, node: dict, depth: int = 0):
        n_objects = len(sub_y)

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = int(sub_y[0])
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        if n_objects < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        parent_gini = self._gini(sub_y)
        feature_best = None
        threshold_best = None
        gini_best = None
        split_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                values = sub_X[:, feature]
                counts = Counter(values)
                clicks = Counter(values[sub_y == 1])
                ratio = {}
                for key, total_count in counts.items():
                    ratio[key] = float(clicks.get(key, 0)) / total_count if total_count > 0 else 0.0
                sorted_categories = [k for k, _ in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[val] for val in values])
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or gini < gini_best:
                split = feature_vector < threshold
                left_size = int(split.sum())
                right_size = n_objects - left_size

                if left_size < self._min_samples_leaf or right_size < self._min_samples_leaf:
                    continue
                if left_size == 0 or right_size == 0:
                    continue

                feature_best = feature
                gini_best = gini
                split_best = split

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = int(Counter(sub_y).most_common(1)[0][0])
            return

        self._feature_importances[feature_best] += (parent_gini - gini_best) * n_objects

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth + 1)

    def _predict_node(self, x: np.ndarray, node: dict) -> int:
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            child = "left_child" if x[feature] < node["threshold"] else "right_child"
        elif self._feature_types[feature] == "categorical":
            child = "left_child" if x[feature] in node["categories_split"] else "right_child"
        else:
            raise ValueError(f"Unknown feature type: {self._feature_types[feature]}")

        return self._predict_node(x, node[child])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        X, y = np.asarray(X), np.asarray(y)
        self._feature_importances = np.zeros(X.shape[1])
        self._tree = {}
        self._fit_node(X, y, self._tree)
        total = self._feature_importances.sum()
        if total > 0:
            self._feature_importances /= total
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._feature_importances.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_node(x, self._tree) for x in np.asarray(X)])
