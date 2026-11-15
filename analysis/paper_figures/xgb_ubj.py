"""Read the gain importances of the E1 XGBoost models without the XGBoost runtime.

Why: the pinned xgboost 3.2.0 is a CUDA build. With the GPU hidden
(``CUDA_VISIBLE_DEVICES=""``, the rule for every post-hoc step) it raises
``cudaErrorNoDevice`` when a multi-class booster is unpickled, even for a model
trained with ``device="cpu"`` (see ``tests/conftest.py::xgb_fits``). The
feature-importance figures only need the trees, so the pickled booster is not
handed to libxgboost: :func:`booster_without_runtime` swaps
``Booster.__setstate__`` during ``joblib.load`` for one that keeps the raw
serialised bytes (UBJSON) in a pure-Python stand-in, :class:`UBJBooster`.

``UBJBooster.get_score`` reproduces ``Booster.get_score`` for the tree
importance types ``weight``, ``gain``, ``total_gain``, ``cover`` and
``total_cover`` from the decoded trees (xgboost ``GBTree::FeatureScore``: per
split node, ``weight`` counts the splits, ``gain`` / ``cover`` add the node's
``loss_changes`` / ``sum_hessian`` and are divided by the split count). The
sklearn property ``XGBModel.feature_importances_`` then works unchanged on top
of it, so ``models._extract_feature_importances_from_pipeline`` is reused as is.
The equality with libxgboost is tested on a model it can still fit with the
GPU hidden (single-output objective).
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional

import numpy as np

_NUMERIC = {b"i": ">i1", b"U": ">u1", b"I": ">i2", b"l": ">i4", b"L": ">i8", b"d": ">f4", b"D": ">f8"}


class UBJDecodeError(ValueError):
    pass


class _Reader:
    def __init__(self, buf: bytes):
        self.b = memoryview(bytes(buf))
        self.i = 0

    def _take(self, n: int) -> memoryview:
        if self.i + n > len(self.b):
            raise UBJDecodeError("truncated UBJSON")
        out = self.b[self.i:self.i + n]
        self.i += n
        return out

    def _peek(self) -> bytes:
        while self.b[self.i:self.i + 1].tobytes() == b"N":   # no-op markers
            self.i += 1
        return self.b[self.i:self.i + 1].tobytes()

    def _marker(self) -> bytes:
        m = self._peek()
        self.i += 1
        return m

    def _int(self) -> int:
        m = self._marker()
        if m not in (b"i", b"U", b"I", b"l", b"L"):
            raise UBJDecodeError(f"expected an integer marker, got {m!r}")
        return int(self.value(m))

    def value(self, marker: Optional[bytes] = None):
        m = marker if marker is not None else self._marker()
        if m in _NUMERIC:
            dt = np.dtype(_NUMERIC[m])
            return np.frombuffer(self._take(dt.itemsize), dtype=dt)[0].item()
        if m in (b"S", b"H"):
            return self._take(self._int()).tobytes().decode("utf-8")
        if m == b"C":
            return self._take(1).tobytes().decode("ascii")
        if m == b"T":
            return True
        if m == b"F":
            return False
        if m == b"Z":
            return None
        if m == b"[":
            return self._array()
        if m == b"{":
            return self._object()
        raise UBJDecodeError(f"unknown UBJSON marker {m!r} at byte {self.i - 1}")

    def _header(self):
        typ = count = None
        if self._peek() == b"$":
            self.i += 1
            typ = self._take(1).tobytes()
            if self._peek() != b"#":
                raise UBJDecodeError("typed container without a count")
        if self._peek() == b"#":
            self.i += 1
            count = self._int()
        return typ, count

    def _array(self):
        typ, count = self._header()
        if count is not None:
            if typ in _NUMERIC:
                dt = np.dtype(_NUMERIC[typ])
                return np.frombuffer(self._take(count * dt.itemsize), dtype=dt).astype(dt.newbyteorder("="))
            return [self.value(typ) for _ in range(count)]
        out = []
        while self._peek() != b"]":
            out.append(self.value())
        self.i += 1
        return out

    def _key(self) -> str:
        return self._take(self._int()).tobytes().decode("utf-8")

    def _object(self):
        typ, count = self._header()
        if count is not None:
            return {self._key(): self.value(typ) for _ in range(count)}
        out = {}
        while self._peek() != b"}":
            k = self._key()
            out[k] = self.value()
        self.i += 1
        return out


def decode_ubjson(buf: bytes):
    r = _Reader(buf)
    out = r.value()
    if r.i != len(r.b):
        raise UBJDecodeError(f"{len(r.b) - r.i} trailing bytes after the UBJSON value")
    return out


def _model(doc: dict) -> dict:
    """The model part of a serialised booster (``{"Config", "Model"}``) or of a saved model."""
    return doc["Model"] if "Model" in doc else doc


def feature_score(doc: dict, importance_type: str = "gain") -> Dict[str, float]:
    """``Booster.get_score(importance_type)`` from a decoded model (tree boosters)."""
    learner = _model(doc)["learner"]
    gb = learner["gradient_booster"]
    if gb.get("name") not in ("gbtree", "dart"):
        raise ValueError(f"only tree boosters are supported, got {gb.get('name')!r}")
    trees = (gb["gbtree"] if gb.get("name") == "dart" else gb)["model"]["trees"]
    n_features = int(learner["learner_model_param"]["num_feature"])
    names = list(learner.get("feature_names") or []) or [f"f{i}" for i in range(n_features)]
    counts = np.zeros(n_features)
    total = np.zeros(n_features)
    stat = {"weight": None, "gain": "loss_changes", "total_gain": "loss_changes",
            "cover": "sum_hessian", "total_cover": "sum_hessian"}
    if importance_type not in stat:
        raise ValueError(f"unsupported importance_type {importance_type!r}")
    for tree in trees:
        left = np.asarray(tree["left_children"])
        split = np.asarray(tree["split_indices"])[left != -1].astype(np.int64)
        np.add.at(counts, split, 1.0)
        if stat[importance_type]:
            np.add.at(total, split, np.asarray(tree[stat[importance_type]], dtype=float)[left != -1])
    if importance_type == "weight":
        score = counts
    elif importance_type.startswith("total_"):
        score = total
    else:
        score = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
    return {names[i]: float(score[i]) for i in range(n_features) if counts[i] > 0}


class UBJBooster:
    """Pure-Python stand-in for an unpickled ``xgboost.Booster`` (read-only)."""

    def __init__(self, raw: Optional[bytes] = None):
        self._raw = bytes(raw) if raw is not None else None
        self._doc = None

    @property
    def doc(self) -> dict:
        if self._doc is None:
            if self._raw is None:
                raise ValueError("booster without serialised bytes")
            self._doc = decode_ubjson(self._raw)
        return self._doc

    @property
    def feature_names(self):
        names = _model(self.doc)["learner"].get("feature_names") or []
        return list(names) or None

    def num_features(self) -> int:
        return int(_model(self.doc)["learner"]["learner_model_param"]["num_feature"])

    def get_score(self, fmap: str = "", importance_type: str = "weight") -> Dict[str, float]:
        if fmap:
            raise ValueError("feature maps are not supported")
        return feature_score(self.doc, importance_type)


def _setstate_without_runtime(self, state) -> None:
    handle = state.get("handle")
    rest = {k: v for k, v in state.items() if k != "handle"}
    self.__class__ = UBJBooster
    self.__dict__.update(rest)
    self._raw = bytes(handle) if handle is not None else None
    self._doc = None


@contextlib.contextmanager
def booster_without_runtime():
    """While active, unpickled ``xgboost.Booster`` objects become :class:`UBJBooster`."""
    import xgboost
    original = xgboost.core.Booster.__setstate__
    xgboost.core.Booster.__setstate__ = _setstate_without_runtime
    try:
        yield
    finally:
        xgboost.core.Booster.__setstate__ = original
