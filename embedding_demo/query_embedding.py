#!/usr/bin/env python3
"""Query nearest neighbors from a saved embedding JSON file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="embedding_demo/artifacts/repo_embedding.json")
    p.add_argument("--word", nargs="+", required=True, help="Query word(s). Multiple words will be averaged.")
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def cosine(a: list[float], b: list[float]) -> float:
    ab = 0.0
    aa = 0.0
    bb = 0.0
    for i in range(len(a)):
        av = a[i]
        bv = b[i]
        ab += av * bv
        aa += av * av
        bb += bv * bv
    if aa <= 1e-12 or bb <= 1e-12:
        return 0.0
    return ab / math.sqrt(aa * bb)


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
    vocab: list[str] = payload["vocab"]
    vectors: list[list[float]] = payload["vectors"]
    word_to_id = {w: i for i, w in enumerate(vocab)}

    ids = []
    missing = []
    for w in args.word:
        w = w.lower()
        if w in word_to_id:
            ids.append(word_to_id[w])
        else:
            missing.append(w)
    if missing:
        print(f"[warn] missing words: {missing}")
    if not ids:
        raise SystemExit("No query words found in vocabulary.")

    dim = len(vectors[0])
    qvec = [0.0] * dim
    for idx in ids:
        row = vectors[idx]
        for d in range(dim):
            qvec[d] += row[d]
    inv = 1.0 / len(ids)
    for d in range(dim):
        qvec[d] *= inv

    scored = []
    query_id_set = set(ids)
    for i, token in enumerate(vocab):
        if i in query_id_set:
            continue
        scored.append((cosine(qvec, vectors[i]), token))
    scored.sort(key=lambda x: (-x[0], x[1]))

    print(f"[query] words={args.word} topk={args.topk}")
    for score, token in scored[: args.topk]:
        print(f"{token:20s} {score:.4f}")


if __name__ == "__main__":
    main()
