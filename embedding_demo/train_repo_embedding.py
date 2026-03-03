#!/usr/bin/env python3
"""Train a tiny word embedding (Skip-gram + Negative Sampling) on local repo text.

Pure Python implementation: no third-party dependencies required.
Default corpus source: .py / .md / .tex files under the repository.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
DEFAULT_EXTS = {".py", ".md", ".tex"}
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".claude",
    "results",
    "embedding_demo",  # avoid training on generated demo outputs unless user wants it
}
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "true",
    "false",
    "none",
    "self",
    "def",
    "class",
    "return",
    "import",
    "args",
    "print",
    "int",
    "float",
    "str",
    "list",
    "dict",
    "if",
    "else",
    "elif",
    "in",
    "to",
    "of",
    "as",
    "is",
    "on",
    "or",
    "not",
    "py",
    "md",
    "tex",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root to scan for corpus files.")
    parser.add_argument("--output", default="embedding_demo/artifacts/repo_embedding.json")
    parser.add_argument("--dim", type=int, default=24, help="Embedding dimension.")
    parser.add_argument("--window", type=int, default=2, help="Context window size.")
    parser.add_argument("--negatives", type=int, default=5, help="Negative samples per positive.")
    parser.add_argument("--steps", type=int, default=12000, help="SGD updates.")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate.")
    parser.add_argument("--min-lr", type=float, default=0.01, help="Final learning rate after decay.")
    parser.add_argument("--min-count", type=int, default=3, help="Minimum token frequency.")
    parser.add_argument("--max-vocab", type=int, default=500, help="Max vocabulary size.")
    parser.add_argument("--max-files", type=int, default=2000, help="Safety limit when scanning files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-stopwords", action="store_true", help="Keep common programming stopwords.")
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument("--topk-demo", type=int, default=6)
    parser.add_argument(
        "--demo-words",
        nargs="*",
        default=["rmha", "mappo", "robot", "communication", "reward", "attention", "policy", "value"],
        help="Words to inspect after training.",
    )
    return parser.parse_args()


def iter_corpus_files(root: Path, max_files: int) -> Iterable[Path]:
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        if path.suffix.lower() not in DEFAULT_EXTS:
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        count += 1
        yield path


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def collect_sentences(root: Path, max_files: int) -> tuple[list[list[str]], Counter, list[str]]:
    sentences: list[list[str]] = []
    counter: Counter = Counter()
    used_files: list[str] = []
    for path in iter_corpus_files(root, max_files):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        used_files.append(str(path))
        for line in content.splitlines():
            toks = tokenize(line)
            if len(toks) < 2:
                continue
            counter.update(toks)
            sentences.append(toks)
    return sentences, counter, used_files


def build_vocab(counter: Counter, min_count: int, max_vocab: int, keep_stopwords: bool) -> tuple[list[str], dict[str, int], dict[str, int]]:
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    if not keep_stopwords:
        items = [(w, c) for w, c in items if w not in STOPWORDS]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[:max_vocab]
    vocab = [w for w, _ in items]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    counts = {w: c for w, c in items}
    return vocab, word_to_id, counts


def sentences_to_ids(sentences: list[list[str]], word_to_id: dict[str, int]) -> list[list[int]]:
    id_sents: list[list[int]] = []
    for sent in sentences:
        ids = [word_to_id[w] for w in sent if w in word_to_id]
        if len(ids) >= 2:
            id_sents.append(ids)
    return id_sents


def build_pairs(id_sents: list[list[int]], window: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for sent in id_sents:
        n = len(sent)
        for i, center in enumerate(sent):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                pairs.append((center, sent[j]))
    return pairs


class NegativeSampler:
    def __init__(self, counts_by_id: list[int], rng: random.Random):
        self.rng = rng
        total = 0.0
        self.cum: list[float] = []
        for c in counts_by_id:
            total += float(c) ** 0.75
            self.cum.append(total)
        self.total = total

    def sample(self, forbid: int) -> int:
        # Retry until the sampled id differs from positive target id.
        while True:
            r = self.rng.random() * self.total
            idx = bisect.bisect_left(self.cum, r)
            if idx != forbid:
                return idx


def sigmoid(x: float) -> float:
    if x < -20.0:
        return 0.0
    if x > 20.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


def cosine(a: list[float], b: list[float]) -> float:
    da = 0.0
    db = 0.0
    ab = 0.0
    for i in range(len(a)):
        av = a[i]
        bv = b[i]
        ab += av * bv
        da += av * av
        db += bv * bv
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return ab / math.sqrt(da * db)


def train_sgns(
    pairs: list[tuple[int, int]],
    counts_by_id: list[int],
    dim: int,
    negatives: int,
    steps: int,
    lr: float,
    min_lr: float,
    seed: int,
    print_every: int,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    rng = random.Random(seed)
    vocab_size = len(counts_by_id)
    bound = 0.5 / dim
    w_in = [[rng.uniform(-bound, bound) for _ in range(dim)] for _ in range(vocab_size)]
    # Small random init speeds up learning in this pure-Python demo.
    w_out = [[rng.uniform(-bound, bound) for _ in range(dim)] for _ in range(vocab_size)]
    sampler = NegativeSampler(counts_by_id, rng)

    avg_losses: list[float] = []
    running_loss = 0.0
    start = time.time()

    for step in range(1, steps + 1):
        progress = (step - 1) / max(1, steps - 1)
        cur_lr = lr + (min_lr - lr) * progress

        center_id, pos_id = pairs[rng.randrange(len(pairs))]
        center_vec = w_in[center_id]
        grad_center = [0.0] * dim
        sample_loss = 0.0

        # Positive pair + negative pairs in one fused update.
        targets = [(pos_id, 1)]
        for _ in range(negatives):
            targets.append((sampler.sample(pos_id), 0))

        for out_id, label in targets:
            out_vec = w_out[out_id]
            score = dot(center_vec, out_vec)
            prob = sigmoid(score)
            # Avoid log(0) in the loss report.
            prob = min(max(prob, 1e-8), 1.0 - 1e-8)
            g = cur_lr * (label - prob)

            if label == 1:
                sample_loss += -math.log(prob)
            else:
                sample_loss += -math.log(1.0 - prob)

            for d in range(dim):
                out_old = out_vec[d]
                grad_center[d] += g * out_old
                out_vec[d] = out_old + g * center_vec[d]

        for d in range(dim):
            center_vec[d] += grad_center[d]

        running_loss += sample_loss

        if step % print_every == 0 or step == steps:
            avg = running_loss / min(print_every, step)
            avg_losses.append(avg)
            elapsed = time.time() - start
            print(f"[train] step={step}/{steps} lr={cur_lr:.4f} avg_loss={avg:.5f} elapsed={elapsed:.1f}s")
            running_loss = 0.0

    return w_in, w_out, avg_losses


def combined_vectors(w_in: list[list[float]], w_out: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(len(w_in)):
        a = w_in[i]
        b = w_out[i]
        out.append([a[d] + b[d] for d in range(len(a))])
    return out


def nearest_neighbors(vectors: list[list[float]], vocab: list[str], word_to_id: dict[str, int], query: str, topk: int) -> list[tuple[str, float]]:
    q = query.lower()
    if q not in word_to_id:
        return []
    qid = word_to_id[q]
    qvec = vectors[qid]
    scored: list[tuple[float, str]] = []
    for i, token in enumerate(vocab):
        if i == qid:
            continue
        scored.append((cosine(qvec, vectors[i]), token))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(token, score) for score, token in scored[:topk]]


def save_model(
    output_path: Path,
    vocab: list[str],
    counts_by_id: list[int],
    vectors: list[list[float]],
    config: dict,
    used_files: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "repo_sgns_v1",
        "config": config,
        "vocab": vocab,
        "counts": counts_by_id,
        "vectors": [[round(v, 6) for v in row] for row in vectors],
        "used_files": used_files,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    root = Path(args.root).resolve()

    print(f"[data] scanning files under: {root}")
    sentences, counter, used_files = collect_sentences(root, args.max_files)
    print(f"[data] files={len(used_files)} raw_sentences={len(sentences)} raw_vocab={len(counter)}")

    vocab, word_to_id, counts_map = build_vocab(counter, args.min_count, args.max_vocab, args.keep_stopwords)
    if len(vocab) < 10:
        raise SystemExit("Vocabulary too small. Lower --min-count or increase corpus.")
    id_sents = sentences_to_ids(sentences, word_to_id)
    pairs = build_pairs(id_sents, args.window)
    counts_by_id = [counts_map[w] for w in vocab]

    if not pairs:
        raise SystemExit("No training pairs generated. Adjust token filtering/window settings.")

    total_tokens = sum(len(s) for s in id_sents)
    print(
        f"[data] vocab={len(vocab)} filtered_sentences={len(id_sents)} "
        f"filtered_tokens={total_tokens} pairs={len(pairs)}"
    )
    print(f"[data] top15={[(w, counts_map[w]) for w in vocab[:15]]}")

    # Shuffle pairs once for a bit more variety in the first steps.
    rng.shuffle(pairs)

    w_in, w_out, avg_losses = train_sgns(
        pairs=pairs,
        counts_by_id=counts_by_id,
        dim=args.dim,
        negatives=args.negatives,
        steps=args.steps,
        lr=args.lr,
        min_lr=args.min_lr,
        seed=args.seed,
        print_every=args.print_every,
    )

    vectors = combined_vectors(w_in, w_out)
    save_model(
        output_path=Path(args.output),
        vocab=vocab,
        counts_by_id=counts_by_id,
        vectors=vectors,
        config=vars(args),
        used_files=used_files,
    )
    print(f"[save] model written to {args.output}")

    print("[demo] nearest neighbors")
    demo_words = [w for w in args.demo_words if w.lower() in word_to_id]
    if not demo_words:
        # Pick a few high-frequency domain tokens that are not too generic.
        candidates = [w for w in vocab if len(w) >= 4][:5]
        demo_words = candidates
    for word in demo_words:
        nns = nearest_neighbors(vectors, vocab, word_to_id, word, args.topk_demo)
        if not nns:
            continue
        formatted = ", ".join(f"{w}:{s:.3f}" for w, s in nns)
        print(f"  {word:>14s} -> {formatted}")

    if avg_losses:
        print(f"[done] avg_loss(last_log)={avg_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
