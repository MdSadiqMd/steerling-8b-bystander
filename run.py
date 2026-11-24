"""Run conditions over tasks on the GPU -> results/raw/<cond>/<task>.jsonl (resumable).

    uv run --extra gpu python run.py --cond C0 --limit 20          # smoke test
    uv run --extra gpu python run.py                                # everything, C0 first

Raw files hold generations (including replies to harmful prompts) and stay out of git;
score.py turns them into per-item scores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from conditions import TARGET, conditions, group
from tasks import TASKS, item_seed, load

RAW = Path(__file__).parent / "results/raw"


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def jaccard_distance(a: set[int], b: set[int]) -> float:
    union = a | b
    return 1 - len(a & b) / len(union) if union else 0.0


def concept_shift(m, item: dict, cond, base_cache: dict) -> float:
    """Mean top-k Jaccard distance from base over all positions, ignoring the steered concepts."""
    seed = item_seed("concept_shift", item["id"])
    if item["id"] not in base_cache:
        base_cache[item["id"]] = m.concept_topk(item["prompt"], conditions()["C0"], seed).tolist()
    base = base_cache[item["id"]]
    steered = m.concept_topk(item["prompt"], cond, seed).tolist()
    skip = set(group(TARGET)) | set(cond.concept_ids) | {c for c, _ in cond.relu}
    dists = [jaccard_distance(set(b) - skip, set(s) - skip) for b, s in zip(base, steered, strict=True)]
    return sum(dists) / len(dists)


def run_item(m, task, item: dict, cond, base_cache: dict) -> dict:
    if task.kind == "letter":
        scores = m.letter_logits(item["prompt"], cond)
        return {"pred": max(scores, key=scores.get), "letter_logits": scores}
    if task.kind == "shift":
        return {"jaccard_dist": concept_shift(m, item, cond, base_cache)}
    seed = item_seed(task.name, item["id"])
    return {"output": m.generate(item["prompt"], cond, seed, task.greedy, task.max_new_tokens)}


def main() -> None:
    all_conds = conditions()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default=",".join(all_conds), help="comma-separated, e.g. C0,C1")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None, help="first N items per task (smoke test)")
    args = ap.parse_args()

    from adapter import Steerling  # GPU-only import

    m = Steerling()
    base_cache: dict = {}
    for cname in args.cond.split(","):
        cond = all_conds[cname]
        for tname in args.tasks.split(","):
            task = TASKS[tname]
            items = load(tname)[: args.limit]
            path = RAW / cname / f"{tname}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            done = done_ids(path)
            todo = [it for it in items if it["id"] not in done]
            t0 = time.time()
            with path.open("a") as f:
                for it in todo:
                    t = time.time()
                    rec = {"id": it["id"], **run_item(m, task, it, cond, base_cache), "seconds": round(time.time() - t, 3)}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            dt = time.time() - t0
            print(f"{cname} {tname:32} {len(todo):4} new / {len(items)} items  {dt:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
