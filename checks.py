"""Day-1 GPU sanity checks. Run once before the full run.

    uv run --extra gpu python checks.py

1. Conditions match the package presets (C1 == SteeringConfig.injection_relu with the group).
2. Chat prompt encodes the header/eot markers as special tokens.
3. Word list for H3: where each penalised word ranks among 18956's top-aligned tokens.
4. Letter scoring: top-5 tokens at the answer slot, base MMLU accuracy, stability across mask counts.
5. Base reproducibility: same seed twice gives the same text.
6. Their notebook example (Books and publishing), base vs suppressed, same sampling.
7. Timing for one sampled generation.
"""

from __future__ import annotations

import time

from steerling import SteeringConfig

from adapter import Steerling
from conditions import TARGET, Condition, conditions, group, suppress
from tasks import load


def main() -> None:
    m = Steerling()
    conds = conditions()

    print("== 1. presets")
    c1 = m.steering(conds["C1"])
    ref = SteeringConfig.injection_relu(TARGET, group_concept_ids=list(group(TARGET)), mai_lm_target=-7.0)
    assert c1 == ref, (c1, ref)
    print("C1 == injection_relu(18956, group, -7.0): ok")

    print("\n== 2. chat encoding")
    ids = m.tok.encode(m.chat("hi"), add_special_tokens=False)
    specials = {100281: "start_header", 100282: "end_header", 100283: "eot"}
    print({name: ids.count(i) for i, name in specials.items()}, "(expect 3, 3, 2)")

    print("\n== 3. top-aligned tokens of 18956")
    top = m.concept_top_tokens(TARGET, 300)
    ranks = {tok.strip().lower(): r for r, (tok, _) in reversed(list(enumerate(top)))}
    print([t for t, _ in top[:60]])
    words = sorted({it["word"] for it in load("words")})
    for w in words:
        hits = sorted((r, t) for t, r in ranks.items() if t.startswith(w))
        print(f"  {w:8} rank {hits[0][0] if hits else '>300':>4}  {[t for _, t in hits[:4]]}")

    print("\n== 4. letter scoring")
    items = load("mmlu_computer_security")
    for it in items[:3]:
        print(it["gold"], m.answer_slot_top(it["prompt"], conds["C0"]))
    acc, agree = {}, {}
    for n_mask in (1, 64, 128):
        preds = []
        for it in items:
            s = m.letter_logits(it["prompt"], conds["C0"], n_mask=n_mask)
            preds.append(max(s, key=s.get))
        acc[n_mask] = sum(p == it["gold"] for p, it in zip(preds, items, strict=True)) / len(items)
        agree[n_mask] = preds
    print("base accuracy by mask count:", acc)
    print("agreement 64 vs 1:", sum(a == b for a, b in zip(agree[64], agree[1], strict=True)) / len(items))

    print("\n== 4b. answer prefix (base only; pick the best, set adapter.ANSWER_PREFIX)")
    subjects = {s: load(f"mmlu_{s}") for s in ("computer_security", "philosophy", "high_school_biology")}
    for prefix in ("", "Answer:", "The answer is", "The correct option is"):
        accs = {}
        for subj, its in subjects.items():
            hits = 0
            for it in its:
                s = m.letter_logits(it["prompt"], conds["C0"], prefix=prefix)
                hits += max(s, key=s.get) == it["gold"]
            accs[subj] = hits / len(its)
        print(f"  {prefix!r:26} {accs}  top-5: {m.answer_slot_top(items[0]['prompt'], conds['C0'], prefix=prefix)}")

    print("\n== 5. reproducibility")
    a = m.generate("Name three uses of a hash map.", conds["C0"], seed=1)
    b = m.generate("Name three uses of a hash map.", conds["C0"], seed=1)
    print("identical:", a == b)

    print("\n== 6. notebook example: 20353 Books and publishing")
    desc = (
        "Topic about (publish, publisher, publication, Publishing, .pub, _publish), industry infrastructure "
        "(imprint, ISBN, Kindle, ePub, Penguin, Harper, Barnes & Noble, Ingram...)"
    )
    prompt = f"We are discussing about Books and publishing: {desc}\nTell me something interesting."
    print("base:", m.generate(prompt, Condition("C0"), seed=0))
    print("suppressed:", m.generate(prompt, suppress("books", 20353), seed=0))

    print("\n== 7. timing")
    t = time.time()
    m.generate("Explain how a refrigerator works.", conds["C1"], seed=2)
    print(f"one 128-token steered generation: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
