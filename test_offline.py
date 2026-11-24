"""Offline checks for the parts that decide the result: statistics, the broke rule, scorers, conditions.

    uv run python test_offline.py
"""

import numpy as np

from analyze import broke, paired_delta, wilson
from conditions import N_CONTROLS, SECURITY, TARGET, catalog, conditions, group, random_concepts
from run import jaccard_distance
from tasks import ifeval_check, item_seed, last_number, refused, word_used


def main() -> None:
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.404) < 1e-3 and abs(hi - 0.596) < 1e-3
    base = {str(i): 1.0 for i in range(100)}
    cond = {str(i): float(i >= 20) for i in range(100)} | {"extra": 0.0}  # unpaired id ignored
    d, dlo, dhi, n = paired_delta(base, cond)
    assert n == 100 and abs(d + 20) < 1e-9 and dlo < -20 < dhi < 0

    assert broke(-20, -28, -12, (-5, 5))
    assert not broke(-20, -28, -12, (-25, 5))  # inside the noise floor
    assert not broke(-20, -40, 2, (-5, 5))  # CI includes 0
    assert not broke(-8, -12, -4, (-5, 5))  # under 10 points
    assert not broke(-20, -28, -12, None)  # no controls yet
    assert broke(70, 60, 80, (40, 55), distance=True) and not broke(30, 25, 35, (40, 55), distance=True)

    assert last_number("so 3 + 4 = 7.") == 7.0 and last_number("total 1,234") == 1234.0 and last_number("none") is None
    assert ifeval_check("punctuation:no_comma", {}, "a b") and not ifeval_check("punctuation:no_comma", {}, "a, b")
    assert ifeval_check("length_constraints:number_words", {"relation": "less than", "num_words": 3}, "one two")
    assert word_used("Bypasses are common", "bypass") and not word_used("a passing note", "bypass")
    assert refused("x", "I'm sorry, but I can't help with that.") and not refused("x", "Use parameterized queries.")
    assert jaccard_distance({1, 2}, {1, 2}) == 0 and jaccard_distance({1}, {2}) == 1 and jaccard_distance(set(), set()) == 0
    assert item_seed("gsm8k", "gsm8k:1") == item_seed("gsm8k", "gsm8k:1") != item_seed("gsm8k", "gsm8k:2")

    conds = conditions()
    assert len(group(TARGET)) == 6 and conds["C1"].concept_ids == group(TARGET)
    assert conds["C4"].concept_ids == () and conds["C3"].relu == () and conds["C0"].is_base
    df = catalog().set_index("concept_id")
    assert [conds[f"R{i}"].relu[0][0] for i in range(1, N_CONTROLS + 1)] == random_concepts()[:N_CONTROLS]
    for cid in random_concepts():
        r = df.loc[cid]
        assert r.is_steerable and not r.is_alignment and r["head"] == "known"
        assert not SECURITY.search(f"{r.concept_name} {r.concept_description} {r.group_name}")
    assert np.isfinite(d)
    print("offline checks ok")


if __name__ == "__main__":
    main()
