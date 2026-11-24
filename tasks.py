"""Prompt sets (pinned sources) and the deterministic scorers.

Harmful prompts are downloaded at run time into .cache/ and never committed.
Every task returns items {"id", "prompt", **meta}; the same ids are used in every condition.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from auditkit.redteam.detectors.builtin import RefusalDetector
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent
CACHE = ROOT / ".cache"
DATA = ROOT / "data"
SAMPLE_SEED = 0

HARMBENCH = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
XSTEST = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/"
    "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/xstest_prompts.csv"
)
HF = {  # repo -> pinned dataset revision
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}

MMLU_TMPL = "{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer with only the letter of the correct option."
GSM_SUFFIX = "\nSolve step by step and end with the final number."


@dataclass(frozen=True)
class Task:
    name: str
    layer: str  # target | defensive | word | far | concept
    kind: str  # gen | letter | shift
    n: int | None  # None = all items
    greedy: bool = False
    max_new_tokens: int = 128


TASKS = {
    t.name: t
    for t in [
        Task("harm_cyber", "target", "gen", None),
        Task("defensive", "defensive", "gen", 50),
        Task("mmlu_computer_security", "defensive", "letter", 100),
        Task("words", "word", "gen", 60),
        Task("xstest_safe", "far", "gen", 100),
        Task("harm_other", "far", "gen", 60),
        Task("mmlu_high_school_biology", "far", "letter", 100),
        Task("mmlu_philosophy", "far", "letter", 100),
        Task("mmlu_high_school_world_history", "far", "letter", 100),
        Task("gsm8k", "far", "gen", 60, greedy=True, max_new_tokens=256),
        Task("ifeval", "far", "gen", 60, greedy=True, max_new_tokens=256),
        Task("concept_shift", "concept", "shift", 200),
    ]
}


def item_seed(task: str, item_id: str) -> int:
    """Per-item seed, identical across conditions (paired design)."""
    return zlib.crc32(f"{task}:{item_id}".encode()) & 0x7FFFFFFF


def _fetch(url: str) -> str:
    path = CACHE / Path(url).name
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_text()


def _hf(repo: str, filename: str) -> str:
    return hf_hub_download(repo, filename, repo_type="dataset", revision=HF[repo])


def _sample(rows: list[dict], n: int | None) -> list[dict]:
    if n is None or n >= len(rows):
        return rows
    idx = np.random.default_rng(SAMPLE_SEED).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(idx)]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@cache
def _harmbench() -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(_fetch(HARMBENCH))))
    out = []
    for r in rows:
        if r["FunctionalCategory"] == "copyright":
            continue
        prompt = r["Behavior"]
        if r["ContextString"].strip():  # HarmBench's own contextual format
            prompt = f"{r['ContextString']}\n\n---\n\n{r['Behavior']}"
        out.append({"id": r["BehaviorID"], "prompt": prompt, "category": r["SemanticCategory"]})
    return out


def _mmlu(subject: str, n: int | None) -> list[dict]:
    df = pd.read_parquet(_hf("cais/mmlu", f"{subject}/test-00000-of-00001.parquet"))
    rows = [
        {
            "id": f"{subject}:{i}",
            "prompt": MMLU_TMPL.format(q=r.question, a=r.choices[0], b=r.choices[1], c=r.choices[2], d=r.choices[3]),
            "gold": "ABCD"[int(r.answer)],
        }
        for i, r in enumerate(df.itertuples())
    ]
    return _sample(rows, n)


def _gsm8k(n: int | None) -> list[dict]:
    df = pd.read_parquet(_hf("openai/gsm8k", "main/test-00000-of-00001.parquet"))
    rows = [
        {"id": f"gsm8k:{i}", "prompt": r.question + GSM_SUFFIX, "gold": float(r.answer.split("####")[-1].replace(",", ""))}
        for i, r in enumerate(df.itertuples())
    ]
    return _sample(rows, n)


def _ifeval(n: int | None) -> list[dict]:
    rows = _jsonl(Path(_hf("google/IFEval", "ifeval_input_data.jsonl")))
    # Drop items asking for more words than the 256-token budget can hold: they fail in every condition.
    rows = [
        r
        for r in rows
        if set(r["instruction_id_list"]) <= IFEVAL_SUPPORTED
        and not any((k or {}).get("relation") == "at least" and k["num_words"] > 150 for k in r["kwargs"])
    ]
    return _sample(
        [{"id": f"ifeval:{r['key']}", "prompt": r["prompt"], "inst": r["instruction_id_list"], "kw": r["kwargs"]} for r in rows],
        n,
    )


def _wikitext(n: int, words: int = 120) -> list[dict]:
    """Neutral passages: consecutive non-heading WikiText-2 test lines, ~`words` words each."""
    df = pd.read_parquet(_hf("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet"))
    lines = [t.strip() for t in df.text if t.strip() and not t.strip().startswith("=")]
    out, buf = [], []
    for line in lines:
        buf += line.split()
        if len(buf) >= words:
            out.append({"id": f"wikitext:{len(out)}", "prompt": " ".join(buf[:words])})
            buf = []
            if len(out) == n:
                break
    return out


def load(name: str) -> list[dict]:
    t = TASKS[name]
    if name == "harm_cyber":
        return _sample([r for r in _harmbench() if r["category"] == "cybercrime_intrusion"], t.n)
    if name == "harm_other":
        return _sample([r for r in _harmbench() if r["category"] != "cybercrime_intrusion"], t.n)
    if name == "xstest_safe":
        rows = [r for r in csv.DictReader(io.StringIO(_fetch(XSTEST))) if r["label"] == "safe"]
        return _sample([{"id": f"xstest:{r['id']}", "prompt": r["prompt"], "type": r["type"]} for r in rows], t.n)
    if name in ("defensive", "words"):
        return _sample(_jsonl(DATA / f"{name}.jsonl"), t.n)
    if name.startswith("mmlu_"):
        return _mmlu(name.removeprefix("mmlu_"), t.n)
    if name == "gsm8k":
        return _gsm8k(t.n)
    if name == "ifeval":
        return _ifeval(t.n)
    if name == "concept_shift":
        return _wikitext(t.n)
    raise KeyError(name)


_refusal = RefusalDetector()


def refused(prompt: str, response: str) -> bool:
    return bool(_refusal.detect(prompt, response).detected)


def last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].rstrip("."))
    except ValueError:
        return None


def keyword_hits(text: str, keywords: list[str]) -> float:
    low = text.lower()
    return sum(k.lower() in low for k in keywords) / len(keywords)


def word_used(text: str, stem: str) -> bool:
    return re.search(rf"\b{re.escape(stem)}", text, re.IGNORECASE) is not None


IFEVAL_SUPPORTED = {
    "punctuation:no_comma",
    "change_case:english_lowercase",
    "change_case:english_capital",
    "length_constraints:number_words",
    "detectable_format:number_bullet_lists",
    "startend:quotation",
    "detectable_format:json_format",
    "keywords:existence",
    "keywords:forbidden_words",
    "detectable_format:title",
    "detectable_format:number_highlighted_sections",
    "startend:end_checker",
    "detectable_content:postscript",
}


def ifeval_check(inst: str, kw: dict, out: str) -> bool:
    """Offline checks for the IFEval instruction types above (ported from bystander-circuits)."""
    words = re.findall(r"\w+", out)
    match inst:
        case "punctuation:no_comma":
            return "," not in out
        case "change_case:english_lowercase":
            return out == out.lower()
        case "change_case:english_capital":
            return out == out.upper()
        case "length_constraints:number_words":
            n = kw["num_words"]
            return len(words) >= n if kw["relation"] == "at least" else len(words) < n
        case "detectable_format:number_bullet_lists":
            return len(re.findall(r"^\s*[\*\-]\s", out, flags=re.MULTILINE)) == kw["num_bullets"]
        case "startend:quotation":
            o = out.strip()
            return len(o) > 1 and o[0] == '"' and o[-1] == '"'
        case "detectable_format:json_format":
            try:
                json.loads(out.strip().strip("`").removeprefix("json").strip())
                return True
            except ValueError:
                return False
        case "keywords:existence":
            return all(k.lower() in out.lower() for k in kw["keywords"])
        case "keywords:forbidden_words":
            return not any(k.lower() in out.lower() for k in kw["forbidden_words"])
        case "detectable_format:title":
            return re.search(r"<<[^>]+>>", out) is not None
        case "detectable_format:number_highlighted_sections":
            return len(re.findall(r"\*[^\*\n]+\*", out)) >= kw["num_highlights"]
        case "startend:end_checker":
            return out.strip().endswith(kw["end_phrase"].strip())
        case "detectable_content:postscript":
            return kw["postscript_marker"].lower() in out.lower()
    raise KeyError(inst)


def ifeval_score(item: dict, out: str) -> bool:
    """Strict prompt-level accuracy: every instruction passes."""
    return all(ifeval_check(i, k or {}, out) for i, k in zip(item["inst"], item["kw"], strict=True))


if __name__ == "__main__":
    for name, t in TASKS.items():
        items = load(name)
        print(f"{name:32} {t.layer:9} {t.kind:6} n={len(items)}")
