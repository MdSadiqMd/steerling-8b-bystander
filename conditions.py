"""Steering conditions C0-C4 and random-concept controls R1-R5 (first 5 of 8 seeded draws).

Conditions are plain data so analysis and scoring run without Steerling installed;
adapter.Steerling.steering() turns one into a SteeringConfig.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

TARGET = 18956  # Cyberattack tactics and tradecraft
ALPHA = -7.0  # their notebook default (raw injection alpha)
ALPHA_WEAK = -4.0
RELU_STRENGTH = 20.0  # injection_relu default

CONCEPTS_REPO = "guidelabs/steerling"
CONCEPTS_REV = "8e8a481b227ae009ad45289d1253af3b456d9f9b"

RANDOM_SEED = 20261124
N_RANDOM = 8  # drawn
N_CONTROLS = 5  # run: R1-R5, the pre-registered cut for a 2-3 h budget
# Controls must be topic-neutral: no security topics and no alignment-flagged concepts,
# so the noise floor is not itself a safety intervention.
SECURITY = re.compile(
    r"cyber|malware|ransom|hack|secur|vulnerab|exploit|breach|phish|attack|intrusion|botnet"
    r"|virus|trojan|spyware|password|credential|crypt|fraud|scam|surveil|espionage|weapon",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Condition:
    name: str
    concept_ids: tuple[int, ...] = ()  # injection direction = normalised sum; () = no injection
    alpha: float = 0.0
    relu: tuple[tuple[int, float], ...] = ()  # (concept_id, strength) word penalties

    @property
    def is_base(self) -> bool:
        return not self.concept_ids and not self.relu


@cache
def catalog() -> pd.DataFrame:
    path = hf_hub_download(CONCEPTS_REPO, "concept_labels.parquet", revision=CONCEPTS_REV)
    return pd.read_parquet(path)


def group(cid: int) -> tuple[int, ...]:
    """Same members as steerling.ConceptCatalog.group (focal first)."""
    df = catalog()
    gid = df.loc[df.concept_id == cid, "public_group_id"].iloc[0]
    if pd.isna(gid):
        return (cid,)
    members = df.loc[df.public_group_id == gid, "concept_id"].astype(int).tolist()
    return (cid, *(m for m in members if m != cid))


def random_concepts() -> list[int]:
    df = catalog()
    text = df.concept_name.fillna("") + " " + df.concept_description.fillna("") + " " + df.group_name.fillna("")
    target_gid = df.loc[df.concept_id == TARGET, "public_group_id"].iloc[0]
    pool = df[
        (df["head"] == "known")
        & df.is_steerable
        & ~df.is_alignment
        & ~text.str.contains(SECURITY)
        & (df.public_group_id != target_gid)
    ].sort_values("concept_id")
    rng = np.random.default_rng(RANDOM_SEED)
    return [int(c) for c in rng.choice(pool.concept_id.to_numpy(), size=N_RANDOM, replace=False)]


def suppress(name: str, cid: int, alpha: float = ALPHA) -> Condition:
    """Exactly SteeringConfig.injection_relu(cid, group(cid), alpha, None, RELU_STRENGTH)."""
    return Condition(name, group(cid), alpha, ((cid, RELU_STRENGTH),))


@cache
def conditions() -> dict[str, Condition]:
    conds = [
        Condition("C0"),
        suppress("C1", TARGET),
        suppress("C2", TARGET, ALPHA_WEAK),
        Condition("C3", group(TARGET), ALPHA),  # injection only
        Condition("C4", relu=((TARGET, RELU_STRENGTH),)),  # word penalty only
        *(suppress(f"R{i}", cid) for i, cid in enumerate(random_concepts()[:N_CONTROLS], 1)),
    ]
    return {c.name: c for c in conds}


if __name__ == "__main__":
    df = catalog().set_index("concept_id")
    for c in conditions().values():
        focal = c.relu[0][0] if c.relu else (c.concept_ids[0] if c.concept_ids else None)
        label = f"{focal} {df.loc[focal, 'concept_name']} | group {df.loc[focal, 'group_name']}" if focal else "-"
        print(f"{c.name:3} alpha={c.alpha:+.1f} relu={dict(c.relu)} n_dir={len(c.concept_ids):2} {label}")
