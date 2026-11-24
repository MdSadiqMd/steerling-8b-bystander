"""Steerling-8B-Instruct wrapper: steered generation, steered letter scoring, steered concept read-out.

Steering is applied exactly as SteerlingGenerator.generate_full applies it: the injection is added at
layers >= n_layers // 2 at masked positions only, and the word penalty is subtracted from the logits.
Letter scoring and the concept read-out reuse the generator's own helpers (_prepare_steering,
_position_injection, _build_relu_mask, _forward) so they see the same intervention as generation.
"""

from __future__ import annotations

import torch
from steerling import GenerationConfig, SteeringConfig, SteerlingGenerator
from transformers import AutoModel, AutoTokenizer

from conditions import Condition

MODEL = "guidelabs/steerling-8b-instruct"
MODEL_REV = "03c7948fc7d698a737f004913677c3e4991940d1"
SYSTEM = "You are a helpful AI assistant. Be concise."

# One setting for base and steered: the notebook's steered sampling (entropy sampling is the
# package default, so `temperature` is ignored; top_p and repetition_penalty still apply).
SAMPLED = dict(temperature=0.9, top_p=0.7, repetition_penalty=1.5)
# Exact-match tasks: argmax decoding (entropy sampling off, temperature 0), production repetition penalty.
GREEDY = dict(temperature=0.0, top_p=0.8, repetition_penalty=1.1, use_entropy_sampling=False)

# Letter scoring mimics the first denoising step of generation: prompt + one block of masks.
N_MASK = 64
LETTERS = "ABCD"
# Text the assistant turn starts with before the answer slot. Chosen on the base model only (checks.py).
ANSWER_PREFIX = "Answer:"


class Steerling:
    def __init__(self, device: str = "cuda"):
        model = AutoModel.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=True, dtype=torch.bfloat16)
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV, trust_remote_code=True)
        self.gen = SteerlingGenerator.from_model(model, self.tok, device=device)
        self.device = self.gen.device
        self.eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
        # " A" and "A" both count as answering A.
        self.letter_ids = {c: [self.tok.encode(v, add_special_tokens=False)[0] for v in (c, f" {c}")] for c in LETTERS}

    def chat(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def steering(cond: Condition) -> SteeringConfig | None:
        if cond.is_base:
            return None
        return SteeringConfig(
            concept_ids=list(cond.concept_ids),
            mai_lm_target=cond.alpha,
            normalize_mai_lm_target=False,
            inject_layer=None,
            inject_alpha_schedule="fixed",
            relu_logit_mask=dict(cond.relu) or None,
        )

    @torch.inference_mode()
    def generate(self, user: str, cond: Condition, seed: int, greedy: bool = False, max_new_tokens: int = 128) -> str:
        cfg = GenerationConfig(
            max_new_tokens=max_new_tokens,
            steps=max_new_tokens,
            seed=seed,
            stop_tokens=[self.eot],
            **(GREEDY if greedy else SAMPLED),
        )
        steer = self.steering(cond)
        prompt = self.chat(user)
        out = self.gen.generate_full(prompt, cfg) if steer is None else self.gen.generate_steered(prompt, cfg, steer)
        return out.text

    def _steered_forward(self, x: torch.Tensor, cond: Condition, return_outputs: bool = False):
        """One forward pass with the condition's injection; returns (logits, outputs, relu_penalty)."""
        steer = self.steering(cond)
        inj, layer, alpha, relu = None, None, 1.0, None
        if steer is not None and steer.has_injection:
            direction, alpha, layer = self.gen._prepare_steering(steer)
            inj = self.gen._position_injection(x, direction)
        if steer is not None and steer.has_logit_mask:
            relu = self.gen._build_relu_mask(steer.relu_logit_mask)
        out = self.gen._forward(x, inj, layer, alpha, return_outputs=return_outputs)
        logits, outputs = out if return_outputs else (out, None)
        return logits, outputs, relu

    def _answer_row(self, user: str, cond: Condition, n_mask: int, prefix: str) -> torch.Tensor:
        """Vocab logits at the first answer slot, with the condition's injection and word penalty."""
        ids = self.tok.encode(self.chat(user) + prefix, add_special_tokens=False)
        x = torch.tensor([ids + [self.gen.mask_token_id] * n_mask], device=self.device)
        logits, _, relu = self._steered_forward(x, cond)
        row = logits[0, len(ids)].float()
        return row - relu if relu is not None else row

    @torch.inference_mode()
    def letter_logits(self, user: str, cond: Condition, n_mask: int = N_MASK, prefix: str = ANSWER_PREFIX) -> dict[str, float]:
        """Logit of each answer letter at the first answer slot (max over the "A"/" A" spellings)."""
        row = self._answer_row(user, cond, n_mask, prefix)
        return {c: max(float(row[i]) for i in self.letter_ids[c]) for c in LETTERS}

    @torch.inference_mode()
    def answer_slot_top(self, user: str, cond: Condition, k: int = 5, prefix: str = ANSWER_PREFIX) -> list[tuple[str, float]]:
        """Top-k tokens at the answer slot, for checking the letter-token assumption."""
        top = self._answer_row(user, cond, N_MASK, prefix).topk(k)
        return [(self.tok.decode([int(i)]), float(v)) for i, v in zip(top.indices, top.values, strict=True)]

    @torch.inference_mode()
    def concept_topk(self, text: str, cond: Condition, seed: int, mask_frac: float = 0.3) -> torch.Tensor:
        """Top-k known-concept ids (T, k) on `text` with a seeded `mask_frac` of tokens masked.

        Injection only acts at masked positions, so base and steered runs mask the same positions.
        """
        ids = torch.tensor(self.tok.encode(text, add_special_tokens=False), device=self.device)
        g = torch.Generator().manual_seed(seed)
        masked = torch.randperm(len(ids), generator=g)[: int(mask_frac * len(ids))].to(self.device)
        ids[masked] = self.gen.mask_token_id
        _, outputs, _ = self._steered_forward(ids[None], cond, return_outputs=True)
        if outputs is None or outputs.known_topk_indices is None:
            raise RuntimeError("model did not return known_topk_indices; check topk_known in the config")
        return outputs.known_topk_indices[0]

    def concept_top_tokens(self, cid: int, k: int) -> list[tuple[str, float]]:
        return self.gen.concept_top_tokens(cid, k)
