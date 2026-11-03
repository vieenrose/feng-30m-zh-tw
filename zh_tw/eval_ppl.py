#!/usr/bin/env python
"""Compute perplexity of a model on a text file. Prints METRIC lines."""
import argparse, json, math, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--device", default="cpu")
A = ap.parse_args()

tok = AutoTokenizer.from_pretrained(A.model, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.float32)
mdl.eval()
text = open(A.text, encoding="utf-8").read()
ids = tok(text, return_tensors="pt")["input_ids"]
if ids.shape[1] > A.max_tokens:
    ids = ids[:, :A.max_tokens]
n = ids.shape[1]
with torch.no_grad():
    out = mdl(ids)
logits = out.logits  # [1, n, V]
# CE over positions 1..n-1
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = ids[:, 1:].contiguous()
ce = torch.nn.functional.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    reduction="mean",
)
ppl = math.exp(float(ce))
print(f"METRIC ppl={ppl:.6f}")
print(f"METRIC ce_nats={float(ce):.6f}")
print(f"METRIC n_tokens={n}")
# tokens per char
print(f"METRIC tok_per_char={n/len(text):.6f}")
