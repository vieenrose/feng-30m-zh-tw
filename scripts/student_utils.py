"""Shared helpers for student training (memory-safe chunked cross-entropy)."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_lm_loss(model, input_ids, labels, chunk=512, head_weight=None):
    """Cross-entropy without materialising [batch, seq, vocab] logits.

    The lm_head+CE is checkpointed per chunk, so peak memory stays ~chunk*vocab.
    head_weight: 显式指定输出投影权重（W-QAT 用：传入按导出格式量化过的
    lm_head，否则损失会绕过量化 forward 直接读原始 fp32 权重）。
    """
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    weight = model.lm_head.weight if head_weight is None else head_weight
    V = weight.shape[0]
    total_sup = (labels != -100).sum().clamp(min=1)
    S = input_ids.shape[1]

    def part(h, lab, w, denom):
        logits = h @ w.t()
        return F.cross_entropy(logits.float().view(-1, V), lab.reshape(-1),
                               ignore_index=-100, reduction="sum") / denom

    losses = []
    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        losses.append(checkpoint(part, hidden[:, c0:c1], labels[:, c0:c1], weight, total_sup,
                                 use_reentrant=False))
    return sum(losses)
