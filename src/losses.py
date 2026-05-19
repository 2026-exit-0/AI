"""마스크 기반 멀티태스크 손실.

- 회귀: 결측 라벨은 mask=0으로 손실 제외 → 평균 시 유효 라벨만 카운트
- 분류: 결측 라벨은 target=-1로 표시 → CrossEntropyLoss(ignore_index=-1)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """결측 라벨을 제외한 SmoothL1 회귀 손실 (Huber)."""
    diff = F.smooth_l1_loss(pred, target, reduction="none")  # (B, R)
    diff = diff * mask
    denom = mask.sum().clamp(min=eps)
    return diff.sum() / denom


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: dict,
    regression_weight: float = 1.0,
    classification_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """전체 손실 계산.

    반환:
      total_loss: 스칼라 텐서
      info:       각 항목별 손실값 (로깅용)
    """
    info: Dict[str, float] = {}
    total = torch.tensor(0.0, device=batch["image"].device)

    # ----- 회귀 -----
    if "regression" in outputs:
        reg_pred = outputs["regression"]
        reg_target = batch["regression"]
        reg_mask = batch["regression_mask"]
        reg_loss = masked_regression_loss(reg_pred, reg_target, reg_mask)
        total = total + regression_weight * reg_loss
        info["loss/regression"] = float(reg_loss.detach().cpu())

    # ----- 분류 -----
    if "classification" in outputs:
        ce = nn.CrossEntropyLoss(ignore_index=-1)
        cls_total = 0.0
        cls_count = 0
        for name, logits in outputs["classification"].items():
            tgt = batch["classification"][name]
            if (tgt != -1).sum() == 0:
                info[f"loss/cls_{name}"] = 0.0
                continue
            cls_l = ce(logits, tgt)
            cls_total = cls_total + cls_l
            cls_count += 1
            info[f"loss/cls_{name}"] = float(cls_l.detach().cpu())
        if cls_count > 0:
            cls_avg = cls_total / cls_count
            total = total + classification_weight * cls_avg
            info["loss/classification"] = float(cls_avg.detach().cpu())

    info["loss/total"] = float(total.detach().cpu())
    return total, info
