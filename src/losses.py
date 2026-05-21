"""마스크 기반 멀티태스크 손실.

- 회귀: 결측 라벨은 mask=0으로 손실 제외 → 평균 시 유효 라벨만 카운트
- 분류: 결측 라벨은 target=-1로 표시 → CrossEntropyLoss(ignore_index=-1)
- 분류 헤드별 class_weights (선택) — sklearn 'balanced' 방식으로 minority class 보강
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
    return_per_head: bool = False,
):
    """결측 라벨을 제외한 SmoothL1 회귀 손실 (Huber).

    return_per_head=True 면 (total_loss, per_head_loss[R]) 튜플 반환.
    """
    diff = F.smooth_l1_loss(pred, target, reduction="none")  # (B, R)
    masked_diff = diff * mask
    total_denom = mask.sum().clamp(min=eps)
    total = masked_diff.sum() / total_denom
    if not return_per_head:
        return total
    per_head_denom = mask.sum(dim=0).clamp(min=eps)  # (R,)
    per_head = masked_diff.sum(dim=0) / per_head_denom  # (R,)
    return total, per_head


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: dict,
    regression_weight: float = 1.0,
    classification_weight: float = 0.5,
    class_weights: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    regression_targets: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """전체 손실 계산.

    Args:
      class_weights: {head_name: tensor(K,) or None}. CE에 weight=... 로 주입.
      regression_targets: 회귀 헤드 이름 리스트. 주어지면 per-head 손실도 로깅.

    Returns:
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
        reg_loss, per_head = masked_regression_loss(
            reg_pred, reg_target, reg_mask, return_per_head=True
        )
        total = total + regression_weight * reg_loss
        info["loss/regression"] = float(reg_loss.detach().cpu())
        if regression_targets is not None:
            for i, name in enumerate(regression_targets):
                if i < per_head.shape[0]:
                    info[f"loss/reg_{name}"] = float(per_head[i].detach().cpu())

    # ----- 분류 -----
    if "classification" in outputs:
        cls_total = 0.0
        cls_count = 0
        for name, logits in outputs["classification"].items():
            tgt = batch["classification"][name]
            if (tgt != -1).sum() == 0:
                info[f"loss/cls_{name}"] = 0.0
                continue
            # 헤드별 class_weight 적용 (있으면)
            weight = None
            if class_weights is not None:
                w = class_weights.get(name)
                if w is not None:
                    weight = w.to(logits.device)
            ce = nn.CrossEntropyLoss(ignore_index=-1, weight=weight)
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
