"""모델: ResNet-50 백본 + 부위 임베딩 + 다중 회귀/분류 헤드.

Phase 2 확장 포인트:
  - forward(image, region_id, sensor=None) 시그니처 유지
  - sensor 입력 시 sensor_branch로 처리해 fusion concat
  - 학습 코드는 그대로 두고 dataset이 sensor 텐서를 추가로 반환하도록 수정만 하면 됨
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models


class DamdaSkinModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_regions: int = 9,
        region_emb_dim: int = 16,
        regression_targets: Optional[list] = None,
        classification_heads: Optional[Dict[str, int]] = None,
        dropout: float = 0.2,
        # Phase 2 확장용
        sensor_dim: int = 0,
        sensor_emb_dim: int = 32,
        # v5.5: categorical inputs (사용자 입력 — skin_type / sensitive 등)
        # 각 입력은 one-hot 으로 인코딩되어 concat → categorical_branch 통과 → trunk 에 fusion
        categorical_dim: int = 0,   # = sum(num_classes for each categorical input)
        categorical_emb_dim: int = 32,
    ):
        super().__init__()

        if regression_targets is None:
            regression_targets = []
        if classification_heads is None:
            classification_heads = {}

        self.regression_targets = list(regression_targets)
        self.classification_head_names = list(classification_heads.keys())

        # ----- Backbone -----
        if backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.feature_dim = net.fc.in_features  # 2048
            net.fc = nn.Identity()
            self.backbone = net
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ----- Region embedding -----
        self.region_embedding = nn.Embedding(num_regions, region_emb_dim)

        # ----- (Phase 2) Sensor branch -----
        self.sensor_dim = sensor_dim
        if sensor_dim > 0:
            self.sensor_branch = nn.Sequential(
                nn.Linear(sensor_dim, sensor_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(sensor_emb_dim, sensor_emb_dim),
            )
            fused_dim = self.feature_dim + region_emb_dim + sensor_emb_dim
        else:
            self.sensor_branch = None
            fused_dim = self.feature_dim + region_emb_dim

        # ----- (v5.5) Categorical branch (사용자 자가 입력) -----
        self.categorical_dim = categorical_dim
        if categorical_dim > 0:
            self.categorical_branch = nn.Sequential(
                nn.Linear(categorical_dim, categorical_emb_dim),
                nn.ReLU(inplace=True),
                nn.Linear(categorical_emb_dim, categorical_emb_dim),
            )
            fused_dim += categorical_emb_dim
        else:
            self.categorical_branch = None

        # ----- Shared trunk -----
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ----- Regression head -----
        if len(self.regression_targets) > 0:
            self.regression_head = nn.Linear(512, len(self.regression_targets))
        else:
            self.regression_head = None

        # ----- Classification heads -----
        self.classification_heads = nn.ModuleDict({
            name: nn.Linear(512, num_classes)
            for name, num_classes in classification_heads.items()
        })

    def forward(
        self,
        image: torch.Tensor,
        region_id: torch.Tensor,
        sensor: Optional[torch.Tensor] = None,
        sensor_mask: Optional[torch.Tensor] = None,
        categorical: Optional[torch.Tensor] = None,
        categorical_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """순전파.

        Args:
            sensor: (B, sensor_dim). sensor_dim > 0 인 경우 필수.
            sensor_mask: (B,) 또는 (B, sensor_dim) — 1=valid, 0=결측.
            categorical: (B, categorical_dim). categorical_dim > 0 인 경우 필수.
                각 categorical input 의 one-hot 을 concat 한 벡터.
            categorical_mask: (B,) — 1=valid (사용자가 입력함), 0=결측 (UI 에서 안 입력).
                None 이면 모두 valid 로 간주.
        """
        img_feat = self.backbone(image)              # (B, 2048)
        reg_emb = self.region_embedding(region_id)   # (B, 16)

        feats = [img_feat, reg_emb]

        if self.sensor_branch is not None:
            if sensor is None:
                raise ValueError("sensor_dim > 0 이지만 sensor 입력이 None")
            sens_feat = self.sensor_branch(sensor)   # (B, sensor_emb_dim)
            if sensor_mask is not None:
                if sensor_mask.dim() == 1:
                    m = sensor_mask.unsqueeze(1)
                elif sensor_mask.dim() == 2 and sensor_mask.shape[1] != 1:
                    m = (sensor_mask.sum(dim=1, keepdim=True) > 0).float()
                else:
                    m = sensor_mask
                sens_feat = sens_feat * m
            feats.append(sens_feat)

        if self.categorical_branch is not None:
            if categorical is None:
                raise ValueError("categorical_dim > 0 이지만 categorical 입력이 None")
            cat_feat = self.categorical_branch(categorical)  # (B, categorical_emb_dim)
            if categorical_mask is not None:
                m = categorical_mask.unsqueeze(1) if categorical_mask.dim() == 1 else categorical_mask
                cat_feat = cat_feat * m
            feats.append(cat_feat)

        fused = torch.cat(feats, dim=1)
        trunk_out = self.trunk(fused)                # (B, 512)

        out = {}
        if self.regression_head is not None:
            out["regression"] = self.regression_head(trunk_out)
        if self.classification_heads:
            out["classification"] = {
                name: head(trunk_out)
                for name, head in self.classification_heads.items()
            }
        return out

    def freeze_backbone(self) -> None:
        """초기 epoch에 백본을 동결해 헤드부터 따뜻하게 학습."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
