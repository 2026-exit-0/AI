"""PyTorch Dataset.

manifest.csv (build_manifest.py 결과물)을 읽어
이미지 + 부위 ID + 회귀/분류 라벨 + 라벨 마스크를 반환한다.

각 부위마다 사용 가능한 라벨이 다르므로, 결측 라벨은 mask=0(회귀)/-1(분류)로 표시해
losses.py에서 손실 계산에서 제외한다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """ResNet-50 표준 전처리.

    학습 시 약한 증강만 적용 (피부 색상·질감 변형은 최소화).
    """
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_regression_stats(
    df: pd.DataFrame, targets: List[str]
) -> Dict[str, Dict[str, float]]:
    """학습셋에서 회귀 타겟별 평균/표준편차 계산. 정규화에 사용."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in targets:
        if col not in df.columns:
            stats[col] = {"mean": 0.0, "std": 1.0}
            continue
        v = df[col].dropna()
        if len(v) > 0:
            std = float(v.std())
            stats[col] = {
                "mean": float(v.mean()),
                "std": std if std > 1e-6 else 1.0,
            }
        else:
            stats[col] = {"mean": 0.0, "std": 1.0}
    return stats


class DamdaSkinDataset(Dataset):
    """담다 피부 데이터셋.

    각 샘플은 다음을 반환:
      image:            (C, H, W) float tensor
      region_id:        long tensor (scalar)
      regression:       (R,) float tensor — 결측은 0, 정규화된 값
      regression_mask:  (R,) float tensor — 결측은 0, 유효는 1
      classification:   dict[str -> long tensor (scalar)] — 결측은 -1
      meta:             dict (subject_id, gender, age 등 디버깅용)

    회귀 타겟은 regression_stats(mean/std)로 표준화. 추론 시 denormalize 필요.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        regression_targets: List[str],
        classification_heads: Dict[str, int],
        image_size: int = 224,
        train: bool = True,
        regression_stats: Dict[str, Dict[str, float]] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.regression_targets = regression_targets
        self.classification_heads = classification_heads
        self.transform = build_transforms(image_size, train)
        # 회귀 정규화 통계 (train.py 에서 학습셋 기준으로 계산해 주입)
        self.regression_stats = regression_stats or {
            col: {"mean": 0.0, "std": 1.0} for col in regression_targets
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ----- 이미지 (bbox 가 있으면 crop) -----
        img = Image.open(row["image_path"]).convert("RGB")

        bx1 = row.get("bbox_x1"); by1 = row.get("bbox_y1")
        bx2 = row.get("bbox_x2"); by2 = row.get("bbox_y2")
        has_bbox = all(
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for v in (bx1, by1, bx2, by2)
        )
        if has_bbox:
            # 약간의 패딩 추가 (5%) — 부위 가장자리 정보 보존
            x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = int(w * 0.05), int(h * 0.05)
            W, H = img.size
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(W, x2 + pad_x)
            y2 = min(H, y2 + pad_y)
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))

        img_tensor = self.transform(img)

        # ----- 부위 ID -----
        region_id = torch.tensor(int(row["region_id"]), dtype=torch.long)

        # ----- 회귀 라벨 + 마스크 (정규화 적용) -----
        reg_values, reg_mask = [], []
        for col in self.regression_targets:
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                reg_values.append(0.0)
                reg_mask.append(0.0)
            else:
                stats = self.regression_stats.get(col, {"mean": 0.0, "std": 1.0})
                normalized = (float(v) - stats["mean"]) / stats["std"]
                reg_values.append(normalized)
                reg_mask.append(1.0)
        regression = torch.tensor(reg_values, dtype=torch.float32)
        regression_mask = torch.tensor(reg_mask, dtype=torch.float32)

        # ----- 분류 라벨 (0-base 가정, 결측은 -1 = ignore_index) -----
        # 실데이터(JSON) 확인: skin_type=0 같이 0-base 값 존재 → -1 보정하지 않음.
        classification = {}
        for col, num_cls in self.classification_heads.items():
            v = row.get(col)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                classification[col] = torch.tensor(-1, dtype=torch.long)
            else:
                cls_idx = int(v)
                # 범위 초과 시 안전하게 clamp
                cls_idx = max(0, min(num_cls - 1, cls_idx))
                classification[col] = torch.tensor(cls_idx, dtype=torch.long)

        return {
            "image": img_tensor,
            "region_id": region_id,
            "regression": regression,
            "regression_mask": regression_mask,
            "classification": classification,
            "meta": {
                "subject_id": str(row.get("subject_id", "")),
                "region": str(row.get("region", "")),
            },
        }


def split_manifest(
    df: pd.DataFrame,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
    split_by: str = "id",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """manifest를 train/val/test로 분할.

    split_by='id' 권장 — 같은 대상자(subject_id)의 사진이 train/val에 섞이지 않게.
    """
    rng = np.random.default_rng(seed)

    if split_by == "id" and "subject_id" in df.columns and df["subject_id"].notna().any():
        ids = df["subject_id"].dropna().unique()
        rng.shuffle(ids)
        n_test = int(len(ids) * test_split)
        n_val = int(len(ids) * val_split)
        test_ids = set(ids[:n_test])
        val_ids = set(ids[n_test:n_test + n_val])
        train_ids = set(ids[n_test + n_val:])
        train_df = df[df["subject_id"].isin(train_ids)].reset_index(drop=True)
        val_df = df[df["subject_id"].isin(val_ids)].reset_index(drop=True)
        test_df = df[df["subject_id"].isin(test_ids)].reset_index(drop=True)
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = int(len(idx) * test_split)
        n_val = int(len(idx) * val_split)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def collate_fn(batch: List[dict]) -> dict:
    """배치 collate — classification 사전을 텐서로 묶음."""
    images = torch.stack([b["image"] for b in batch])
    region_ids = torch.stack([b["region_id"] for b in batch])
    regression = torch.stack([b["regression"] for b in batch])
    regression_mask = torch.stack([b["regression_mask"] for b in batch])

    cls_keys = batch[0]["classification"].keys()
    classification = {k: torch.stack([b["classification"][k] for b in batch]) for k in cls_keys}

    return {
        "image": images,
        "region_id": region_ids,
        "regression": regression,
        "regression_mask": regression_mask,
        "classification": classification,
    }
