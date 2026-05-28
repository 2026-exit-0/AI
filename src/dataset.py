"""PyTorch Dataset.

manifest.csv (build_manifest.py 결과물)을 읽어
이미지 + 부위 ID + 회귀/분류 라벨 + 라벨 마스크를 반환한다.

각 부위마다 사용 가능한 라벨이 다르므로, 결측 라벨은 mask=0(회귀)/-1(분류)로 표시해
losses.py에서 손실 계산에서 제외한다.
"""

from __future__ import annotations

import io
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JPEGCompress:
    """JPEG 압축 시뮬 — 무작위 quality 로 한 번 인코딩/디코딩 거쳐 압축 artifact 주입.

    ESP32-CAM (OV2640) 의 강한 JPEG 압축 (대역폭 절감 목적) 재현용.
    """

    def __init__(self, quality_range: Tuple[int, int] = (30, 70), p: float = 0.7):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class LowResSimulate:
    """저해상도 다운샘플 -> 업샘플로 디테일 손실 재현.

    ESP32-CAM 의 부위 crop 후 실해상도 (~100-200px) 를 224x224 로 강제 업샘플하면
    디테일이 사라지는 효과 재현.
    """

    def __init__(self, low_range: Tuple[int, int] = (64, 128), p: float = 0.5):
        self.low_range = low_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        target_size = random.randint(self.low_range[0], self.low_range[1])
        w, h = img.size
        if min(w, h) <= target_size:
            return img
        down = img.resize((target_size, target_size), Image.BILINEAR)
        up = down.resize((w, h), Image.BILINEAR)
        return up


class GaussianBlurRandom:
    """가벼운 Gaussian blur — OV2640 의 부드러운 출력 + 미세 손떨림 재현."""

    def __init__(self, radius_range: Tuple[float, float] = (0.3, 1.2), p: float = 0.5):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        r = random.uniform(self.radius_range[0], self.radius_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class GaussianNoiseTensor:
    """텐서 변환 후 적용하는 Gaussian noise — OV2640 의 노이즈 (특히 저조도) 재현."""

    def __init__(self, std_range: Tuple[float, float] = (0.0, 0.05), p: float = 0.6):
        self.std_range = std_range
        self.p = p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return t
        std = random.uniform(self.std_range[0], self.std_range[1])
        if std <= 0:
            return t
        return (t + torch.randn_like(t) * std).clamp(0.0, 1.0)


def build_transforms(
    image_size: int = 224,
    train: bool = True,
    augment_mode: str = "normal",
) -> transforms.Compose:
    """ResNet-50 표준 전처리.

    학습 시 중간 강도 증강 적용 — 피부 색감(hue/saturation)은 신호이므로 약하게,
    공간 변형(rotation/crop/flip/erasing)은 일반화 강화에 도움이라 약간 강하게.

    Args:
        augment_mode:
            'normal'  — 기본 학습 증강 (v1~v4 와 동일).
            'scanner' — ESP32-CAM 시연 환경 시뮬. 'normal' 위에
                        저해상도 시뮬 + Gaussian blur + ColorJitter 강화 +
                        JPEG compression + Gaussian noise 를 추가.
                        도메인 갭 (AI-Hub 학습 vs ESP32-CAM 시연) 대비.
                        Phase 2 / v5 부터 사용 예정. 상세는 NOTES.md 8절 참고.
    """
    if not train:
        # 평가/추론 transform 은 augment_mode 와 무관 — 결정론적
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    mode = (augment_mode or "normal").lower()
    if mode not in ("normal", "scanner"):
        raise ValueError(f"augment_mode 는 'normal' | 'scanner' 만 허용 (got: {augment_mode!r})")

    if mode == "normal":
        return transforms.Compose([
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.5, 2.0)),
        ])

    # mode == "scanner" — ESP32-CAM 시연 환경 시뮬레이션
    #
    # v5.1 (2026-05-27): augmentation 강도 ~40% 수준으로 약화.
    # v5 결과 분석에서 고주파 디테일 의존 헤드 (pore_value +38% MAE, pigmentation_value +45%,
    # wrinkle_grade -25% F1) 가 명백히 악화 — LowResSimulate 가 미세 텍스처 정보를 파괴한 것이 주범.
    # 거시적 정보 의존 헤드 (elasticity, pigmentation_grade, sensitive) 는 거의 영향 없었음.
    # 가설: aug 강도를 줄이면 디테일 헤드 회복 + scanner robustness 일부는 유지 가능.
    # 상세는 PROGRESS.md §4 v5 / v5.1 절 참고.
    return transforms.Compose([
        transforms.Resize((image_size + 24, image_size + 24)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        # 저해상도 (v5 의 주범) — low_range 64-128→128-192, p 0.6→0.3
        LowResSimulate(low_range=(128, 192), p=0.3),
        # 블러 — radius 0.3-1.2→0.2-0.8, p 0.5→0.3
        GaussianBlurRandom(radius_range=(0.2, 0.8), p=0.3),
        # ColorJitter — v5 의 강화값에서 v3 / normal 수준으로 복귀
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
        # JPEG 압축 — quality 30-70→60-85 (덜 압축), p 0.7→0.4
        JPEGCompress(quality_range=(60, 85), p=0.4),
        transforms.ToTensor(),
        # Gaussian noise — std 0-0.05→0-0.025, p 0.6→0.4
        GaussianNoiseTensor(std_range=(0.0, 0.025), p=0.4),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.5, 2.0)),
    ])


def compute_class_weights(
    df: pd.DataFrame,
    classification_heads: Dict[str, int],
    smooth: float = 0.5,
    max_weight: float = 5.0,
    min_count: int = 5,
) -> Dict[str, torch.Tensor]:
    """학습셋 분포 기반 분류 헤드별 클래스 가중치 (sklearn 'balanced' 방식).

    공식: w_c = N / (K * (count_c + smooth))
    안정성 보강:
      - min_count: 학습셋 등장 횟수가 min_count 미만인 클래스는 1.0 으로 고정
        (희소 클래스가 폭주 가중치를 받아 학습 망가지는 것 방지)
      - max_weight: 그 외 모든 가중치는 절대 상한 max_weight 로 클리핑

    반환: {head_name: tensor(K,)} — losses.py 의 multitask_loss(class_weights=...) 에 그대로 주입.
    """
    weights: Dict[str, torch.Tensor] = {}
    for col, num_cls in classification_heads.items():
        if col not in df.columns:
            continue
        counts = np.zeros(num_cls, dtype=np.float64)
        for v in df[col].dropna():
            idx = int(v)
            if 0 <= idx < num_cls:
                counts[idx] += 1
        total = counts.sum()
        if total == 0:
            continue
        w = total / (num_cls * (counts + smooth))
        w = np.minimum(w, max_weight)        # 절대 상한
        w[counts < min_count] = 1.0          # 너무 희소한 클래스는 가중치 무효화
        weights[col] = torch.tensor(w, dtype=torch.float32)
    return weights


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


def compute_sensor_stats(
    df: pd.DataFrame, sensor_inputs: List[str]
) -> Dict[str, Dict[str, float]]:
    """학습셋에서 sensor 입력별 평균/std 계산. 정규화에 사용.

    compute_regression_stats 와 같은 형태로 반환 (재활용 가능).
    Phase 2 / v5+ 의 ESP32-CAM 시연 환경 통합용.
    """
    return compute_regression_stats(df, sensor_inputs)


class DamdaSkinDataset(Dataset):
    """담다 피부 데이터셋.

    각 샘플은 다음을 반환:
      image:            (C, H, W) float tensor
      region_id:        long tensor (scalar)
      regression:       (R,) float tensor — 결측은 0, 정규화된 값
      regression_mask:  (R,) float tensor — 결측은 0, 유효는 1
      classification:   dict[str -> long tensor (scalar)] — 결측은 -1
      sensor:           (S,) float tensor — 결측은 0, 정규화된 값 (sensor_inputs 가 빈 리스트면 (0,))
      sensor_mask:      (S,) float tensor — 결측은 0, 유효는 1
      meta:             dict (subject_id, gender, age 등 디버깅용)

    회귀 타겟은 regression_stats(mean/std)로 표준화. 추론 시 denormalize 필요.
    Phase 2 / v5+ 의 sensor 통합: sensor_inputs 컬럼명 리스트 + sensor_stats 로 정규화.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        regression_targets: List[str],
        classification_heads: Dict[str, int],
        image_size: int = 224,
        train: bool = True,
        regression_stats: Dict[str, Dict[str, float]] = None,
        augment_mode: str = "normal",
        sensor_inputs: List[str] = None,
        sensor_stats: Dict[str, Dict[str, float]] = None,
        # v5.5: categorical inputs (사용자 자가입력 — skin_type / sensitive 등)
        # dict: {col_name: num_classes}. one-hot 인코딩되어 단일 벡터로 concat.
        categorical_inputs: Dict[str, int] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.regression_targets = regression_targets
        self.classification_heads = classification_heads
        self.augment_mode = augment_mode
        self.transform = build_transforms(image_size, train, augment_mode=augment_mode)
        # 회귀 정규화 통계 (train.py 에서 학습셋 기준으로 계산해 주입)
        self.regression_stats = regression_stats or {
            col: {"mean": 0.0, "std": 1.0} for col in regression_targets
        }
        # Sensor 입력 (v5+). 빈 리스트면 sensor 비활성 (model.sensor_dim=0 필요)
        self.sensor_inputs = list(sensor_inputs or [])
        self.sensor_stats = sensor_stats or {
            col: {"mean": 0.0, "std": 1.0} for col in self.sensor_inputs
        }
        # Categorical 입력 (v5.5+). 빈 dict 면 categorical 비활성
        self.categorical_inputs: Dict[str, int] = dict(categorical_inputs or {})
        # 총 one-hot 차원 (model.categorical_dim 과 일치해야 함)
        self.categorical_dim = sum(self.categorical_inputs.values())

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
        # 실데이터(JSON) 확인: skin_type=0 같이 0-base 값 존재 -> -1 보정하지 않음.
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

        # ----- 센서 입력 (v5+) — 정규화 + mask -----
        if self.sensor_inputs:
            s_values, s_mask = [], []
            for col in self.sensor_inputs:
                v = row.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    s_values.append(0.0)
                    s_mask.append(0.0)
                else:
                    st = self.sensor_stats.get(col, {"mean": 0.0, "std": 1.0})
                    s_values.append((float(v) - st["mean"]) / st["std"])
                    s_mask.append(1.0)
            sensor = torch.tensor(s_values, dtype=torch.float32)
            sensor_mask = torch.tensor(s_mask, dtype=torch.float32)
        else:
            sensor = torch.zeros(0, dtype=torch.float32)
            sensor_mask = torch.zeros(0, dtype=torch.float32)

        # ----- Categorical 입력 (v5.5+) — one-hot + mask -----
        # 각 categorical col 을 one-hot 으로 변환 후 concat. 결측이면 0-벡터 + mask=0.
        # 학습 시엔 manifest 의 정수값 사용. 추론 시엔 사용자 자가입력 (UI/questionnaire).
        if self.categorical_inputs:
            one_hots: List[float] = []
            any_valid = False
            for col, num_cls in self.categorical_inputs.items():
                v = row.get(col)
                vec = [0.0] * num_cls
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    idx = int(v)
                    if 0 <= idx < num_cls:
                        vec[idx] = 1.0
                        any_valid = True
                one_hots.extend(vec)
            categorical = torch.tensor(one_hots, dtype=torch.float32)
            categorical_mask = torch.tensor(1.0 if any_valid else 0.0, dtype=torch.float32)
        else:
            categorical = torch.zeros(0, dtype=torch.float32)
            categorical_mask = torch.zeros((), dtype=torch.float32)

        return {
            "image": img_tensor,
            "region_id": region_id,
            "regression": regression,
            "regression_mask": regression_mask,
            "classification": classification,
            "sensor": sensor,
            "sensor_mask": sensor_mask,
            "categorical": categorical,
            "categorical_mask": categorical_mask,
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
    """배치 collate — classification 사전을 텐서로 묶음. sensor / categorical 도 함께 (v5.5+)."""
    images = torch.stack([b["image"] for b in batch])
    region_ids = torch.stack([b["region_id"] for b in batch])
    regression = torch.stack([b["regression"] for b in batch])
    regression_mask = torch.stack([b["regression_mask"] for b in batch])

    cls_keys = batch[0]["classification"].keys()
    classification = {k: torch.stack([b["classification"][k] for b in batch]) for k in cls_keys}

    sensor = torch.stack([b["sensor"] for b in batch])             # (B, S)
    sensor_mask = torch.stack([b["sensor_mask"] for b in batch])   # (B, S)

    categorical = torch.stack([b["categorical"] for b in batch])             # (B, C)
    categorical_mask = torch.stack([b["categorical_mask"] for b in batch])   # (B,)

    return {
        "image": images,
        "region_id": region_ids,
        "regression": regression,
        "regression_mask": regression_mask,
        "classification": classification,
        "sensor": sensor,
        "sensor_mask": sensor_mask,
        "categorical": categorical,
        "categorical_mask": categorical_mask,
    }
