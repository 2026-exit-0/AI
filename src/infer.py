"""단일 이미지 추론 — 시연용 entry point.

evaluate.py 가 test set 전체를 batch 추론한다면, 본 스크립트는 ESP32-CAM 시연
시나리오에 맞춰 **이미지 한 장 + 부위 명시 + (선택) 센서값** 을 받아 부위별
측정값/등급 dict 를 반환한다.

사용 예 (Python API — Gradio UI 등에서 호출):

    from src.infer import DamdaInferenceModel

    model = DamdaInferenceModel(
        "checkpoints/epoch030.pt",
        config_path="configs/baseline.yaml",
    )
    result = model.predict(
        image_path="scan.jpg",      # 또는 PIL.Image
        region="L_CHEEK",           # 또는 region_id=5
        sensor={"moisture": 42.5},  # 선택. 학습된 sensor_inputs 와 매칭
        bbox=None,                  # 선택. (x1,y1,x2,y2)
        return_probs=True,          # 분류 확률 함께 반환
    )
    # result["regression"]["moisture"] -> 38.2 (denormalized)
    # result["classification"]["wrinkle_grade"] -> 2 (predicted class)
    # result["classification_probs"]["wrinkle_grade"] -> [0.05, 0.15, 0.62, ...]

사용 예 (CLI — 디버깅용):
    python -m src.infer --checkpoint checkpoints/epoch030.pt ^
        --image scan.jpg --region L_CHEEK --sensor moisture=42.5

설계 의도 / 시연 시나리오는 NOTES.md 8절 참고.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from torchvision import transforms

from .dataset import (
    build_transforms,
    compute_regression_stats,
    compute_sensor_stats,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from .model import DamdaSkinModel
from .utils import ID_TO_REGION, REGION_TO_ID, get_device


# ============================================================
# TTA (Test-Time Augmentation) — 학습 없이 정확도 +2~5%
# ============================================================
# 같은 이미지에 N가지 결정적 변형 적용 → 모델 N번 실행 → 출력 평균.
# 회귀: 그대로 평균. 분류: softmax 평균 후 argmax.
# 변형 종류:
#   1. 원본 (224 resize)
#   2. 수평 flip
#   3. 큰 resize (248) + center crop 224 — 약간 zoom out
#   4. 더 큰 resize (272) + center crop 224 — 더 zoom out

def build_tta_transforms(image_size: int = 224):
    """TTA 용 4가지 결정적 변형 transform 반환."""
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return [
        # 1. 원본
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(), norm,
        ]),
        # 2. 수평 flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
        # 3. 큰 resize + center crop (약한 zoom out)
        transforms.Compose([
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), norm,
        ]),
        # 4. 더 큰 resize + center crop (강한 zoom out)
        transforms.Compose([
            transforms.Resize((image_size + 48, image_size + 48)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), norm,
        ]),
    ]


# ============================================================
# Categorical 입력 string → int 매핑 (시연용 placeholder)
# ============================================================
# 시연 시 사용자 questionnaire 가 한국어 string 으로 결과 반환.
# 모델 학습은 manifest 의 정수값으로 했으므로 변환 필요.
# ⚠ 정확한 매핑은 AI-Hub 028 데이터셋 문서 확인 후 보정 권장 (현재는 placeholder)
SKIN_TYPE_TO_INT = {
    "건성": 0, "지성": 1, "복합성": 2, "민감성": 3, "중성": 4,
}
SENSITIVE_TO_INT = {
    "yes": 1, "no": 0, "있음": 1, "없음": 0, True: 1, False: 0,
}


def _resolve_categorical_value(col: str, v, num_classes: int):
    """string 또는 정수를 모델용 정수 인덱스로 변환. 알 수 없으면 None."""
    if isinstance(v, int):
        return v if 0 <= v < num_classes else None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        v_s = v.strip()
        if col == "skin_type" and v_s in SKIN_TYPE_TO_INT:
            return SKIN_TYPE_TO_INT[v_s]
        if col == "sensitive" and v_s in SENSITIVE_TO_INT:
            return SENSITIVE_TO_INT[v_s]
        # 숫자 문자열도 허용
        try:
            return int(v_s)
        except ValueError:
            return None
    return None


# ============================================================
# Inference 클래스
# ============================================================

class DamdaInferenceModel:
    """단일 샘플 추론 엔진. Gradio/Flask UI 의 백엔드 또는 CLI 의 동작 주체.

    초기화 시 한 번만 ckpt 를 로드하고, predict() 호출마다 빠르게 추론한다.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or get_device()

        # ---- Checkpoint 로드 ----
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.ckpt_epoch = int(ckpt.get("epoch", -1))

        # ---- Config 결정 (이중 source) ----
        # yaml (인자) — 환경/데이터 경로/image_size 용도
        # ckpt 의 저장된 config — model architecture 의 ground truth
        # → 둘이 충돌 시 model architecture 는 무조건 ckpt 우선 (실제 학습된 구조와 일치해야 weight 로드 가능)
        yaml_cfg: Optional[dict] = None
        if config_path is not None:
            yaml_cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        ckpt_cfg: Optional[dict] = ckpt.get("config")

        if yaml_cfg is None and ckpt_cfg is None:
            raise ValueError("config_path 가 None 이고 ckpt 에도 config 없음")

        # 환경 설정 (image_size 등) 은 yaml > ckpt 순. yaml 없으면 ckpt.
        self.cfg = yaml_cfg if yaml_cfg is not None else ckpt_cfg

        # ---- Model architecture (ckpt 가 ground truth) ----
        # v3 ckpt 를 v5 yaml 로 로드하면 회귀 헤드 수 / sensor_branch 차이로 state_dict mismatch 발생.
        # 학습 시 실제 구조 = ckpt 에 저장된 config. yaml 과 다르더라도 ckpt 의 것 사용.
        arch_cfg = ckpt_cfg if ckpt_cfg is not None else yaml_cfg
        arch_model = arch_cfg["model"]
        self.regression_targets: List[str] = list(arch_model["regression_targets"])
        self.classification_heads: Dict[str, int] = dict(arch_model["classification_heads"])
        # sensor_inputs 도 ckpt 우선 (이미 그러고 있었음 — 일관성 유지)
        self.sensor_inputs: List[str] = list(
            ckpt.get("sensor_inputs", arch_cfg.get("data", {}).get("sensor_inputs", []) or [])
        )
        self.sensor_dim = len(self.sensor_inputs)

        # v5.5+ categorical_inputs (사용자 자가입력) — ckpt 우선
        self.categorical_inputs: Dict[str, int] = dict(
            ckpt.get("categorical_inputs", arch_cfg.get("data", {}).get("categorical_inputs", {}) or {})
        )
        self.categorical_dim = sum(self.categorical_inputs.values())

        # ---- 그 외 환경 설정 (yaml 우선, 없으면 ckpt) ----
        self.image_size: int = int(self.cfg["data"]["image_size"])

        # ---- 통계 (정규화/역정규화용) ----
        self.regression_stats: Dict[str, Dict[str, float]] = ckpt.get(
            "regression_stats", {col: {"mean": 0.0, "std": 1.0} for col in self.regression_targets}
        )
        self.sensor_stats: Dict[str, Dict[str, float]] = ckpt.get(
            "sensor_stats", {col: {"mean": 0.0, "std": 1.0} for col in self.sensor_inputs}
        )

        # ---- 모델 구성 + 가중치 로드 ----
        # backbone / region_emb_dim / dropout / sensor/categorical 모두 ckpt arch_cfg 우선
        self.model = DamdaSkinModel(
            backbone=arch_model["backbone"],
            pretrained=False,
            num_regions=arch_model["num_regions"],
            region_emb_dim=arch_model["region_emb_dim"],
            regression_targets=self.regression_targets,
            classification_heads=self.classification_heads,
            dropout=arch_model.get("dropout", 0.2),
            sensor_dim=self.sensor_dim,
            sensor_emb_dim=arch_model.get("sensor_emb_dim", 32),
            categorical_dim=self.categorical_dim,
            categorical_emb_dim=arch_model.get("categorical_emb_dim", 32),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # ---- 추론용 transform (결정론적, augmentation 없음) ----
        self.transform = build_transforms(self.image_size, train=False)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image_path: Union[str, Path, Image.Image],
        region: Union[str, int],
        sensor: Optional[Dict[str, float]] = None,
        categorical: Optional[Dict[str, Union[int, str]]] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        return_probs: bool = False,
        tta: bool = False,
    ) -> dict:
        """단일 이미지 추론.

        Args:
            image_path: 이미지 파일 경로 또는 이미 열린 PIL.Image.
            region: 부위 이름 ('L_CHEEK', 'FOREHEAD', ...) 또는 region_id (정수).
            sensor: 센서 측정값 dict. 학습된 sensor_inputs 와 매칭되지 않으면 mask=0.
                예: {"moisture": 42.5}. None 이면 전부 결측 처리 (모델은 영향 받지만 동작은 함).
            bbox: (x1, y1, x2, y2). 없으면 이미지 전체 사용 (ESP32-CAM 이 이미 부위 crop 한 가정).
            return_probs: True 면 classification 헤드별 softmax 확률 (List[float]) 도 반환.

        Returns:
            dict {
                "regression": {head: denormalized_value},
                "classification": {head: predicted_class_idx},
                "classification_probs": {head: [p0, p1, ...]},   # return_probs=True 일 때만
                "meta": {...},
            }
        """
        # ---- Region 처리 ----
        if isinstance(region, str):
            if region not in REGION_TO_ID:
                raise ValueError(f"알 수 없는 region: {region!r}. 후보: {list(REGION_TO_ID)}")
            region_id = REGION_TO_ID[region]
            region_name = region
        else:
            region_id = int(region)
            region_name = ID_TO_REGION.get(region_id, f"REGION_{region_id}")

        # ---- 이미지 로드 + (선택) bbox crop ----
        if isinstance(image_path, Image.Image):
            img = image_path.convert("RGB")
        else:
            img = Image.open(image_path).convert("RGB")

        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)  # (1, C, H, W)
        rid_tensor = torch.tensor([region_id], dtype=torch.long, device=self.device)

        # ---- Sensor 텐서 준비 ----
        sensor_tensor: Optional[torch.Tensor] = None
        sensor_mask_tensor: Optional[torch.Tensor] = None
        if self.sensor_dim > 0:
            values, mask = [], []
            sensor = sensor or {}
            for col in self.sensor_inputs:
                if col in sensor and sensor[col] is not None:
                    st = self.sensor_stats.get(col, {"mean": 0.0, "std": 1.0})
                    std = max(float(st["std"]), 1e-6)
                    values.append((float(sensor[col]) - float(st["mean"])) / std)
                    mask.append(1.0)
                else:
                    values.append(0.0)
                    mask.append(0.0)
            sensor_tensor = torch.tensor([values], dtype=torch.float32, device=self.device)
            sensor_mask_tensor = torch.tensor([mask], dtype=torch.float32, device=self.device)

        # ---- Categorical 텐서 준비 (v5.5+) ----
        # categorical={"skin_type": 0 or "건성", "sensitive": 1} 같은 dict 받아 one-hot 으로 변환
        categorical_tensor: Optional[torch.Tensor] = None
        categorical_mask_tensor: Optional[torch.Tensor] = None
        if self.categorical_dim > 0:
            one_hots: List[float] = []
            any_valid = False
            cat_input = categorical or {}
            for col, num_cls in self.categorical_inputs.items():
                vec = [0.0] * num_cls
                v = cat_input.get(col)
                if v is not None:
                    idx = _resolve_categorical_value(col, v, num_cls)
                    if idx is not None and 0 <= idx < num_cls:
                        vec[idx] = 1.0
                        any_valid = True
                one_hots.extend(vec)
            categorical_tensor = torch.tensor([one_hots], dtype=torch.float32, device=self.device)
            categorical_mask_tensor = torch.tensor([1.0 if any_valid else 0.0],
                                                   dtype=torch.float32, device=self.device)

        # ---- Forward ----
        kwargs = {}
        if self.sensor_dim > 0:
            kwargs["sensor"] = sensor_tensor
            kwargs["sensor_mask"] = sensor_mask_tensor
        if self.categorical_dim > 0:
            kwargs["categorical"] = categorical_tensor
            kwargs["categorical_mask"] = categorical_mask_tensor

        if tta:
            # TTA: 4가지 변형 → 각각 forward → 평균
            tta_xforms = build_tta_transforms(self.image_size)
            reg_outs: List[torch.Tensor] = []
            cls_outs: Dict[str, List[torch.Tensor]] = {n: [] for n in self.classification_heads}
            for xform in tta_xforms:
                img_t = xform(img).unsqueeze(0).to(self.device)
                o = self.model(img_t, rid_tensor, **kwargs)
                if "regression" in o:
                    reg_outs.append(o["regression"])
                if "classification" in o:
                    for n, logits in o["classification"].items():
                        cls_outs[n].append(F.softmax(logits, dim=-1))
            # 평균 — 회귀는 그대로, 분류는 softmax 평균 후 argmax 가 더 안정적
            out = {}
            if reg_outs:
                out["regression"] = torch.stack(reg_outs).mean(dim=0)
            if any(cls_outs.values()):
                out["classification"] = {n: torch.stack(probs).mean(dim=0)
                                         for n, probs in cls_outs.items() if probs}
            tta_applied = True
        else:
            out = self.model(img_tensor, rid_tensor, **kwargs)
            tta_applied = False

        # ---- 후처리: 회귀 denormalize ----
        regression_out: Dict[str, float] = {}
        if "regression" in out:
            reg = out["regression"].squeeze(0).cpu().numpy()  # (R,)
            for i, name in enumerate(self.regression_targets):
                st = self.regression_stats.get(name, {"mean": 0.0, "std": 1.0})
                std = max(float(st["std"]), 1e-6)
                regression_out[name] = float(reg[i]) * std + float(st["mean"])

        # ---- 후처리: 분류 argmax + (선택) softmax 확률 ----
        # TTA 일 때는 out["classification"] 가 이미 softmax 평균값. 일반 forward 일 때는 logits.
        classification_out: Dict[str, int] = {}
        classification_probs: Dict[str, List[float]] = {}
        if "classification" in out:
            for name, vals in out["classification"].items():
                vals_1d = vals.squeeze(0)  # (K,)
                pred = int(torch.argmax(vals_1d).item())
                classification_out[name] = pred
                if return_probs:
                    if tta_applied:
                        probs = vals_1d.cpu().numpy().tolist()  # 이미 softmax 평균
                    else:
                        probs = F.softmax(vals_1d, dim=-1).cpu().numpy().tolist()
                    classification_probs[name] = [float(p) for p in probs]

        result = {
            "regression": regression_out,
            "classification": classification_out,
            "meta": {
                "region": region_name,
                "region_id": region_id,
                "ckpt_epoch": self.ckpt_epoch,
                "checkpoint": str(self.checkpoint_path),
                "sensor_dim": self.sensor_dim,
                "sensor_inputs_used": self.sensor_inputs,
                "categorical_dim": self.categorical_dim,
                "categorical_inputs_used": list(self.categorical_inputs.keys()),
                "tta": tta_applied,
            },
        }
        if return_probs:
            result["classification_probs"] = classification_probs
        return result

    def regression_target_names(self) -> List[str]:
        """반환 dict 의 regression 키 순서 — UI 에서 표 헤더 만들 때 사용."""
        return list(self.regression_targets)

    def classification_head_names(self) -> List[str]:
        """반환 dict 의 classification 키 순서 — UI 에서 표 헤더 만들 때 사용."""
        return list(self.classification_heads.keys())


# ============================================================
# 앙상블 — 여러 ckpt 의 예측 평균
# ============================================================

class DamdaEnsembleModel:
    """여러 ckpt 의 예측을 평균. 학습 없이 정확도 +3~7% 기대.

    모든 모델이 같은 회귀 헤드 / 분류 헤드 가져야 깔끔. 다르면 교집합만 평균.

    사용 예:
        ens = DamdaEnsembleModel(
            ["checkpoints/epoch048.pt", "checkpoints_v5.1/epoch048.pt"],
            config_path="configs/baseline.yaml",
        )
        result = ens.predict(image_path="scan.jpg", region="L_CHEEK", tta=True)
    """

    def __init__(
        self,
        checkpoint_paths: List[Union[str, Path]],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        if not checkpoint_paths:
            raise ValueError("ensemble 은 최소 1개 ckpt 필요")
        self.models = [
            DamdaInferenceModel(p, config_path=config_path, device=device)
            for p in checkpoint_paths
        ]
        # 공통 헤드만 평균 (불일치 회귀/분류 헤드는 평균 제외)
        self.regression_targets = list(set.intersection(
            *(set(m.regression_targets) for m in self.models)
        ))
        self.classification_heads = list(set.intersection(
            *(set(m.classification_heads) for m in self.models)
        ))

    @torch.no_grad()
    def predict(self, **kwargs) -> dict:
        """각 모델 predict → 회귀 평균 + 분류 softmax 평균 후 argmax."""
        # return_probs 강제 True (분류 확률 평균 위해)
        original_return_probs = kwargs.get("return_probs", False)
        kwargs["return_probs"] = True

        sub_results = [m.predict(**kwargs) for m in self.models]

        # 회귀 평균 (공통 헤드만)
        regression_out: Dict[str, float] = {}
        for name in self.regression_targets:
            vals = [r["regression"][name] for r in sub_results if name in r.get("regression", {})]
            if vals:
                regression_out[name] = sum(vals) / len(vals)

        # 분류 확률 평균 → argmax
        classification_out: Dict[str, int] = {}
        classification_probs: Dict[str, List[float]] = {}
        for name in self.classification_heads:
            prob_lists = [r["classification_probs"][name]
                          for r in sub_results
                          if name in r.get("classification_probs", {})]
            if not prob_lists:
                continue
            # 모든 길이 같은지 확인 (다른 헤드 수 / 클래스 수 시 skip)
            n = len(prob_lists[0])
            if not all(len(p) == n for p in prob_lists):
                continue
            avg = [sum(p[i] for p in prob_lists) / len(prob_lists) for i in range(n)]
            classification_out[name] = int(max(range(n), key=lambda i: avg[i]))
            if original_return_probs:
                classification_probs[name] = avg

        result = {
            "regression": regression_out,
            "classification": classification_out,
            "meta": {
                "ensemble_size": len(self.models),
                "ckpt_epochs": [m.ckpt_epoch for m in self.models],
                "checkpoints": [str(m.checkpoint_path) for m in self.models],
                "regression_targets_common": self.regression_targets,
                "classification_heads_common": self.classification_heads,
                "tta": kwargs.get("tta", False),
            },
        }
        if original_return_probs:
            result["classification_probs"] = classification_probs
        return result


# ============================================================
# CLI
# ============================================================

def _parse_sensor_args(items: List[str]) -> Dict[str, float]:
    """`--sensor key=value` 여러 개를 dict 로. CLI 헬퍼."""
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--sensor 항목은 key=value 형식: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _parse_categorical_args(items: List[str]) -> Dict[str, Union[int, str]]:
    """`--categorical key=value` 여러 개를 dict 로. value 는 string 또는 int."""
    out: Dict[str, Union[int, str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--categorical 항목은 key=value 형식: {item!r}")
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = v  # 한국어 string (예: "건성")
    return out


def _parse_bbox(s: str) -> Tuple[int, int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox 는 'x1,y1,x2,y2' 형식: {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser(description="단일 이미지 추론 (ESP32-CAM 시연용)")
    ap.add_argument("--checkpoint", default="",
                    help="단일 ckpt 경로. --ensemble 미사용 시 필수")
    ap.add_argument("--ensemble", default="",
                    help="콤마 구분 ckpt 경로 리스트 (예: ckpt1.pt,ckpt2.pt). 평균 앙상블")
    ap.add_argument("--config", default="",
                    help="config yaml (생략 시 ckpt 내부의 config 사용)")
    ap.add_argument("--image", required=True, help="이미지 경로")
    ap.add_argument("--region", required=True,
                    help="부위 이름 (FOREHEAD, L_EYE, R_EYE, L_CHEEK, R_CHEEK, LIP, CHIN, GLABELLA, PART_0)")
    ap.add_argument("--sensor", action="append", default=[],
                    metavar="KEY=VAL", help="센서 값. 여러 개 가능: --sensor moisture=42.5")
    ap.add_argument("--categorical", action="append", default=[],
                    metavar="KEY=VAL", help="사용자 자가입력. 예: --categorical skin_type=건성")
    ap.add_argument("--bbox", default="",
                    help="x1,y1,x2,y2 형식. 없으면 이미지 전체 사용")
    ap.add_argument("--probs", action="store_true",
                    help="분류 헤드별 softmax 확률 함께 출력")
    ap.add_argument("--tta", action="store_true",
                    help="Test-Time Augmentation (4 변형 평균). 정확도 +2~5%%, 추론 4배 느림")
    ap.add_argument("--out", default="",
                    help="결과 JSON 저장 경로 (기본: stdout 만)")
    args = ap.parse_args()

    if not args.checkpoint and not args.ensemble:
        ap.error("--checkpoint 또는 --ensemble 중 하나 필수")

    if args.ensemble:
        ckpts = [p.strip() for p in args.ensemble.split(",") if p.strip()]
        model = DamdaEnsembleModel(ckpts, config_path=args.config or None)
    else:
        model = DamdaInferenceModel(
            checkpoint_path=args.checkpoint,
            config_path=args.config or None,
        )

    sensor = _parse_sensor_args(args.sensor) if args.sensor else None
    categorical = _parse_categorical_args(args.categorical) if args.categorical else None
    bbox = _parse_bbox(args.bbox) if args.bbox else None

    result = model.predict(
        image_path=args.image,
        region=args.region,
        sensor=sensor,
        categorical=categorical,
        bbox=bbox,
        return_probs=args.probs,
        tta=args.tta,
    )

    # ---- 콘솔 출력 ----
    print()
    print(f"========== Inference ({result['meta']['region']}) ==========")
    print(f"  ckpt epoch     : {result['meta']['ckpt_epoch']}")
    print(f"  sensor used    : {result['meta']['sensor_inputs_used']} (provided: {list(sensor or {})})")
    print()
    print("  Regression (denormalized):")
    for name, val in result["regression"].items():
        print(f"    {name:24s} = {val:.3f}")
    print()
    print("  Classification (predicted class):")
    for name, cls in result["classification"].items():
        line = f"    {name:24s} = {cls}"
        if args.probs:
            probs = result["classification_probs"][name]
            top = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{i}:{p:.2f}" for i, p in top)
            line += f"   (top3: {top_str})"
        print(line)
    print()

    # ---- (선택) JSON 저장 ----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"결과 JSON 저장: {out_path}")


if __name__ == "__main__":
    main()
