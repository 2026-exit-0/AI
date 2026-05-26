"""held-out test set 평가.

train.py 와 동일한 split (seed=42, split_by='id') 로 test set 을 재현해
체크포인트 모델로 추론 → per-head MAE/RMSE/Pearson, Accuracy/F1/Confusion 계산.

dump_tb.py 가 학습 중 val loss 곡선을 보여준다면, 본 스크립트는 학습 종료 후
held-out 데이터에서 사람이 읽을 수 있는 메트릭 (MAE / Accuracy / F1 / Confusion)
을 뽑아 졸업논문 표 / v3 vs v4 비교에 그대로 쓸 수 있게 만든다.

사용 예 (lab PC cmd):
    # v4 본 학습 끝난 직후
    python -m src.evaluate ^
        --config configs/baseline.yaml ^
        --checkpoint checkpoints/epoch030.pt ^
        --split test ^
        --config-version v4 ^
        --out runs/eval/v4_test.json

    # v3 체크포인트도 남아있다면 동일하게 평가 후 diff
    python -m src.evaluate --checkpoint checkpoints/v3_epoch030.pt --config-version v3 ...
    python -m src.evaluate --checkpoint checkpoints/epoch030.pt --config-version v4 ^
        --compare-to runs/eval/v3_test.json

옵션:
    --no-per-region          : 부위(region) 별 슬라이스 비활성 (기본은 켜짐)
    --save-predictions PATH  : 샘플별 예측/정답 CSV 저장 (오류 사례 수동 분석용)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    DamdaSkinDataset,
    collate_fn,
    compute_regression_stats,
    split_manifest,
)
from .model import DamdaSkinModel
from .utils import ID_TO_REGION, get_device, set_seed, setup_logger


# sklearn 은 선택사항. 없으면 manual 폴백.
try:
    from sklearn.metrics import (
        confusion_matrix as sk_confusion_matrix,
        f1_score as sk_f1_score,
        precision_recall_fscore_support as sk_prfs,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================
# 메트릭 계산
# ============================================================

def regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    targets: List[str],
    stats: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """회귀 헤드별 메트릭.

    pred/target 은 정규화된 값 (학습/추론 공간). stats 로 denormalize 후
    원본 단위에서 MAE/RMSE/median_ae/Pearson r 계산.

    `mae_normalized` = 원본 MAE / std (헤드 간 비교용 무차원 값).
    """
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(targets):
        m = mask[:, i] > 0.5
        n_valid = int(m.sum())
        if n_valid == 0:
            out[name] = {"n_valid": 0}
            continue
        s = stats.get(name, {"mean": 0.0, "std": 1.0})
        std = max(float(s["std"]), 1e-6)
        p = pred[m, i] * std + float(s["mean"])
        t = target[m, i] * std + float(s["mean"])
        diff = p - t
        mae = float(np.mean(np.abs(diff)))
        out[name] = {
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "median_ae": float(np.median(np.abs(diff))),
            "pearson_r": float(np.corrcoef(p, t)[0, 1]) if n_valid > 1 else float("nan"),
            "n_valid": n_valid,
            "mae_normalized": mae / std,
            "target_mean": float(s["mean"]),
            "target_std": std,
        }

    valid_norm = [v["mae_normalized"] for v in out.values() if v.get("n_valid", 0) > 0]
    out["__aggregate__"] = {
        "mean_mae_normalized": float(np.mean(valid_norm)) if valid_norm else float("nan"),
        "n_heads_valid": len(valid_norm),
    }
    return out


def _manual_classification_metrics(t: np.ndarray, pred: np.ndarray, K: int):
    """sklearn 없을 때 폴백. macro/weighted F1, per-class P/R/F1, confusion 직접 계산."""
    cm = np.zeros((K, K), dtype=int)
    for ti, pi in zip(t, pred):
        if 0 <= int(ti) < K and 0 <= int(pi) < K:
            cm[int(ti), int(pi)] += 1

    per_class = []
    f1s, supports = [], []
    for c in range(K):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = int(cm[c, :].sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append({
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": support,
        })
        f1s.append(f1)
        supports.append(support)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    total_sup = sum(supports)
    weighted_f1 = float(
        sum(f * s for f, s in zip(f1s, supports)) / total_sup
    ) if total_sup > 0 else 0.0
    return macro_f1, weighted_f1, per_class, cm.tolist()


def classification_metrics(
    logits: np.ndarray, target: np.ndarray, num_classes: int
) -> Dict[str, object]:
    """분류 헤드 메트릭. target == -1 은 ignore."""
    valid = target != -1
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"n_valid": 0}

    p = logits[valid]
    t = target[valid].astype(int)
    pred = np.argmax(p, axis=1).astype(int)

    accuracy = float(np.mean(pred == t))

    if HAS_SKLEARN:
        labels = list(range(num_classes))
        macro_f1 = float(sk_f1_score(t, pred, labels=labels, average="macro", zero_division=0))
        weighted_f1 = float(sk_f1_score(t, pred, labels=labels, average="weighted", zero_division=0))
        prec, rec, f1, support = sk_prfs(
            t, pred, labels=labels, average=None, zero_division=0
        )
        per_class = [
            {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        ]
        cm = sk_confusion_matrix(t, pred, labels=labels).tolist()
    else:
        macro_f1, weighted_f1, per_class, cm = _manual_classification_metrics(t, pred, num_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "n_valid": n_valid,
    }


def compute_composite_score(reg_metrics: dict, cls_metrics: dict) -> float:
    """단일 비교 점수. 클수록 좋음.

    composite = mean_macro_F1 + (1 - mean_mae_normalized)
    """
    reg_norm = reg_metrics.get("__aggregate__", {}).get("mean_mae_normalized", float("nan"))
    cls_f1s = [
        v["macro_f1"] for v in cls_metrics.values()
        if isinstance(v, dict) and "macro_f1" in v
    ]
    cls_mean = float(np.mean(cls_f1s)) if cls_f1s else float("nan")
    if math.isnan(reg_norm) or math.isnan(cls_mean):
        return float("nan")
    return float(cls_mean + (1.0 - reg_norm))


# ============================================================
# 추론
# ============================================================

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    regression_targets: List[str],
    classification_heads: Dict[str, int],
) -> dict:
    """전체 loader 순회하며 예측/정답/마스크/region_id 누적."""
    model.eval()

    reg_preds, reg_targets, reg_masks = [], [], []
    cls_logits = {name: [] for name in classification_heads}
    cls_targets = {name: [] for name in classification_heads}
    region_ids = []

    for batch in tqdm(loader, desc="[eval]", ncols=100):
        img = batch["image"].to(device, non_blocking=True)
        rid = batch["region_id"].to(device, non_blocking=True)

        out = model(img, rid)

        if "regression" in out and regression_targets:
            reg_preds.append(out["regression"].cpu().numpy())
            reg_targets.append(batch["regression"].numpy())
            reg_masks.append(batch["regression_mask"].numpy())

        if "classification" in out:
            for name, logits in out["classification"].items():
                cls_logits[name].append(logits.cpu().numpy())
                cls_targets[name].append(batch["classification"][name].numpy())

        region_ids.append(batch["region_id"].numpy())

    return {
        "regression_pred": np.concatenate(reg_preds) if reg_preds else None,
        "regression_target": np.concatenate(reg_targets) if reg_targets else None,
        "regression_mask": np.concatenate(reg_masks) if reg_masks else None,
        "classification_logits": {n: np.concatenate(v) if v else np.zeros((0, 0))
                                  for n, v in cls_logits.items()},
        "classification_target": {n: np.concatenate(v) if v else np.zeros(0, dtype=int)
                                  for n, v in cls_targets.items()},
        "region_id": np.concatenate(region_ids) if region_ids else np.zeros(0, dtype=int),
    }


# ============================================================
# 보고서
# ============================================================

def _delta_arrow(delta: float, lower_is_better: bool, eps: float = 1e-4) -> str:
    if abs(delta) < eps:
        return "·"
    if lower_is_better:
        return "↓ (improved)" if delta < 0 else "↑ (regressed)"
    return "↑ (improved)" if delta > 0 else "↓ (regressed)"


def write_markdown_report(
    out_dict: dict, md_path: Path, compare_to: Optional[dict] = None
) -> None:
    """사람용 마크다운 리포트."""
    lines: List[str] = []
    meta = out_dict["meta"]
    lines.append(f"# Evaluation Report — {meta['split']} split")
    lines.append("")
    lines.append(f"- **Checkpoint**: `{meta['checkpoint']}` (epoch {meta['ckpt_epoch']})")
    lines.append(f"- **Samples**: {meta['n_samples']}")
    lines.append(f"- **Version label**: {meta.get('config_version', 'unknown')}")
    lines.append(f"- **Composite score**: **{out_dict['composite_score']:.4f}**  "
                 f"(macro F1 + (1 - normalized MAE), 클수록 좋음)")
    if not meta.get("sklearn_used", True):
        lines.append("- ⚠ scikit-learn 없음 → manual F1/confusion 사용")
    lines.append("")

    # ---- Regression ----
    lines.append("## Regression (denormalized)")
    lines.append("")
    lines.append("| Target | MAE | RMSE | Median AE | Pearson r | N | MAE/σ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in out_dict["regression"].items():
        if name == "__aggregate__":
            continue
        if m.get("n_valid", 0) == 0:
            lines.append(f"| {name} | — | — | — | — | 0 | — |")
            continue
        lines.append(
            f"| {name} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['median_ae']:.3f} | "
            f"{m['pearson_r']:.3f} | {m['n_valid']} | {m['mae_normalized']:.3f} |"
        )
    agg = out_dict["regression"].get("__aggregate__", {})
    if "mean_mae_normalized" in agg:
        lines.append("")
        lines.append(f"Mean MAE / σ: **{agg['mean_mae_normalized']:.3f}** "
                     f"({agg.get('n_heads_valid', 0)} heads valid)")
    lines.append("")

    # ---- Classification ----
    lines.append("## Classification")
    lines.append("")
    lines.append("| Head | Accuracy | Macro F1 | Weighted F1 | N |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in out_dict["classification"].items():
        if not isinstance(m, dict) or m.get("n_valid", 0) == 0:
            lines.append(f"| {name} | — | — | — | 0 |")
            continue
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['weighted_f1']:.3f} | {m['n_valid']} |"
        )
    f1s = [v["macro_f1"] for v in out_dict["classification"].values()
           if isinstance(v, dict) and "macro_f1" in v]
    if f1s:
        lines.append("")
        lines.append(f"Mean macro F1: **{float(np.mean(f1s)):.3f}**")
    lines.append("")

    # ---- Per-region ----
    if "per_region" in out_dict:
        lines.append("## Per-region")
        lines.append("")
        lines.append("| Region | N | Reg MAE/σ | Cls macro F1 |")
        lines.append("|---|---:|---:|---:|")
        for rname, data in out_dict["per_region"].items():
            rn = data["regression"].get("__aggregate__", {}).get("mean_mae_normalized", float("nan"))
            rf = [v["macro_f1"] for v in data["classification"].values()
                  if isinstance(v, dict) and "macro_f1" in v]
            cf = float(np.mean(rf)) if rf else float("nan")
            lines.append(f"| {rname} | {data['n_samples']} | {rn:.3f} | {cf:.3f} |")
        lines.append("")

    # ---- Comparison ----
    if compare_to is not None:
        lines.append("## Comparison vs baseline")
        lines.append("")
        lines.append(f"Baseline: `{compare_to.get('meta', {}).get('checkpoint', '?')}` "
                     f"(version: {compare_to.get('meta', {}).get('config_version', '?')})")
        lines.append("")
        lines.append("| Metric | Baseline | Current | Δ | Direction |")
        lines.append("|---|---:|---:|---:|---|")
        # Regression MAE (lower is better)
        for name, cur in out_dict["regression"].items():
            if name == "__aggregate__":
                continue
            ref = compare_to.get("regression", {}).get(name, {})
            if "mae" in cur and "mae" in ref:
                d = cur["mae"] - ref["mae"]
                lines.append(
                    f"| reg/{name}/mae | {ref['mae']:.3f} | {cur['mae']:.3f} | "
                    f"{d:+.3f} | {_delta_arrow(d, lower_is_better=True)} |"
                )
        # Classification macro F1 (higher is better)
        for name, cur in out_dict["classification"].items():
            ref = compare_to.get("classification", {}).get(name, {})
            if isinstance(cur, dict) and isinstance(ref, dict) \
                    and "macro_f1" in cur and "macro_f1" in ref:
                d = cur["macro_f1"] - ref["macro_f1"]
                lines.append(
                    f"| cls/{name}/macro_f1 | {ref['macro_f1']:.3f} | {cur['macro_f1']:.3f} | "
                    f"{d:+.3f} | {_delta_arrow(d, lower_is_better=False)} |"
                )
        # Composite
        if "composite_score" in out_dict and "composite_score" in compare_to:
            d = out_dict["composite_score"] - compare_to["composite_score"]
            lines.append(
                f"| **composite** | **{compare_to['composite_score']:.4f}** | "
                f"**{out_dict['composite_score']:.4f}** | **{d:+.4f}** | "
                f"**{_delta_arrow(d, lower_is_better=False)}** |"
            )
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 메인
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="학습에 쓰인 config (split/모델 구조 일치 필수)")
    ap.add_argument("--checkpoint", required=True, help="평가할 체크포인트 .pt 경로")
    ap.add_argument("--split", choices=["test", "val", "train"], default="test",
                    help="평가할 split (기본 test). val 은 sanity check 용")
    ap.add_argument("--out", default="",
                    help="결과 JSON 경로 (기본: runs/eval/<ckpt_stem>_<split>.json)")
    ap.add_argument("--no-per-region", action="store_true",
                    help="region 별 슬라이스 비활성 (기본은 ON)")
    ap.add_argument("--save-predictions", default="",
                    help="샘플별 예측/정답 CSV 저장 경로 (오류 사례 수동 분석용)")
    ap.add_argument("--compare-to", default="",
                    help="비교할 baseline 결과 JSON 경로 (diff 표 자동 생성)")
    ap.add_argument("--config-version", default="",
                    help="결과 메타에 저장할 버전 라벨 (예: v3, v4)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))
    device = get_device()

    # 출력 경로
    ckpt_stem = Path(args.checkpoint).stem
    out_path = Path(args.out) if args.out else \
        Path("runs/eval") / f"{ckpt_stem}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm_dir = out_path.parent / f"{out_path.stem}_cm"
    cm_dir.mkdir(parents=True, exist_ok=True)

    # 로거 (stdout 만 — train.log 오염 방지)
    logger = setup_logger("evaluate")
    logger.info(f"device={device}  split={args.split}  ckpt={args.checkpoint}")
    if not HAS_SKLEARN:
        logger.info("scikit-learn 미설치 → manual F1/confusion 폴백")

    # ---- 데이터 분할 (train.py 와 동일한 seed/split_by) ----
    manifest_path = cfg["data"]["manifest_path"]
    df = pd.read_csv(manifest_path)
    train_df, val_df, test_df = split_manifest(
        df,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg.get("seed", 42),
        split_by=cfg["data"].get("split_by", "id"),
    )
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    target_df = split_map[args.split].reset_index(drop=True)
    logger.info(f"manifest={len(df)}행 → split={args.split} → {len(target_df)}샘플")

    # ---- 모델 구성 ----
    model_cfg = cfg["model"]
    regression_targets = model_cfg["regression_targets"]
    classification_heads = model_cfg["classification_heads"]

    model = DamdaSkinModel(
        backbone=model_cfg["backbone"],
        pretrained=False,  # 어차피 ckpt 로 덮어쓸 거라 ImageNet 다운로드 불필요
        num_regions=model_cfg["num_regions"],
        region_emb_dim=model_cfg["region_emb_dim"],
        regression_targets=regression_targets,
        classification_heads=classification_heads,
        dropout=model_cfg.get("dropout", 0.2),
        sensor_dim=0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    ckpt_epoch = int(ckpt.get("epoch", -1))
    logger.info(f"checkpoint 로드 완료 (epoch={ckpt_epoch})")

    # regression_stats — ckpt 우선, 없으면 train_df 로 재계산
    if "regression_stats" in ckpt:
        regression_stats = ckpt["regression_stats"]
        logger.info("regression_stats 복원 (ckpt 내부)")
    else:
        regression_stats = compute_regression_stats(train_df, regression_targets)
        logger.info("regression_stats 재계산 (ckpt 에 없음 — train_df 기준)")

    # ---- Dataset / DataLoader (train=False, no shuffle) ----
    ds = DamdaSkinDataset(
        target_df, regression_targets, classification_heads,
        image_size=cfg["data"]["image_size"], train=False,
        regression_stats=regression_stats,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ---- 추론 ----
    bundle = run_inference(model, loader, device, regression_targets, classification_heads)

    # ---- 전체 메트릭 ----
    reg_metrics = regression_metrics(
        bundle["regression_pred"], bundle["regression_target"], bundle["regression_mask"],
        regression_targets, regression_stats,
    ) if bundle["regression_pred"] is not None else {}

    cls_metrics: Dict[str, dict] = {}
    for name, K in classification_heads.items():
        cm = classification_metrics(
            bundle["classification_logits"][name],
            bundle["classification_target"][name],
            K,
        )
        cls_metrics[name] = cm
        # Confusion matrix CSV 저장
        if "confusion_matrix" in cm:
            arr = np.array(cm["confusion_matrix"], dtype=int)
            np.savetxt(cm_dir / f"{name}.csv", arr, fmt="%d", delimiter=",")

    composite = compute_composite_score(reg_metrics, cls_metrics)

    out_dict = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "n_samples": int(len(target_df)),
            "ckpt_epoch": ckpt_epoch,
            "config_version": args.config_version or "unknown",
            "sklearn_used": HAS_SKLEARN,
        },
        "regression": reg_metrics,
        "classification": cls_metrics,
        "composite_score": composite,
    }

    # ---- Per-region 슬라이스 ----
    if not args.no_per_region:
        per_region: Dict[str, dict] = {}
        region_ids_arr = bundle["region_id"].astype(int)
        for rid in range(model_cfg["num_regions"]):
            sel = region_ids_arr == rid
            if sel.sum() == 0:
                continue
            rname = ID_TO_REGION.get(rid, f"REGION_{rid}")
            rm = regression_metrics(
                bundle["regression_pred"][sel],
                bundle["regression_target"][sel],
                bundle["regression_mask"][sel],
                regression_targets, regression_stats,
            ) if bundle["regression_pred"] is not None else {}
            cm = {
                name: classification_metrics(
                    bundle["classification_logits"][name][sel],
                    bundle["classification_target"][name][sel],
                    classification_heads[name],
                )
                for name in classification_heads
            }
            per_region[rname] = {
                "n_samples": int(sel.sum()),
                "regression": rm,
                "classification": cm,
            }
        out_dict["per_region"] = per_region

    # ---- JSON 저장 ----
    out_path.write_text(
        json.dumps(out_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"JSON 저장: {out_path}")
    logger.info(f"Confusion CSV: {cm_dir}/")

    # ---- 샘플별 예측 CSV (선택) ----
    if args.save_predictions:
        pred_path = Path(args.save_predictions)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        cols: Dict[str, np.ndarray] = {
            "region_id": bundle["region_id"].astype(int),
            "region": np.array([ID_TO_REGION.get(int(r), "?") for r in bundle["region_id"]]),
        }
        # subject_id 가 target_df 에 있고 row 순서가 일치 (shuffle=False) 하므로 안전
        if "subject_id" in target_df.columns and len(target_df) == len(bundle["region_id"]):
            cols["subject_id"] = target_df["subject_id"].values
        if bundle["regression_pred"] is not None:
            for i, name in enumerate(regression_targets):
                s = regression_stats.get(name, {"mean": 0.0, "std": 1.0})
                std = max(float(s["std"]), 1e-6)
                cols[f"pred_{name}"] = bundle["regression_pred"][:, i] * std + float(s["mean"])
                cols[f"true_{name}"] = bundle["regression_target"][:, i] * std + float(s["mean"])
                cols[f"mask_{name}"] = bundle["regression_mask"][:, i]
        for name in classification_heads:
            cols[f"pred_cls_{name}"] = np.argmax(bundle["classification_logits"][name], axis=1)
            cols[f"true_cls_{name}"] = bundle["classification_target"][name]
        pd.DataFrame(cols).to_csv(pred_path, index=False)
        logger.info(f"predictions CSV: {pred_path}")

    # ---- 마크다운 리포트 ----
    md_path = out_path.with_suffix(".md")
    compare_dict = None
    if args.compare_to:
        cp = Path(args.compare_to)
        if cp.exists():
            compare_dict = json.loads(cp.read_text(encoding="utf-8"))
            logger.info(f"비교 baseline 로드: {cp}")
        else:
            logger.info(f"⚠ --compare-to 파일 없음, 비교 스킵: {cp}")
    write_markdown_report(out_dict, md_path, compare_to=compare_dict)
    logger.info(f"Markdown 보고서: {md_path}")

    # ---- 콘솔 요약 ----
    print()
    print(f"========== Eval summary ({args.split}, {args.config_version or '?'}) ==========")
    print(f"  Composite score        : {composite:.4f}")
    reg_norm = reg_metrics.get("__aggregate__", {}).get("mean_mae_normalized", float("nan"))
    print(f"  Reg mean MAE / σ       : {reg_norm:.3f}")
    cls_f1s = [v["macro_f1"] for v in cls_metrics.values()
               if isinstance(v, dict) and "macro_f1" in v]
    if cls_f1s:
        print(f"  Cls mean macro F1      : {float(np.mean(cls_f1s)):.3f}")
    if compare_dict is not None:
        d = composite - compare_dict.get("composite_score", composite)
        print(f"  Composite Δ vs baseline: {d:+.4f}")
    print()


if __name__ == "__main__":
    main()
