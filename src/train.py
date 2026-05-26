"""학습 루프.

사용 예:
  # Baseline Validation (500장 / 10 epoch)
  python -m src.train --config configs/baseline.yaml --validation-mode
  # 본 학습
  python -m src.train --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import (
    DamdaSkinDataset,
    collate_fn,
    compute_class_weights,
    compute_regression_stats,
    split_manifest,
)
from .losses import multitask_loss
from .model import DamdaSkinModel
from .utils import device_supports_amp, get_device, set_seed, setup_logger


# ---------------- Helpers ----------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["training"]["optimizer"].lower()
    lr = cfg["training"]["lr"]
    wd = cfg["training"].get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(name)


def make_scheduler(optimizer, cfg: dict, num_epochs: int):
    name = cfg["training"].get("scheduler", "none").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(num_epochs // 3, 1), gamma=0.5)
    return None


# ---------------- Loops ----------------

def _forward_with_sensor(model, batch):
    """sensor_dim>0 일 때 sensor + sensor_mask 도 전달. v5+ Phase 2 통합."""
    if getattr(model, "sensor_dim", 0) > 0 and "sensor" in batch:
        return model(
            batch["image"], batch["region_id"],
            sensor=batch["sensor"],
            sensor_mask=batch.get("sensor_mask"),
        )
    return model(batch["image"], batch["region_id"])


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, writer, global_step, epoch,
                    class_weights=None, regression_targets=None):
    model.train()
    use_amp = scaler is not None
    pbar = tqdm(loader, desc=f"[train ep{epoch}]", ncols=100)
    for batch in pbar:
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                out = _forward_with_sensor(model, batch)
                loss, info = multitask_loss(
                    out, batch,
                    regression_weight=cfg["training"]["regression_weight"],
                    classification_weight=cfg["training"]["classification_weight"],
                    class_weights=class_weights,
                    regression_targets=regression_targets,
                    classification_loss_type=cfg["training"].get("classification_loss", "ce"),
                    focal_gamma=float(cfg["training"].get("focal_gamma", 2.0)),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = _forward_with_sensor(model, batch)
            loss, info = multitask_loss(
                out, batch,
                regression_weight=cfg["training"]["regression_weight"],
                classification_weight=cfg["training"]["classification_weight"],
                class_weights=class_weights,
                regression_targets=regression_targets,
                classification_loss_type=cfg["training"].get("classification_loss", "ce"),
                focal_gamma=float(cfg["training"].get("focal_gamma", 2.0)),
            )
            loss.backward()
            optimizer.step()

        if writer is not None:
            for k, v in info.items():
                writer.add_scalar(f"train/{k}", v, global_step)
        pbar.set_postfix(loss=f"{info['loss/total']:.4f}")
        global_step += 1
    return global_step


@torch.no_grad()
def evaluate(model, loader, device, cfg, writer, epoch, tag: str = "val",
             class_weights=None, regression_targets=None):
    model.eval()
    total_losses: Dict[str, float] = {}
    n = 0
    for batch in tqdm(loader, desc=f"[{tag} ep{epoch}]", ncols=100):
        batch = move_to_device(batch, device)
        out = _forward_with_sensor(model, batch)
        _, info = multitask_loss(
            out, batch,
            regression_weight=cfg["training"]["regression_weight"],
            classification_weight=cfg["training"]["classification_weight"],
            class_weights=class_weights,
            regression_targets=regression_targets,
            classification_loss_type=cfg["training"].get("classification_loss", "ce"),
            focal_gamma=float(cfg["training"].get("focal_gamma", 2.0)),
        )
        for k, v in info.items():
            total_losses[k] = total_losses.get(k, 0.0) + v
        n += 1

    avg = {k: v / max(n, 1) for k, v in total_losses.items()}
    if writer is not None:
        for k, v in avg.items():
            writer.add_scalar(f"{tag}/{k}", v, epoch)
    return avg


def move_to_device(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if torch.is_tensor(vv) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


# ---------------- Main ----------------

def find_latest_checkpoint(out_dir: Path):
    """checkpoints 폴더에서 가장 최근 epoch ckpt 반환 (없으면 None)."""
    if not out_dir.exists():
        return None
    ckpts = sorted(out_dir.glob("epoch*.pt"))
    return ckpts[-1] if ckpts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--validation-mode", action="store_true",
                    help="Baseline Validation: 소규모(500장) × 짧은 epoch으로 학습 가능성 검증")
    ap.add_argument("--resume", action="store_true",
                    help="checkpoints 폴더에서 최신 ckpt 자동 탐지 후 이어 학습")
    ap.add_argument("--resume-from", type=str, default="",
                    help="특정 ckpt 파일에서 이어 학습 (예: checkpoints/epoch015.pt)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["logging"]["log_dir"]) / ("baseline_val" if args.validation_mode else "main")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("train", str(log_dir))
    writer = SummaryWriter(str(log_dir))
    logger.info(f"device={device}, validation_mode={args.validation_mode}")

    # ----- Data -----
    manifest_path = cfg["data"]["manifest_path"]
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest 없음: {manifest_path}. 먼저 build_manifest.py 를 실행하세요.")
    df = pd.read_csv(manifest_path)
    logger.info(f"manifest 로드: {len(df)}행")

    if args.validation_mode:
        max_s = cfg["validation_mode"]["max_samples"]
        df = df.sample(n=min(max_s, len(df)), random_state=cfg.get("seed", 42)).reset_index(drop=True)
        logger.info(f"validation-mode 활성화 → 샘플 {len(df)}개로 축소")

    train_df, val_df, test_df = split_manifest(
        df,
        val_split=cfg["data"]["val_split"],
        test_split=cfg["data"]["test_split"],
        seed=cfg.get("seed", 42),
        split_by=cfg["data"].get("split_by", "id"),
    )
    logger.info(f"split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}")

    model_cfg = cfg["model"]
    regression_targets = model_cfg["regression_targets"]
    classification_heads = model_cfg["classification_heads"]

    # 회귀 타겟 정규화 통계 — 학습셋에서만 계산해 train/val 공유
    regression_stats = compute_regression_stats(train_df, regression_targets)
    logger.info("회귀 정규화 통계 (학습셋 기준):")
    for col, s in regression_stats.items():
        logger.info(f"  {col:22s} mean={s['mean']:.3f}  std={s['std']:.3f}")

    # 분류 헤드별 class_weights (use_class_weights=true 일 때만)
    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        class_weights = compute_class_weights(train_df, classification_heads)
        logger.info("class_weights (balanced, capped):")
        for col, w in class_weights.items():
            w_str = ", ".join(f"{x:.2f}" for x in w.tolist())
            logger.info(f"  {col:22s} [{w_str}]")

    # 분류 손실 종류 (v4~)
    _cls_loss = (cfg["training"].get("classification_loss", "ce") or "ce").lower()
    _focal_g  = float(cfg["training"].get("focal_gamma", 2.0))
    if _cls_loss == "focal":
        logger.info(f"분류 손실: focal (gamma={_focal_g})")
    else:
        logger.info("분류 손실: ce")

    # augment_mode: 'normal' (기본) | 'scanner' (ESP32-CAM 시연 환경 시뮬, v5+)
    augment_mode = cfg["data"].get("augment_mode", "normal")
    logger.info(f"augment_mode = {augment_mode}")

    # sensor 입력 (v5+, Phase 2). 빈 리스트면 sensor 비활성 → model.sensor_dim=0
    sensor_inputs = cfg["data"].get("sensor_inputs", []) or []
    sensor_dim = len(sensor_inputs)
    sensor_stats: Dict[str, Dict[str, float]] = {}
    if sensor_dim > 0:
        # 안전 체크: sensor_inputs 와 regression_targets 가 겹치면 data leakage
        # (모델이 sensor 입력값을 회귀 출력으로 그대로 echo 학습 → 추론 시 sensor 없으면 망함)
        overlap = set(sensor_inputs) & set(regression_targets)
        if overlap:
            raise ValueError(
                f"data leakage 가능성: 다음 컬럼이 sensor_inputs 와 regression_targets 양쪽에 있음: "
                f"{sorted(overlap)}. 한쪽에서 제거할 것 (시연용이면 regression_targets 에서 제거 권장)."
            )
        from .dataset import compute_sensor_stats
        sensor_stats = compute_sensor_stats(train_df, sensor_inputs)
        logger.info(f"sensor_inputs = {sensor_inputs}  (sensor_dim={sensor_dim})")
        for col, s in sensor_stats.items():
            logger.info(f"  {col:22s} mean={s['mean']:.3f}  std={s['std']:.3f}")
    else:
        logger.info("sensor_inputs 비활성 (sensor_dim=0)")

    train_ds = DamdaSkinDataset(train_df, regression_targets, classification_heads,
                                image_size=cfg["data"]["image_size"], train=True,
                                regression_stats=regression_stats,
                                augment_mode=augment_mode,
                                sensor_inputs=sensor_inputs,
                                sensor_stats=sensor_stats)
    val_ds   = DamdaSkinDataset(val_df,   regression_targets, classification_heads,
                                image_size=cfg["data"]["image_size"], train=False,
                                regression_stats=regression_stats,
                                augment_mode=augment_mode,
                                sensor_inputs=sensor_inputs,
                                sensor_stats=sensor_stats)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=cfg["data"]["num_workers"],
                              collate_fn=collate_fn, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=cfg["data"]["num_workers"],
                              collate_fn=collate_fn, pin_memory=pin)

    # ----- Model -----
    # sensor_dim 은 위에서 cfg["data"]["sensor_inputs"] 기반으로 계산됨. Phase 1 (v1~v4) = 0.
    model = DamdaSkinModel(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        num_regions=model_cfg["num_regions"],
        region_emb_dim=model_cfg["region_emb_dim"],
        regression_targets=regression_targets,
        classification_heads=classification_heads,
        dropout=model_cfg.get("dropout", 0.2),
        sensor_dim=sensor_dim,
        sensor_emb_dim=model_cfg.get("sensor_emb_dim", 32),
    ).to(device)
    logger.info(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optimizer = make_optimizer(model, cfg)
    num_epochs = cfg["validation_mode"]["epochs"] if args.validation_mode else cfg["training"]["epochs"]
    scheduler = make_scheduler(optimizer, cfg, num_epochs)

    # AMP는 CUDA 환경에서만
    use_amp = cfg["training"].get("amp", True) and device_supports_amp(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if not use_amp and device.type != "cuda":
        logger.info("AMP 비활성화 (CUDA 외 환경)")

    # ----- Resume from checkpoint (if requested) -----
    start_epoch = 1
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume:
        resume_path = find_latest_checkpoint(out_dir)

    if resume_path and resume_path.exists():
        logger.info(f"재개 모드: {resume_path} 에서 이어 학습")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # regression_stats 복원 (동일 seed면 어차피 같지만 안전)
        if "regression_stats" in ckpt:
            train_ds.regression_stats = ckpt["regression_stats"]
            val_ds.regression_stats = ckpt["regression_stats"]
            regression_stats = ckpt["regression_stats"]
        logger.info(f"resume 완료. epoch {start_epoch} 부터 진행")

    # ----- Train -----
    best_val = math.inf
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = cfg["training"].get("early_stop_patience", 0)  # 0 = 비활성
    global_step = 0
    for epoch in range(start_epoch, num_epochs + 1):
        global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, writer, global_step, epoch,
            class_weights=class_weights, regression_targets=regression_targets,
        )
        val_metrics = evaluate(model, val_loader, device, cfg, writer, epoch, tag="val",
                               class_weights=class_weights, regression_targets=regression_targets)
        cur_val = val_metrics.get("loss/total", math.inf)
        logger.info(f"[epoch {epoch}] val total={cur_val:.4f}")

        if scheduler is not None:
            scheduler.step()

        improved = cur_val < best_val
        if improved:
            best_val = cur_val
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        save_every = cfg["logging"].get("save_every", 5)
        if epoch % save_every == 0 or improved:
            ckpt_path = out_dir / f"epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
                "regression_stats": regression_stats,
                "sensor_stats": sensor_stats,
                "sensor_inputs": sensor_inputs,
            }, ckpt_path)
            logger.info(f"체크포인트 저장: {ckpt_path}")

        if early_stop_patience and patience_counter >= early_stop_patience:
            logger.info(
                f"Early stopping: {patience_counter} epochs 동안 val 개선 없음 "
                f"(best={best_val:.4f} @ epoch{best_epoch})"
            )
            break

    logger.info(f"학습 완료. best val loss = {best_val:.4f} @ epoch{best_epoch}")
    writer.close()


if __name__ == "__main__":
    main()
