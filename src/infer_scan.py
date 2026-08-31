"""infer_scan.py — scan_images(ESP32-CAM 실측) 트리에 학습 모델을 일괄 추론.

목적 (PROGRESS.md §6 "시연 후 작업"): AI-Hub 학습 앙상블(v3+v5.1)을 실물 매크로
스캔에 돌려 '도메인 갭으로 예측이 얼마나 무너지는지' before-number 를 확보한다.

⚠️ 주의: 이 모델은 AI-Hub '얼굴부위' 이미지로 학습됐고 scan_images 는 '접사(macro)'라
도메인이 다르다. 따라서 여기 나오는 예측값은 실제 피부수치가 아니라 '도메인 갭 진단용'
참고치다. 라벨이 없으므로 정답 대비 정확도는 못 구하고, 대신 (a) 예측 분포와
(b) 같은 부위 반복샷 간 예측 흔들림(불안정성) 으로 열화 정도를 가늠한다.

사용 (졸프실 서버, C:\\damda\\AI):
  # 단일 모델
  python -m src.infer_scan --root <scan_images> --ckpt checkpoints/epoch048.pt
  # 앙상블 (v3 + v5.1) + TTA — 시연 모델과 동일 구성
  python -m src.infer_scan --root <scan_images> ^
      --ckpt checkpoints_v3/epoch045.pt checkpoints_v5.1/epoch048.pt --tta
  # 반복샷 전부 추론 + 부위별 예측 불안정성 리포트
  python -m src.infer_scan --root <scan_images> --ckpt ... --frames all
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from statistics import mean, pstdev
from typing import Dict, List, Optional

from PIL import Image, ImageFilter

try:  # 패키지 실행 (python -m src.infer_scan)
    from .infer import DamdaEnsembleModel, DamdaInferenceModel
    from .utils import REGION_TO_ID, get_device, setup_logger
except ImportError:  # 단독 실행 폴백
    from infer import DamdaEnsembleModel, DamdaInferenceModel
    from utils import REGION_TO_ID, get_device, setup_logger


# 스캔 폴더 부위명 → 레포 표준 부위명 (스캔은 GLABELLA 대신 NOSE 폴더/라벨 사용)
SCAN_REGION_ALIAS = {"NOSE": "GLABELLA"}


def _sharpness(path: str) -> float:
    """초점 프록시 — grayscale edge 이미지의 분산 (PIL만 사용, cv2 불필요)."""
    try:
        edges = Image.open(path).convert("L").filter(ImageFilter.FIND_EDGES)
        hist = edges.histogram()
        total = sum(hist) or 1
        m = sum(i * h for i, h in enumerate(hist)) / total
        var = sum(((i - m) ** 2) * h for i, h in enumerate(hist)) / total
        return float(var)
    except Exception:
        return -1.0


def _select_best_frame(paths: List[str]) -> Optional[str]:
    if not paths:
        return None
    return max(paths, key=_sharpness)


def _iter_region_dirs(led_root: str):
    """led_root/{subject}/{region_folder} 를 순회."""
    if not os.path.isdir(led_root):
        return
    for subj in sorted(d for d in os.listdir(led_root) if os.path.isdir(os.path.join(led_root, d))):
        sdir = os.path.join(led_root, subj)
        for folder in sorted(d for d in os.listdir(sdir) if os.path.isdir(os.path.join(sdir, d))):
            yield subj, folder, os.path.join(sdir, folder)


def build_model(ckpts: List[str], config: Optional[str], device):
    if len(ckpts) == 1:
        model = DamdaInferenceModel(ckpts[0], config_path=config, device=device)
        reg_heads = model.regression_target_names()
        cls_heads = model.classification_head_names()
    else:
        model = DamdaEnsembleModel(ckpts, config_path=config, device=device)
        reg_heads = sorted(model.regression_targets)
        cls_heads = sorted(model.classification_heads)
    return model, reg_heads, cls_heads


def main():
    ap = argparse.ArgumentParser(description="scan_images 일괄 추론 (도메인 갭 진단용)")
    ap.add_argument("--root", required=True, help="scan_images 루트 (하위 WHITE_LED/, UV/)")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="체크포인트 1개(단일) 또는 여러 개(앙상블). 예: checkpoints/epoch048.pt")
    ap.add_argument("--config", default="configs/baseline.yaml",
                    help="환경 config (image_size 등). 모델 구조는 ckpt 우선이라 안전.")
    ap.add_argument("--led", default="WHITE_LED",
                    help="추론에 쓸 LED 폴더. 모델은 RGB(백색광)로 학습 → 기본 WHITE_LED. "
                         "UV는 별도 모달리티라 제외(넣으려면 UV 지정).")
    ap.add_argument("--frames", choices=["best", "all"], default="best",
                    help="best=부위별 최선명 1프레임 / all=모든 반복샷 + 부위별 예측 불안정성 리포트")
    ap.add_argument("--tta", action="store_true", help="강화 TTA 사용(시연 모델과 동일)")
    ap.add_argument("--out", default="scan_predictions.csv")
    ap.add_argument("--limit", type=int, default=0, help="개발용 부위 수 제한(0=전체)")
    args = ap.parse_args()

    logger = setup_logger("infer_scan")
    device = get_device()
    logger.info(f"device={device.type}  ckpt={args.ckpt}  led={args.led}  tta={args.tta}")
    logger.warning("주의: AI-Hub 얼굴부위 학습 모델을 접사(macro) 스캔에 추론 → 예측은 "
                   "실제 수치가 아니라 도메인 갭 진단용 참고치.")

    model, reg_heads, cls_heads = build_model(args.ckpt, args.config, device)
    logger.info(f"회귀 헤드: {reg_heads}")
    logger.info(f"분류 헤드: {cls_heads}")

    led_root = os.path.join(args.root, args.led)
    rows: List[dict] = []
    n_regions = 0
    for subj, folder, rdir in _iter_region_dirs(led_root):
        region = SCAN_REGION_ALIAS.get(folder, folder)
        if region not in REGION_TO_ID:
            logger.warning(f"건너뜀: 알 수 없는 부위 {folder!r} (subj={subj})")
            continue
        paths = sorted(glob.glob(os.path.join(rdir, "*.jpg")))
        if not paths:
            continue
        frames = [_select_best_frame(paths)] if args.frames == "best" else paths
        for fp in frames:
            try:
                out = model.predict(image_path=fp, region=region, tta=args.tta)
            except Exception as e:
                logger.warning(f"추론 실패 {fp}: {e}")
                continue
            row = {"subject": subj, "region": region, "region_id": REGION_TO_ID[region],
                   "frame": os.path.basename(fp), "sharpness": round(_sharpness(fp), 1)}
            for h in reg_heads:
                row[f"reg_{h}"] = round(float(out["regression"].get(h, float("nan"))), 4)
            for h in cls_heads:
                row[f"cls_{h}"] = out["classification"].get(h, "")
            rows.append(row)
        n_regions += 1
        if args.limit and n_regions >= args.limit:
            break

    if not rows:
        logger.error("추론 결과 0건. --root/--led 경로와 이미지 확인.")
        return

    fieldnames = (["subject", "region", "region_id", "frame", "sharpness"]
                  + [f"reg_{h}" for h in reg_heads] + [f"cls_{h}" for h in cls_heads])
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    logger.info(f"저장: {args.out}  ({len(rows)}행, 부위 {n_regions}개)")

    # --frames all: 부위별 반복샷 예측 불안정성(라벨 없는 열화 지표)
    if args.frames == "all":
        logger.info("=== 부위별 회귀 예측 불안정성 (반복샷 표준편차 / 평균, %) ===")
        groups: Dict[tuple, List[dict]] = {}
        for r in rows:
            groups.setdefault((r["subject"], r["region"]), []).append(r)
        cvs_by_head: Dict[str, List[float]] = {h: [] for h in reg_heads}
        for (subj, region), rs in sorted(groups.items()):
            if len(rs) < 2:
                continue
            parts = []
            for h in reg_heads:
                vals = [r[f"reg_{h}"] for r in rs]
                mu = mean(vals)
                cv = (pstdev(vals) / abs(mu) * 100) if mu else 0.0
                cvs_by_head[h].append(cv)
                parts.append(f"{h}={cv:.0f}%")
            logger.info(f"  {subj} {region:9s} n={len(rs)}  " + " ".join(parts))
        logger.info("--- 헤드별 평균 불안정성(높을수록 도메인 갭 큼) ---")
        for h in reg_heads:
            if cvs_by_head[h]:
                logger.info(f"  {h:20s} 평균 CV {mean(cvs_by_head[h]):.0f}%")


if __name__ == "__main__":
    main()
