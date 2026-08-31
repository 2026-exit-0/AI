"""surface_features.py — 담다: 접사(macro) ESP32-CAM 스캔에서 표면 지표 추출.

배경 (PROGRESS.md / 도메인갭 진단, 2026-08):
  실측 스캔(scan_images)은 AI-Hub 얼굴부위 학습데이터와 달리 '접사(contact-macro)'다.
  v3+v5.1 앙상블(infer.py)은 얼굴부위 FOV를 전제하므로 이 매크로 이미지에는 그대로
  전이되지 않는다(scanner_aug는 해상도/압축만 흉내낼 뿐 화각을 못 메움).
  → 이 모듈은 매크로에서 실제로 추출 가능한 '표면 지표'를 해석가능한 방식으로 뽑는다.

원칙:
  - 라벨이 없으므로 절대 임상수치가 아니라 코호트 분위수 기반 상대 3-band(양호/보통/주의).
  - 화질(초점) 한계가 크므로 지표마다 신뢰도 플래그를 함께 반환. 약한 신호를 강조하지 않는다.
  - infer.py 의 12지표와는 별개 경로(매크로 전용). BE/FE 연동은 demo_provider(시연) 또는
    별도 매핑을 통한다.

의존성: numpy, opencv-python(-headless).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("opencv-python(-headless) 필요") from e

# 부위 매핑은 레포 표준(utils)을 재사용. 패키지/단독 실행 모두 지원.
try:
    from .utils import REGION_TO_ID
except ImportError:  # 단독 실행
    try:
        from utils import REGION_TO_ID
    except ImportError:
        REGION_TO_ID = {"PART_0": 0, "FOREHEAD": 1, "GLABELLA": 2, "L_EYE": 3,
                        "R_EYE": 4, "L_CHEEK": 5, "R_CHEEK": 6, "LIP": 7, "CHIN": 8}


# 스캔 폴더 부위명 → 레포 표준 부위명 (스캔은 GLABELLA 대신 NOSE 폴더/라벨 혼용)
SCAN_REGION_ALIAS = {"NOSE": "GLABELLA"}

# 품질 임계값 (실측 표본 기준 보수값 — 데이터 늘면 재조정)
SHARP_MIN_OK = 6.0
OVEREXPOSE_FRAC_MAX = 0.05
CENTER_CROP_FRAC = 0.70


@dataclass
class SurfaceScan:
    subject: str
    region: str                            # 레포 표준 부위명 (GLABELLA 등)
    region_id: Optional[int] = None
    pigment: Optional[float] = None        # UV 멜라닌 암부율 %
    heterogeneity: Optional[float] = None  # UV 휘도 표준편차
    porphyrin: Optional[float] = None      # UV 적색형광 %
    texture: Optional[float] = None        # WHITE 국소표준편차
    redness: Optional[float] = None        # WHITE (R-G)/(R+G) %
    pore: Optional[float] = None           # WHITE 소형 암부 면적 %
    white_sharpness: Optional[float] = None
    uv_sharpness: Optional[float] = None
    confidence: Dict[str, str] = field(default_factory=dict)  # 지표→ ok|low|unusable
    quality_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
def _center_crop(img: np.ndarray, frac: float = CENTER_CROP_FRAC) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y, x = (h - ch) // 2, (w - cw) // 2
    return img[y:y + ch, x:x + cw]


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frame(paths: List[str]) -> Optional[str]:
    """반복샷 중 가장 선명한 프레임 선택(초점이 샷마다 출렁이므로 유효)."""
    best, best_s = None, -1.0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        s = _sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if s > best_s:
            best, best_s = p, s
    return best


def extract_white(bgr: np.ndarray) -> Dict[str, float]:
    crop = _center_crop(bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(crop.astype(np.float32))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    pore = float((blackhat > 12).mean() * 100.0)
    gf = gray.astype(np.float32)
    mean = cv2.blur(gf, (7, 7))
    local_std = np.sqrt(np.maximum(cv2.blur(gf * gf, (7, 7)) - mean * mean, 0.0))
    texture = float(local_std.mean())
    redness = float(((r - g) / (r + g + 1e-6)).mean() * 100.0)
    over = float((gray >= 250).mean())
    return dict(texture=texture, redness=redness, pore=pore,
                sharpness=_sharpness(gray), overexpose=over)


def extract_uv(bgr: np.ndarray) -> Dict[str, float]:
    crop = _center_crop(bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lum = gray.astype(np.float32)
    b, g, r = cv2.split(crop.astype(np.float32))
    thr = lum.mean() - lum.std()
    pigment = float((lum < thr).mean() * 100.0)
    heterogeneity = float(lum.std())
    porphyrin = float(((r > g + 12) & (r > b)).mean() * 100.0)
    return dict(pigment=pigment, heterogeneity=heterogeneity,
                porphyrin=porphyrin, sharpness=_sharpness(gray))


def process_scan(white_path: Optional[str], uv_path: Optional[str],
                 subject: str = "", region: str = "") -> SurfaceScan:
    region = SCAN_REGION_ALIAS.get(region, region)
    scan = SurfaceScan(subject=subject, region=region, region_id=REGION_TO_ID.get(region))

    if white_path:
        wimg = cv2.imread(white_path)
        if wimg is not None:
            wf = extract_white(wimg)
            scan.texture, scan.redness, scan.pore = wf["texture"], wf["redness"], wf["pore"]
            scan.white_sharpness = wf["sharpness"]
            reasons = []
            if wf["sharpness"] < SHARP_MIN_OK:
                reasons.append(f"초점부족(Lap={wf['sharpness']:.1f}<{SHARP_MIN_OK})")
            if wf["overexpose"] > OVEREXPOSE_FRAC_MAX:
                reasons.append(f"과노출({wf['overexpose']*100:.0f}%)")
            # pore 는 실측상 노이즈 수준 → 초점 좋아도 최대 'low', 나쁘면 'unusable'
            scan.confidence["pore"] = "low" if not reasons else "unusable"
            scan.confidence["texture"] = "ok" if not reasons else "low"
            scan.confidence["redness"] = "low"
            scan.quality_notes += reasons

    if uv_path:
        uimg = cv2.imread(uv_path)
        if uimg is not None:
            uf = extract_uv(uimg)
            scan.pigment, scan.heterogeneity, scan.porphyrin = (
                uf["pigment"], uf["heterogeneity"], uf["porphyrin"])
            scan.uv_sharpness = uf["sharpness"]
            scan.confidence["pigment"] = "ok"
            scan.confidence["heterogeneity"] = "ok"
            scan.confidence["porphyrin"] = "unusable" if (scan.porphyrin or 0) < 0.3 else "low"

    return scan


# ---------------------------------------------------------------------------
HIGHER_IS_WORSE = {"pigment": True, "heterogeneity": True, "porphyrin": True,
                   "pore": True, "redness": True, "texture": False}
BANDS = ["양호", "보통", "주의"]


def fit_cohort_thresholds(scans: List[SurfaceScan], metrics: List[str]) -> Dict[str, Tuple[float, float]]:
    th = {}
    for m in metrics:
        vals = [getattr(s, m) for s in scans
                if getattr(s, m) is not None and s.confidence.get(m) != "unusable"]
        if len(vals) >= 3:
            th[m] = (float(np.percentile(vals, 33)), float(np.percentile(vals, 67)))
    return th


def to_band(metric: str, value: Optional[float], thresholds: Dict[str, Tuple[float, float]]) -> Optional[str]:
    if value is None or metric not in thresholds:
        return None
    lo, hi = thresholds[metric]
    idx = 0 if value <= lo else (1 if value <= hi else 2)
    if not HIGHER_IS_WORSE.get(metric, True):
        idx = 2 - idx
    return BANDS[idx]


def process_tree(root: str) -> List[SurfaceScan]:
    """scan_images/{WHITE_LED,UV}/{subj}/{region_folder}/*.jpg → (subj,region)별 1스캔."""
    scans: List[SurfaceScan] = []
    white_root = os.path.join(root, "WHITE_LED")
    if not os.path.isdir(white_root):
        return scans
    for subj in sorted(d for d in os.listdir(white_root) if os.path.isdir(os.path.join(white_root, d))):
        wsubj = os.path.join(white_root, subj)
        for folder in sorted(os.listdir(wsubj)):
            wdir = os.path.join(wsubj, folder)
            if not os.path.isdir(wdir):
                continue
            wbest = select_best_frame(sorted(glob.glob(os.path.join(wdir, "*.jpg"))))
            ubest = select_best_frame(sorted(glob.glob(os.path.join(root, "UV", subj, folder, "*.jpg"))))
            scans.append(process_scan(wbest, ubest, subject=subj, region=folder))
    return scans


def main():
    import argparse, csv
    from collections import Counter
    ap = argparse.ArgumentParser(description="접사 스캔 표면 지표 추출")
    ap.add_argument("--root", required=True, help="scan_images 루트 (하위 WHITE_LED/, UV/)")
    ap.add_argument("--out", default="surface_scores.csv")
    args = ap.parse_args()

    scans = process_tree(args.root)
    metrics = ["pigment", "heterogeneity", "texture", "redness", "pore", "porphyrin"]
    th = fit_cohort_thresholds(scans, metrics)

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["subject", "region", "region_id"] + metrics
                   + [f"{m}_band" for m in metrics] + [f"{m}_conf" for m in metrics]
                   + ["white_sharp", "uv_sharp", "notes"])
        for s in scans:
            w.writerow([s.subject, s.region, s.region_id]
                       + [round(getattr(s, m), 2) if getattr(s, m) is not None else "" for m in metrics]
                       + [to_band(m, getattr(s, m), th) or "" for m in metrics]
                       + [s.confidence.get(m, "") for m in metrics]
                       + [round(s.white_sharpness or 0, 1), round(s.uv_sharpness or 0, 1),
                          "; ".join(s.quality_notes)])

    print(f"저장: {args.out}  (스캔 {len(scans)}건)")
    for m in metrics:
        print(f"  {m:14s} 신뢰도 {dict(Counter(s.confidence.get(m,'na') for s in scans))}")


if __name__ == "__main__":
    main()
