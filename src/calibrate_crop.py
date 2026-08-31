"""calibrate_crop.py — AI-Hub 얼굴부위 크롭 스케일을 HW 접사에 맞춰 보정.

목적: crop-재학습(dataset.py 'macro' 모드) 전에, AI-Hub 부위 이미지를 '얼마나 작게 잘라
224로 확대'해야 실제 HW 접사(scan_images)와 텍스처 스케일이 맞는지 수치+눈으로 찾는다.

지표: 224 grayscale 의 FFT 스펙트럴 센트로이드(정규화 반경 가중 평균 주파수).
      값이 클수록 세밀(고주파 多). AI-Hub 패치를 작게 자를수록(=확대율↑) 세밀함이 줄어
      HW 의 낮은 센트로이드에 근접 → 두 분포가 겹치는 스케일을 고른다.

서버 실행 (AI-Hub·HW 둘 다 서버에 있음):
  python -m src.calibrate_crop ^
    --manifest data/manifest.csv ^
    --hw-root "C:\\Users\\DS\\Desktop\\HW\\scan_images" ^
    --scales 0.6,0.45,0.32,0.22,0.15,0.10 ^
    --out crop_calib.png
결과: 콘솔에 스케일별 센트로이드 vs HW + 추천 스케일, crop_calib.png 몽타주 저장.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# HW 폴더 부위 == 이 부위들만 비교 대상(AI-Hub도 동일 부위 샘플링)
TARGET_REGIONS = ["FOREHEAD", "GLABELLA", "L_CHEEK", "R_CHEEK", "CHIN"]


def spectral_centroid(gray224: Image.Image) -> float:
    a = np.asarray(gray224, dtype=np.float32)
    a = a - a.mean()
    F = np.fft.fftshift(np.fft.fft2(a))
    P = np.abs(F) ** 2
    h, w = P.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    P[cy, cx] = 0.0
    return float((r * P).sum() / (P.sum() + 1e-9) / r.max())


def region_crop(row) -> Optional[Image.Image]:
    """manifest 한 행 → bbox 부위 크롭 (dataset.py 와 동일 로직, 5% 패딩)."""
    try:
        img = Image.open(row["image_path"]).convert("RGB")
    except Exception:
        return None
    bx1, by1, bx2, by2 = row.get("bbox_x1"), row.get("bbox_y1"), row.get("bbox_x2"), row.get("bbox_y2")
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (bx1, by1, bx2, by2)):
        return None
    x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)
    w, h = x2 - x1, y2 - y1
    if w <= 4 or h <= 4:
        return None
    px, py = int(w * 0.05), int(h * 0.05)
    W, H = img.size
    return img.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))


def patch_from_region(region: Image.Image, scale: float, rng: random.Random) -> Image.Image:
    """부위 이미지에서 scale 비율의 정사각 패치를 무작위 위치로 떼어 224 로 확대."""
    w, h = region.size
    side = int(min(w, h) * scale)
    side = max(8, min(side, min(w, h)))
    left = rng.randint(0, max(0, w - side))
    top = rng.randint(0, max(0, h - side))
    patch = region.crop((left, top, left + side, top + side))
    return patch.resize((224, 224), Image.BILINEAR)


def load_hw(hw_root: str, n: int, rng: random.Random) -> List[Image.Image]:
    paths = sorted(glob.glob(os.path.join(hw_root, "WHITE_LED", "**", "*.jpg"), recursive=True))
    rng.shuffle(paths)
    out = []
    for p in paths[:n]:
        try:
            out.append(Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser(description="AI-Hub 크롭 스케일 ↔ HW 접사 보정")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--hw-root", required=True)
    ap.add_argument("--scales", default="0.6,0.45,0.32,0.22,0.15,0.10")
    ap.add_argument("--n-aihub", type=int, default=60, help="스케일별 AI-Hub 패치 표본 수")
    ap.add_argument("--n-hw", type=int, default=60)
    ap.add_argument("--out", default="crop_calib.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    scales = [float(s) for s in args.scales.split(",")]

    df = pd.read_csv(args.manifest)
    df = df[df["region"].isin(TARGET_REGIONS)].copy()
    # 부위 균형 샘플링 (얼굴 중복 허용, 충분한 표본)
    sample_rows = df.sample(min(len(df), args.n_aihub * 3), random_state=args.seed).to_dict("records")

    # 부위 크롭 캐시
    regions = []
    for row in sample_rows:
        rc = region_crop(row)
        if rc is not None:
            regions.append(rc)
        if len(regions) >= args.n_aihub:
            break
    if not regions:
        print("AI-Hub 부위 크롭 0건 — manifest image_path/bbox 확인.")
        return

    hw_imgs = load_hw(args.hw_root, args.n_hw, rng)
    if not hw_imgs:
        print("HW 이미지 0건 — --hw-root 확인.")
        return
    hw_cent = np.array([spectral_centroid(im.convert("L")) for im in hw_imgs])
    hw_mean = hw_cent.mean()

    print(f"HW WHITE 스펙트럴 센트로이드: mean={hw_mean:.4f} std={hw_cent.std():.4f} (n={len(hw_imgs)})\n")
    print(f"{'scale':>7}{'AIHub cent':>12}{'|Δ vs HW|':>11}  판정")
    results = []
    for s in scales:
        cents = [spectral_centroid(patch_from_region(rng.choice(regions), s, rng).convert("L"))
                 for _ in range(args.n_aihub)]
        m = float(np.mean(cents))
        d = abs(m - hw_mean)
        results.append((s, m, d))
        print(f"{s:>7.2f}{m:>12.4f}{d:>11.4f}")
    best = min(results, key=lambda r: r[2])
    print(f"\n★ 추천 크롭 스케일: {best[0]:.2f}  (AIHub cent {best[1]:.4f} ≈ HW {hw_mean:.4f})")
    print("  → dataset.py 'macro' 모드의 crop_scale 기본값으로 사용. 몽타주로 눈 확인도 필수.")

    # ---- 몽타주: 행=스케일, 열=예시 패치 5개 + 맨아래 HW 행 ----
    cols = 5
    cell = 150
    pad = 30
    rows = len(scales) + 1
    W = pad + cols * cell + 120
    H = pad + rows * cell + 30
    canvas = Image.new("RGB", (W, H), (24, 26, 32))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 8), f"AI-Hub crop scale vs HW (HW cent={hw_mean:.3f})", fill=(240, 240, 245))
    for ri, s in enumerate(scales):
        y = pad + ri * cell + 20
        d.text((pad + cols * cell + 8, y + cell // 2 - 8), f"s={s:.2f}", fill=(180, 200, 230))
        for ci in range(cols):
            patch = patch_from_region(rng.choice(regions), s, rng).resize((cell - 6, cell - 6))
            canvas.paste(patch, (pad + ci * cell, y))
    # HW 행
    y = pad + len(scales) * cell + 20
    d.text((pad + cols * cell + 8, y + cell // 2 - 8), "HW 실측", fill=(230, 180, 120))
    for ci in range(cols):
        if ci < len(hw_imgs):
            canvas.paste(hw_imgs[ci].resize((cell - 6, cell - 6)), (pad + ci * cell, y))
    canvas.save(args.out)
    print(f"\n몽타주 저장: {args.out}  — 맨아래 'HW 실측' 행과 텍스처 결이 가장 비슷한 스케일을 눈으로 확정.")


if __name__ == "__main__":
    main()
