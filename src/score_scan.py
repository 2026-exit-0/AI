"""score_scan.py — 담다 매크로 스캔 '실측 표면 스코어' 산출 (신뢰 지표만).

surface_features 의 원지수를 받아, 실제로 신호가 잡히는 것만 골라
사용자에게 보여줄 3-band 스코어로 정리한다.

포함(신뢰):
  - pigment_index (0~100)  : UV 색소(암부율)+불균일을 코호트 z-score 로 합쳐 백분위화.
                             현재 가장 변별력 있는 지표.
  - texture_band           : WHITE 표면 텍스처. 초점 ok 인 스캔만 채움(아니면 공백).
제외(현 데이터 신뢰 부족): pore, porphyrin (unusable), redness(약함) → notes 로만.

라벨이 없으므로 절대 임상수치가 아니라 '코호트 내 상대 지수'다.
"""
from __future__ import annotations

import argparse
import csv
from typing import Dict, List

import numpy as np

try:
    from .surface_features import process_tree, fit_cohort_thresholds, to_band, SurfaceScan
except ImportError:
    from surface_features import process_tree, fit_cohort_thresholds, to_band, SurfaceScan

BANDS = ["양호", "보통", "주의"]


def _zscores(vals: List[float]) -> np.ndarray:
    a = np.array(vals, dtype=float)
    mu, sd = a.mean(), a.std()
    return (a - mu) / (sd if sd > 1e-9 else 1.0)


def compute_pigment_index(scans: List[SurfaceScan]) -> Dict[int, float]:
    """pigment + heterogeneity 를 z-score 평균 → 코호트 백분위(0~100). 높을수록 색소 많음(주의)."""
    idxs = [i for i, s in enumerate(scans)
            if s.pigment is not None and s.heterogeneity is not None]
    if not idxs:
        return {}
    zp = _zscores([scans[i].pigment for i in idxs])
    zh = _zscores([scans[i].heterogeneity for i in idxs])
    composite = (zp + zh) / 2.0
    order = np.argsort(np.argsort(composite))         # 0..n-1 순위
    pct = order / max(len(order) - 1, 1) * 100.0       # 0~100 백분위
    return {idxs[k]: float(pct[k]) for k in range(len(idxs))}


def band_from_percentile(p: float) -> str:
    return BANDS[0] if p < 33.3 else (BANDS[1] if p < 66.7 else BANDS[2])


def main():
    ap = argparse.ArgumentParser(description="매크로 스캔 실측 표면 스코어(신뢰 지표만)")
    ap.add_argument("--root", required=True, help="scan_images 루트 (WHITE_LED/, UV/)")
    ap.add_argument("--out", default="surface_scores_clean.csv")
    args = ap.parse_args()

    scans = process_tree(args.root)
    if not scans:
        print("스캔 0건 — 경로 확인.")
        return

    pig_idx = compute_pigment_index(scans)
    tex_th = fit_cohort_thresholds(scans, ["texture"])

    rows = []
    for i, s in enumerate(scans):
        p = pig_idx.get(i)
        # texture 는 신뢰(ok)일 때만 표기
        tex_ok = s.confidence.get("texture") == "ok"
        rows.append({
            "subject": s.subject, "region": s.region,
            "pigment_index": round(p, 1) if p is not None else "",
            "pigment_band": band_from_percentile(p) if p is not None else "",
            "texture": round(s.texture, 2) if (s.texture is not None and tex_ok) else "",
            "texture_band": (to_band("texture", s.texture, tex_th) if tex_ok else ""),
            "texture_conf": s.confidence.get("texture", ""),
            "notes": "; ".join(s.quality_notes),
        })

    fields = ["subject", "region", "pigment_index", "pigment_band",
              "texture", "texture_band", "texture_conf", "notes"]
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"저장: {args.out}  ({len(rows)}건)")

    # 변별력 요약 — 붕괴 아님을 정직하게 보여줌
    from collections import Counter
    print("\n색소지수 band 분포:", dict(Counter(r["pigment_band"] for r in rows if r["pigment_band"])))
    by_subj: Dict[str, List[float]] = {}
    for i, s in enumerate(scans):
        if i in pig_idx:
            by_subj.setdefault(s.subject, []).append(pig_idx[i])
    print("사람별 색소지수 평균:")
    for k in sorted(by_subj):
        v = by_subj[k]
        print(f"  {k}: {np.mean(v):.0f}  (부위 {len(v)}개)")
    print("texture band 분포:", dict(Counter(r["texture_band"] for r in rows if r["texture_band"])))


if __name__ == "__main__":
    main()
