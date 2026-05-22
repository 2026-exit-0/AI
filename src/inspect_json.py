"""원본 AI-Hub JSON 1~N 개를 까서 equipment/annotations 의 실제 키 목록을 출력.

manifest의 pigmentation_value / wrinkle_value 가 통째로 결측인 원인을 진단하기 위해
실제 JSON에 어떤 키가 있는지 직접 확인한다.

사용 예 (cmd, 인자 없이 자동 탐색):
    python -m src.inspect_json

명시적으로 경로 지정하고 싶으면:
    python -m src.inspect_json --json-root "C:\\damda\\dataset\\...\\TL"
    python -m src.inspect_json --n 20
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional


def auto_find_tl_roots() -> List[Path]:
    """C:\\damda\\dataset 와 그 주변에서 'TL' 폴더를 자동 탐색."""
    candidates_search = [
        Path("C:/damda/dataset"),
        Path("C:/damda"),
        Path("D:/damda/dataset"),
        Path("D:/damda"),
        Path.cwd().parent / "dataset",
        Path.cwd() / "dataset",
    ]
    found: List[Path] = []
    seen = set()
    for base in candidates_search:
        if not base.exists():
            continue
        try:
            for p in base.rglob("TL"):
                if p.is_dir() and str(p) not in seen:
                    found.append(p)
                    seen.add(str(p))
                    if len(found) >= 5:  # 너무 깊이 안 들어가게
                        return found
        except (PermissionError, OSError):
            continue
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", default="",
                    help="TL 라벨 루트 (생략 시 자동 탐색)")
    ap.add_argument("--n", type=int, default=10,
                    help="샘플링할 JSON 개수 (기본 10)")
    args = ap.parse_args()

    # ---- TL 루트 결정 ----
    if args.json_root:
        root = Path(args.json_root)
        if not root.exists():
            print(f"[ERROR] 지정한 경로 없음: {root}")
            return
    else:
        print("--json-root 미지정 → 자동 탐색 중...")
        cands = auto_find_tl_roots()
        if not cands:
            print("[ERROR] TL 폴더를 찾지 못했습니다.")
            print("        C:\\damda\\dataset 또는 그 주변에 TL 폴더가 있는지 확인하세요.")
            print("        또는 --json-root 로 직접 지정해주세요.")
            return
        print("TL 후보:")
        for i, c in enumerate(cands):
            print(f"  [{i}] {c}")
        root = cands[0]
        print(f"\n→ 첫 번째 후보 사용: {root}\n")

    # ---- JSON 샘플링 ----
    print(f"JSON 검색 중: {root}")
    all_jsons = list(root.rglob("*.json"))
    print(f"총 JSON 파일: {len(all_jsons)}")
    if not all_jsons:
        print("[ERROR] JSON이 하나도 없습니다.")
        return

    # facepart 다양성 확보: _01, _02, _05, _06, _08 우선
    diversified: List[Path] = []
    for nn in ["01", "02", "03", "04", "05", "06", "07", "08", "00"]:
        matches = [p for p in all_jsons if p.stem.endswith(f"_{nn}")]
        if matches:
            random.seed(42 + int(nn))
            diversified.append(random.choice(matches))

    # 추가로 랜덤 보충
    random.seed(42)
    remaining = [p for p in all_jsons if p not in diversified]
    if remaining:
        diversified += random.sample(remaining, min(max(0, args.n - len(diversified)), len(remaining)))
    samples = diversified[: args.n]

    # ---- 키 통계 ----
    eq_counter: Counter = Counter()
    an_counter: Counter = Counter()
    info_counter: Counter = Counter()

    for p in samples:
        try:
            with open(p, "r", encoding="utf-8") as f:
                o = json.load(f)
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
            continue

        eq = o.get("equipment") or {}
        an = o.get("annotations") or {}
        im = o.get("images") or {}
        info = o.get("info") or {}
        eq_counter.update(eq.keys())
        an_counter.update(an.keys())
        info_counter.update(info.keys())

        print(f"\n=== {p.name}  (facepart={im.get('facepart')}) ===")
        print(f"  EQUIPMENT  ({len(eq)} keys): {sorted(eq.keys())}")
        print(f"  ANNOTATIONS({len(an)} keys): {sorted(an.keys())}")
        suspect = [k for k in eq if any(
            t in k.lower() for t in ["pig", "mela", "wrink", "rough", "ra_", "_ra"]
        )]
        if suspect:
            print(f"  SUSPECT EQ : {suspect}")
            for k in suspect:
                print(f"      {k} = {eq[k]}")
        else:
            print(f"  SUSPECT EQ : (pigmentation/wrinkle/melanin/Ra/roughness 관련 키 없음)")

    print("\n========== EQUIPMENT 전체 등장 키 빈도 ==========")
    for k, c in eq_counter.most_common():
        print(f"  {c:3d}× {k}")

    print("\n========== ANNOTATIONS 전체 등장 키 빈도 ==========")
    for k, c in an_counter.most_common():
        print(f"  {c:3d}× {k}")


if __name__ == "__main__":
    main()
