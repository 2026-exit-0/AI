"""원본 AI-Hub JSON 1~N 개를 까서 equipment/annotations 의 실제 키 목록을 출력.

사용 예:
    # 기본: TL 루트 자동 탐색, 3개 부위(forehead, l_cheek, chin) 샘플링
    python -m src.inspect_json --json-root "C:\\damda\\dataset\\..\\Training\\02.라벨링데이터\\TL"

    # 더 많이 보고 싶으면
    python -m src.inspect_json --json-root <PATH> --n 10
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-root", required=True, help="TL 라벨 루트")
    ap.add_argument("--n", type=int, default=5, help="샘플링할 JSON 개수")
    ap.add_argument("--prefix-filter", default="",
                    help="특정 부위만 보고 싶으면 (예: forehead, l_cheek)")
    args = ap.parse_args()

    root = Path(args.json_root)
    print(f"검색 중: {root}")
    all_jsons = list(root.rglob("*.json"))
    print(f"총 JSON 파일: {len(all_jsons)}")

    if args.prefix_filter:
        # facepart 번호로 필터링 어렵고 키 prefix 로 판단해야 하니 그냥 랜덤
        pass

    random.seed(42)
    samples = random.sample(all_jsons, min(args.n, len(all_jsons)))

    # 전체 통계 수집용 (모든 N개를 합쳐 key 등장 빈도)
    eq_keys_counter = Counter()
    an_keys_counter = Counter()

    for p in samples:
        try:
            with open(p, "r", encoding="utf-8") as f:
                o = json.load(f)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            continue

        eq = o.get("equipment") or {}
        an = o.get("annotations") or {}
        eq_keys = sorted(eq.keys())
        an_keys = sorted(an.keys())
        eq_keys_counter.update(eq_keys)
        an_keys_counter.update(an_keys)

        print(f"\n=== {p.name} ===")
        # facepart 정보
        im = o.get("images") or {}
        print(f"  facepart   : {im.get('facepart')}")
        print(f"  EQUIPMENT  : {eq_keys}")
        print(f"  ANNOTATIONS: {an_keys}")
        # pigmentation/wrinkle 관련 키만 따로 빨강
        suspect_eq = [k for k in eq_keys if "pig" in k.lower() or "mela" in k.lower()
                       or "wrink" in k.lower() or "ra" == k.lower() or "rough" in k.lower()]
        if suspect_eq:
            print(f"  SUSPECT EQ : {suspect_eq}")
            for k in suspect_eq:
                print(f"      {k} = {eq[k]}")

    print("\n========== 전체 등장 키 빈도 (equipment) ==========")
    for k, c in eq_keys_counter.most_common():
        print(f"  {c:3d}× {k}")

    print("\n========== 전체 등장 키 빈도 (annotations) ==========")
    for k, c in an_keys_counter.most_common():
        print(f"  {c:3d}× {k}")


if __name__ == "__main__":
    main()
