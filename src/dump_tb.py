"""TensorBoard 이벤트 파일에서 per-head 손실을 추출해 콘솔/CSV로 덤프.

train.log 에는 `val total` 만 찍히는데, per-head 손실은 SummaryWriter 로만 기록되어
TB 이벤트 파일에 들어있다. 이 스크립트로 v1/v2 비교 가능한 표를 만든다.

사용 예 (졸프실 PC cmd):
    python -m src.dump_tb runs\main
    python -m src.dump_tb runs\main_v1 runs\main
    python -m src.dump_tb runs\main --csv runs\main\val_per_head.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("tensorboard 패키지가 필요합니다: pip install tensorboard")
    sys.exit(1)


def load_run(run_dir: Path, tag_prefix: str = "val/") -> Dict[str, Dict[int, float]]:
    """run 디렉토리에서 tag_prefix 로 시작하는 모든 스칼라 시리즈를 dict 로 반환.

    반환 형식: {tag: {step(epoch): value}}
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"run 디렉토리 없음: {run_dir}")
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out: Dict[str, Dict[int, float]] = {}
    for tag in ea.Tags().get("scalars", []):
        if not tag.startswith(tag_prefix):
            continue
        series = {ev.step: ev.value for ev in ea.Scalars(tag)}
        out[tag] = series
    return out


def best_epoch(series: Dict[int, float], minimize: bool = True) -> Optional[int]:
    if not series:
        return None
    return min(series, key=lambda e: series[e]) if minimize else max(series, key=lambda e: series[e])


def print_run_summary(name: str, data: Dict[str, Dict[int, float]]):
    print(f"\n========== {name} ==========")
    if not data:
        print("  (val/* 태그 없음)")
        return
    total = data.get("val/loss/total", {})
    last_epoch = max(total) if total else None
    best_e = best_epoch(total) if total else None
    print(f"  epochs   : {sorted(total)[0]} ~ {last_epoch}  (총 {len(total)} 평가)")
    if best_e is not None:
        print(f"  best val/loss/total = {total[best_e]:.4f} @ epoch {best_e}")
    print()
    # tag 별로 final(last_epoch) / best(min) 표
    print(f"  {'tag':38s} {'final':>10s} {'best':>10s} {'best@ep':>9s}")
    print(f"  {'-'*38} {'-'*10} {'-'*10} {'-'*9}")
    for tag in sorted(data):
        s = data[tag]
        if not s:
            continue
        finv = s.get(last_epoch, float("nan")) if last_epoch is not None else float("nan")
        be = best_epoch(s)
        bv = s[be] if be is not None else float("nan")
        print(f"  {tag:38s} {finv:>10.4f} {bv:>10.4f} {be if be is not None else '-':>9}")


def print_comparison(name_a: str, a: Dict[str, Dict[int, float]],
                     name_b: str, b: Dict[str, Dict[int, float]]):
    """두 run의 best 값을 옆에 놓고 비교 (b - a, 음수가 개선)."""
    print(f"\n========== {name_a}  vs  {name_b} (best 기준) ==========")
    tags = sorted(set(a) | set(b))
    print(f"  {'tag':38s} {name_a:>12s} {name_b:>12s} {'delta':>10s}")
    print(f"  {'-'*38} {'-'*12} {'-'*12} {'-'*10}")
    for tag in tags:
        sa, sb = a.get(tag, {}), b.get(tag, {})
        if not sa or not sb:
            continue
        be_a = best_epoch(sa); be_b = best_epoch(sb)
        va = sa[be_a] if be_a is not None else float("nan")
        vb = sb[be_b] if be_b is not None else float("nan")
        delta = vb - va
        arrow = "↓" if delta < -1e-4 else ("↑" if delta > 1e-4 else "·")
        print(f"  {tag:38s} {va:>12.4f} {vb:>12.4f} {delta:>+9.4f} {arrow}")


def write_csv(run_name: str, data: Dict[str, Dict[int, float]], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_epochs = sorted({e for s in data.values() for e in s})
    tags = sorted(data)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(tags) + "\n")
        for e in all_epochs:
            row = [str(e)]
            for t in tags:
                v = data[t].get(e)
                row.append(f"{v:.6f}" if v is not None else "")
            f.write(",".join(row) + "\n")
    print(f"\nCSV 저장: {csv_path}  ({len(all_epochs)} epoch × {len(tags)} tag)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run 디렉토리 1~2개 (예: runs\\main_v1 runs\\main)")
    ap.add_argument("--prefix", default="val/", help="추출할 태그 접두어 (기본 val/)")
    ap.add_argument("--csv", default="", help="첫 번째 run을 CSV로 저장할 경로")
    args = ap.parse_args()

    loaded: List = []
    for r in args.runs:
        p = Path(r)
        data = load_run(p, tag_prefix=args.prefix)
        loaded.append((p.name, data))
        print_run_summary(p.name, data)

    if len(loaded) >= 2:
        print_comparison(loaded[0][0], loaded[0][1], loaded[1][0], loaded[1][1])

    if args.csv:
        write_csv(loaded[0][0], loaded[0][1], Path(args.csv))


if __name__ == "__main__":
    main()
