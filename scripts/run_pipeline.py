"""
피부 분석 통합 실행 파이프라인
=================================
[파일 구조]
  preprocess_v3.py   → STEP 1: 원본 이미지 전처리 후 저장
  skin_analysis.py   → STEP 2: 전처리된 이미지 분석 (12개 항목)
  run_pipeline.py    → 본 파일: 두 파일을 연결하여 전체 파이프라인 실행

[폴더 구조]
  ./data/cropped_img/         ← 원본 부위별 크롭 이미지
      ├── forehead/           ← 부위명이 곧 라벨
      ├── nose/
      ├── left_cheek/
      ├── right_cheek/
      └── chin/
      
  ./data/final_processed_v3/  ← STEP 1 결과 (전처리 완료 이미지)
  ./data/analysis_results/    ← STEP 2 결과 (분석 점수 CSV + 시각화)

[실행 방법]
  # 전체 파이프라인 (전처리 + 분석)
  python run_pipeline.py

  # 전처리만
  python run_pipeline.py --step preprocess

  # 분석만 (이미 전처리된 이미지가 있을 때)
  python run_pipeline.py --step analyze

  # 분석 결과 시각화 저장 추가
  python run_pipeline.py --step analyze --save-vis
"""

import os
import sys
import csv
import argparse
import importlib.util
from pathlib import Path
import cv2
import numpy as np

# tqdm이 없는 환경에서도 동작하도록 fallback
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        """tqdm 미설치 환경용 fallback: 진행률을 텍스트로 출력"""
        def __init__(self, iterable, desc="", **kwargs):
            self._iter = list(iterable)
            self._desc = desc
            self._total = len(self._iter)
        def __iter__(self):
            for i, item in enumerate(self._iter, 1):
                print(f"\r  {self._desc} {i}/{self._total}", end="", flush=True)
                yield item
            print()
        @staticmethod
        def write(msg):
            print(f"\n{msg}")

# ===============================
# 0. 경로 설정 (여기만 수정하면 됨)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # AI/scripts
ROOT_DIR = os.path.dirname(BASE_DIR)                   # AI/

INPUT_DIR       = os.path.join(ROOT_DIR, "data", "cropped_img")         # 원본 이미지 폴더
PROCESSED_DIR   = os.path.join(ROOT_DIR, "data", "final_processed_v3")  # 전처리 결과 폴더
ANALYSIS_DIR    = os.path.join(ROOT_DIR, "data", "analysis_results")    # 분석 결과 폴더
VIS_DIR         = os.path.join(ROOT_DIR, "data", "analysis_vis")        # 시각화 저장 폴더

PREPROCESS_FILE = os.path.join(BASE_DIR, "preprocess_v3.py")            # 전처리 모듈 경로
ANALYSIS_FILE   = os.path.join(BASE_DIR, "skin_analysis.py")            # 분석 모듈 경로      

# ===============================
# 유틸: 동적 모듈 로드
# ===============================

def load_module(filepath: str, module_name: str):
    """경로로부터 파이썬 모듈 동적 로드"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===============================
# STEP 1: 전처리 실행
# ===============================

def run_preprocess():
    """
    preprocess_v3.py의 main()을 그대로 실행.
    INPUT_DIR → PROCESSED_DIR 로 전처리 후 저장.
    """
    print("\n" + "="*55)
    print("  STEP 1 / 2  |  이미지 전처리 (preprocess_v3.py)")
    print("="*55)

    if not os.path.exists(PREPROCESS_FILE):
        print(f"❌ 전처리 파일을 찾을 수 없습니다: {PREPROCESS_FILE}")
        sys.exit(1)

    preprocess = load_module(PREPROCESS_FILE, "preprocess_v3")

    preprocess.main()
    print(f"\n✅ 전처리 완료 → 저장 경로: {PROCESSED_DIR}")


# ===============================
# STEP 2: 분석 실행
# ===============================

def run_analysis(save_vis: bool = False):
    """
    skin_analysis.py의 SkinAnalyzer를 사용하여
    PROCESSED_DIR 내 모든 이미지를 분석 후 CSV로 저장.
    """
    print("\n" + "="*55)
    print("  STEP 2 / 2  |  피부 분석 (skin_analysis.py)")
    print("="*55)

    if not os.path.exists(ANALYSIS_FILE):
        print(f"❌ 분석 파일을 찾을 수 없습니다: {ANALYSIS_FILE}")
        sys.exit(1)

    if not os.path.exists(PROCESSED_DIR):
        print(f"❌ 전처리된 이미지 폴더가 없습니다: {PROCESSED_DIR}")
        print("   먼저 전처리를 실행하세요: python run_pipeline.py --step preprocess")
        sys.exit(1)

    analysis_mod = load_module(ANALYSIS_FILE, "skin_analysis")
    SkinAnalyzer = analysis_mod.SkinAnalyzer

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    if save_vis:
        os.makedirs(VIS_DIR, exist_ok=True)

    # CSV 컬럼 정의
    csv_columns = [
        "file", "equipment", "label",
        "oil", "wrinkle", "sebaceous", "pore",
        "spot", "pigmentation", "dark_circle", "radiance",
        "redness", "shine", "dullness", "elasticity",
        "skin_color_score", "skin_type",
    ]

    csv_path = os.path.join(ANALYSIS_DIR, "analysis_results.csv")

    # 분석 대상 이미지 수집
    image_paths = []

    EQUIPMENT_LIST = ["D", "P", "T"]

    for equipment in EQUIPMENT_LIST:
        equipment_path = os.path.join(PROCESSED_DIR, equipment)
        for root, dirs, files in os.walk(equipment_path):
            for file in files:
                if file.lower().endswith((".jpg", ".png")):
                    image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"⚠️  분석할 이미지가 없습니다: {equipment_path}")
        return

    print(f"\n  분석 대상: {len(image_paths)}장\n")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for img_path in tqdm(image_paths, desc="  분석 중"):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 라벨 = 부위 폴더명 (예: forehead, nose, ...)
            rel_file = os.path.relpath(img_path, PROCESSED_DIR)
            parts = Path(rel_file).parts
            # parts 예시: ("D", "right_cheek", "0011_01_R30_05.jpg")
            equipment = parts[0] if len(parts) >= 3 else ""
            label     = parts[1] if len(parts) >= 3 else parts[0]
            # → equipment = "D",  label = "right_cheek"

            try:
                analyzer = SkinAnalyzer(img)
                # 개별 분석 함수 직접 호출 (tqdm 출력 억제)
                results = {
                    "oil":          analysis_mod.analyze_oil(img),
                    "wrinkle":      analysis_mod.analyze_wrinkle(img),
                    "sebaceous":    analysis_mod.analyze_sebaceous(img),
                    "pore":         analysis_mod.analyze_pore(img),
                    "spot":         analysis_mod.analyze_spot(img),
                    "pigmentation": analysis_mod.analyze_pigmentation(img),
                    "dark_circle":  analysis_mod.analyze_dark_circle(img),
                    "radiance":     analysis_mod.analyze_radiance(img),
                    "redness":      analysis_mod.analyze_redness(img),
                    "shine":        analysis_mod.analyze_shine(img),
                    "dullness":     analysis_mod.analyze_dullness(img),
                    "skin_color":   analysis_mod.analyze_skin_color(img),
                }
                elasticity = analysis_mod.calc_elasticity(
                    results["wrinkle"].score, results["dullness"].score
                )

                row = {
                    "file":              rel_file,
                    "equipment":         equipment,
                    "label":             label,
                    "oil":               round(results["oil"].score, 2),
                    "wrinkle":           round(results["wrinkle"].score, 2),
                    "sebaceous":         round(results["sebaceous"].score, 2),
                    "pore":              round(results["pore"].score, 2),
                    "spot":              round(results["spot"].score, 2),
                    "pigmentation":      round(results["pigmentation"].score, 2),
                    "dark_circle":       round(results["dark_circle"].score, 2),
                    "radiance":          round(results["radiance"].score, 2),
                    "redness":           round(results["redness"].score, 2),
                    "shine":             round(results["shine"].score, 2),
                    "dullness":          round(results["dullness"].score, 2),
                    "elasticity":        round(elasticity, 2),
                    "skin_color_score":  round(results["skin_color"].score, 2),
                    "skin_type":         results["skin_color"].detail.get("skin_type", ""),
                }
                writer.writerow(row)

                # 시각화 저장 (옵션)
                if save_vis:
                    analyzer_obj = SkinAnalyzer(img)
                    vis = analyzer_obj.visualize(results)
                    vis_filename = os.path.splitext(os.path.basename(img_path))[0] + "_vis.jpg"
                    vis_subdir = os.path.join(VIS_DIR, label)
                    os.makedirs(vis_subdir, exist_ok=True)
                    cv2.imwrite(os.path.join(vis_subdir, vis_filename), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

            except Exception as e:
                tqdm.write(f"  ⚠️  오류 ({rel_file}): {e}")
                continue

    print(f"\n✅ 분석 완료 → CSV 저장: {csv_path}")
    if save_vis:
        print(f"   시각화 저장: {VIS_DIR}")

    # 간단한 요약 출력
    _print_summary(csv_path)


# ===============================
# 결과 요약 출력
# ===============================

def _print_summary(csv_path: str):
    """CSV를 읽어 라벨(부위)별 평균 점수 요약 출력"""
    if not os.path.exists(csv_path):
        return

    from collections import defaultdict

    score_cols = [
        "oil", "wrinkle", "sebaceous", "pore", "spot",
        "pigmentation", "dark_circle", "radiance",
        "redness", "shine", "dullness", "elasticity",
    ]
    label_scores = defaultdict(lambda: defaultdict(list))

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            for col in score_cols:
                try:
                    label_scores[label][col].append(float(row[col]))
                except (ValueError, KeyError):
                    pass

    print("\n" + "="*55)
    print("  📊  부위별 평균 분석 점수 요약 (0~100)")
    print("="*55)

    col_names = {
        "oil": "유분", "wrinkle": "주름", "sebaceous": "피지",
        "pore": "모공", "spot": "기미잡티", "pigmentation": "색소침착",
        "dark_circle": "다크서클", "radiance": "광채",
        "redness": "홍조", "shine": "번들거림",
        "dullness": "칙칙함", "elasticity": "탄력도",
    }

    for label, scores in sorted(label_scores.items()):
        n = len(list(scores.values())[0]) if scores else 0
        print(f"\n  [{label}]  ({n}장)")
        for col in score_cols:
            if col not in scores or not scores[col]:
                continue
            avg = sum(scores[col]) / len(scores[col])
            bar = "█" * int(avg / 5) + "░" * (20 - int(avg / 5))
            print(f"    {col_names[col]:<8} {bar} {avg:5.1f}")

    print("="*55 + "\n")


# ===============================
# 진입점
# ===============================

def main():
    parser = argparse.ArgumentParser(
        description="피부 분석 통합 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--step",
        choices=["all", "preprocess", "analyze"],
        default="all",
        help=(
            "실행할 단계 선택:\n"
            "  all        : 전처리 + 분석 순서대로 실행 (기본값)\n"
            "  preprocess : 전처리만 실행\n"
            "  analyze    : 분석만 실행 (전처리 완료 가정)\n"
        )
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="분석 결과 오버레이 시각화 이미지를 저장합니다."
    )
    args = parser.parse_args()

    print("\n🚀 피부 분석 파이프라인 시작")
    print(f"   전처리 모듈 : {PREPROCESS_FILE}")
    print(f"   분석 모듈   : {ANALYSIS_FILE}")
    print(f"   입력 경로   : {INPUT_DIR}")
    print(f"   전처리 출력 : {PROCESSED_DIR}")
    print(f"   분석 결과   : {ANALYSIS_DIR}")

    if args.step in ("all", "preprocess"):
        run_preprocess()

    if args.step in ("all", "analyze"):
        run_analysis(save_vis=args.save_vis)

    print("🎉 파이프라인 완료\n")


if __name__ == "__main__":
    main()