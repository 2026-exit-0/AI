"""AI-Hub 원본 이미지 + JSON 라벨을 단일 manifest.csv로 정규화.

AI-Hub 표준 계층 구조 (Training/Validation 동일):
  <image_root>/                       (예: Training/01.원천데이터/TS)
    1. 디지털카메라/
      0002/
        0002_01_F.jpg
        0002_01_L15.jpg
        ...
      0003/
      ...
    2. 스마트패드/
    3. 스마트폰/

  <json_root>/                        (예: Training/02.라벨링데이터/TL)
    1. 디지털카메라/
      0002/
        0002_01_F_00.json   ← facepart 0 (전체)
        0002_01_F_01.json   ← facepart 1 (이마)
        ...
        0002_01_F_08.json   ← facepart 8 (턱)
    ...

이미지 1장은 9개 facepart JSON과 매칭되어 manifest 9행 생성.
각 행은 (이미지 경로, bbox, 부위, 라벨)을 포함, 학습 시점에 bbox로 crop.

사용 예:
  python -m src.build_manifest ^
      --image-root "C:\\damda\\dataset\\028...\\Training\\01.원천데이터\\TS" ^
      --json-root  "C:\\damda\\dataset\\028...\\Training\\02.라벨링데이터\\TL" ^
      --output     data\\manifest.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .utils import (
    REGION_TO_ID,
    REGION_TO_JSON_PREFIX,
    setup_logger,
)

# 폴더명 → device 문자 매핑
# AI-Hub 폴더는 "1. 디지털카메라", "2. 스마트패드", "3. 스마트폰" 형태
def folder_to_device_char(folder_name: str) -> Optional[str]:
    name = folder_name.strip()
    if "디지털카메라" in name or name.startswith("1"):
        return "D"
    if "스마트패드" in name or name.startswith("2"):
        return "T"
    if "스마트폰" in name or name.startswith("3"):
        return "P"
    return None


# facepart 번호 → 부위명 (REGION_TO_ID와 정합)
FACEPART_NUM_TO_REGION = {
    0: "PART_0",
    1: "FOREHEAD",
    2: "GLABELLA",
    3: "L_EYE",
    4: "R_EYE",
    5: "L_CHEEK",   # JSON 실측 확인
    6: "R_CHEEK",
    7: "LIP",
    8: "CHIN",
}

OUTPUT_COLUMNS = [
    "image_path", "region", "region_id",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "subject_id", "device", "angle",
    "gender", "age",
    "skin_type", "sensitive",
    # 회귀
    "moisture", "elasticity_mean", "pore_value",
    "pigmentation_value", "wrinkle_value",
    # 분류 (전문가 등급)
    "wrinkle_grade", "pigmentation_grade", "pore_grade",
    "dryness_grade", "sagging_grade",
]


def _try_keys(d: dict, *keys: str) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _gather_elasticity_mean(equipment: dict, prefix: str) -> Optional[float]:
    """탄력 R0~R9 평균."""
    vals = []
    for i in range(10):
        key = f"{prefix}_elasticity_R{i}" if prefix else f"elasticity_R{i}"
        if key in equipment and equipment[key] is not None:
            vals.append(float(equipment[key]))
    return sum(vals) / len(vals) if vals else None


def _parse_bbox(bbox) -> tuple:
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(bbox)
    if isinstance(bbox, dict):
        x = bbox.get("x", 0); y = bbox.get("y", 0)
        w = bbox.get("w", 0); h = bbox.get("h", 0)
        return (x, y, x + w, y + h)
    return (None, None, None, None)


def process_one_image(
    img_path: Path,
    image_root: Path,
    json_root: Path,
):
    """이미지 1장 → 최대 9개 manifest 행 반환 (facepart별).

    image_root 기준 상대경로에서 device 폴더와 subject ID 추출:
      image_root/{device_folder}/{subject_id}/{filename}.jpg
    """
    try:
        rel = img_path.relative_to(image_root)
    except ValueError:
        return []
    parts = rel.parts  # (device_folder, subject_id, filename)
    if len(parts) < 3:
        return []

    device_folder = parts[0]
    subject_id    = parts[1]
    filename      = parts[-1]
    stem          = Path(filename).stem  # "0002_01_F"

    device_char = folder_to_device_char(device_folder)
    if device_char is None:
        return []

    # 파일명에서 angle 추출 — "{ID}_{sub}_{angle}" 형식
    name_parts = stem.split("_")
    if len(name_parts) < 3:
        return []
    angle = "_".join(name_parts[2:])

    # JSON 경로: json_root/{device_folder}/{subject_id}/{stem}_{NN}.json
    json_dir = json_root / device_folder / subject_id
    if not json_dir.exists():
        return []

    rows = []
    for facepart_num in range(9):
        json_path = json_dir / f"{stem}_{facepart_num:02d}.json"
        if not json_path.exists():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        info        = obj.get("info", {})
        images_meta = obj.get("images", {})
        annotations = obj.get("annotations", {}) or {}
        equipment   = obj.get("equipment", {}) or {}

        fp_in_json = images_meta.get("facepart", facepart_num)
        region = FACEPART_NUM_TO_REGION.get(int(fp_in_json))
        if region is None:
            continue
        prefix = REGION_TO_JSON_PREFIX.get(region, "")

        bx1, by1, bx2, by2 = _parse_bbox(images_meta.get("bbox"))

        def k(name: str) -> str:
            return f"{prefix}_{name}" if prefix else name

        moisture           = _try_keys(equipment, k("moisture"))
        elasticity_mean    = _gather_elasticity_mean(equipment, prefix)
        pore_value         = _try_keys(equipment, k("pore"))

        # v3: AI-Hub 데이터는 부위별로 다른 측정 항목을 가짐.
        # pigmentation 측정값은 PART_0(전체 얼굴)의 'pigmentation_count' 만 존재.
        # 다른 부위(이마/볼)의 pigmentation 은 annotations 의 전문가 등급만 있음.
        if region == "PART_0":
            pigmentation_value = _try_keys(equipment, "pigmentation_count")
        else:
            pigmentation_value = None

        # wrinkle 측정값은 L_EYE/R_EYE 의 8개 거칠기 파라미터(Ra/Rmax/Rt/...) 형태로 존재.
        # 단일 대표값으로 Ra(평균 거칠기) 사용. 다른 부위(이마/미간 등)는 annotations 등급만.
        if region in ("L_EYE", "R_EYE"):
            wrinkle_value = _try_keys(equipment, k("wrinkle_Ra"))
        else:
            wrinkle_value = None

        wrinkle_grade      = _try_keys(annotations, k("wrinkle"))
        pigmentation_grade = _try_keys(annotations, k("pigmentation"))
        pore_grade         = _try_keys(annotations, k("pore"))
        dryness_grade      = _try_keys(annotations, k("dryness"))
        sagging_grade      = _try_keys(annotations, k("sagging"))

        rows.append({
            "image_path": str(img_path),
            "region": region,
            "region_id": REGION_TO_ID[region],
            "bbox_x1": bx1, "bbox_y1": by1, "bbox_x2": bx2, "bbox_y2": by2,
            "subject_id": str(info.get("id", subject_id)),
            "device": device_char,
            "angle": angle,
            "gender": info.get("gender"),
            "age": info.get("age"),
            "skin_type": info.get("skin_type"),
            "sensitive": info.get("sensitive"),
            "moisture": moisture,
            "elasticity_mean": elasticity_mean,
            "pore_value": pore_value,
            "pigmentation_value": pigmentation_value,
            "wrinkle_value": wrinkle_value,
            "wrinkle_grade": wrinkle_grade,
            "pigmentation_grade": pigmentation_grade,
            "pore_grade": pore_grade,
            "dryness_grade": dryness_grade,
            "sagging_grade": sagging_grade,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", type=str, required=True,
                    help="이미지 루트 (예: Training/01.원천데이터/TS). "
                         "하위에 device 폴더(1. 디지털카메라 등)가 있어야 함")
    ap.add_argument("--json-root", type=str, required=True,
                    help="JSON 라벨 루트 (예: Training/02.라벨링데이터/TL)")
    ap.add_argument("--output", type=str, default="data/manifest.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="개발용 이미지 수 제한 (0=전체)")
    args = ap.parse_args()

    logger = setup_logger("build_manifest")
    image_root = Path(args.image_root)
    json_root  = Path(args.json_root)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 모든 device 폴더에서 재귀 이미지 검색
    logger.info(f"이미지 검색 중: {image_root}")
    images = sorted(image_root.rglob("*.jpg")) + sorted(image_root.rglob("*.jpeg"))
    if args.limit:
        images = images[: args.limit]
    logger.info(f"이미지 {len(images)}개 발견. 매니페스트 생성 시작")

    rows = []
    miss_image = 0
    for img_path in tqdm(images):
        new_rows = process_one_image(img_path, image_root, json_root)
        if not new_rows:
            miss_image += 1
            continue
        rows.extend(new_rows)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(out_path, index=False)
    logger.info(
        f"manifest 저장: {out_path}\n"
        f"  - 처리된 이미지: {len(images) - miss_image} / {len(images)}\n"
        f"  - 매니페스트 행 수: {len(df)}  "
        f"(이미지당 평균 {len(df) / max(1, len(images) - miss_image):.1f} 부위)"
    )

    if len(df) == 0:
        logger.warning("manifest가 비었습니다. 경로/매칭 로직을 점검하세요.")
        return

    logger.info(f"\n=== 부위별 분포 ===\n{df['region'].value_counts().to_string()}")
    logger.info(f"\n=== 디바이스별 분포 ===\n{df['device'].value_counts().to_string()}")
    logger.info(
        f"\n=== 라벨 결측률(%) ===\n"
        + (df.isna().mean() * 100).round(1).to_string()
    )


if __name__ == "__main__":
    main()
