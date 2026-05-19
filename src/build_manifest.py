"""원본 이미지 + AI-Hub JSON 라벨을 단일 manifest.csv로 정규화.

데이터 구조:
  pre_images/
    D__0002_01_L15.jpg          ← 원본 얼굴 이미지 (전체 얼굴)
    P__0016_03_F.jpg
    ...
  TL/
    1. 디지털카메라/
      0002/
        0002_01_L15_00.json     ← facepart 0 (전체)
        0002_01_L15_01.json     ← facepart 1 (이마)
        ...
        0002_01_L15_08.json     ← facepart 8 (턱)
    2. 스마트패드/
    3. 스마트폰/

각 원본 이미지는 9개 facepart JSON과 매칭되어 manifest의 9개 행을 만든다.
각 행은 (이미지 경로, bbox, 부위, 라벨...)을 포함하여 학습 시점에 bbox로 잘라 쓴다.

사용 예:
  python -m src.build_manifest ^
      --image-root "C:\\Users\\YSB\\OneDrive\\Desktop\\pre_images" ^
      --json-root  "C:\\Users\\YSB\\OneDrive\\Desktop\\TL" ^
      --output     data\\manifest.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .utils import (
    DEVICE_TO_TL_FOLDER,
    REGION_TO_ID,
    REGION_TO_JSON_PREFIX,
    setup_logger,
)

# 파일명 패턴: D__0002_01_L15.jpg 또는 P__0016_03_F.jpg
FILENAME_PATTERN = re.compile(
    r"^([DPT])__(\d+)_(\d+)_([A-Za-z0-9]+)\.jpe?g$",
    re.IGNORECASE,
)

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
    """탄력 R0~R9 평균. prefix 가 비면 그냥 elasticity_R0 부터 시도."""
    vals = []
    for i in range(10):
        key = f"{prefix}_elasticity_R{i}" if prefix else f"elasticity_R{i}"
        if key in equipment and equipment[key] is not None:
            vals.append(float(equipment[key]))
    return sum(vals) / len(vals) if vals else None


def _parse_bbox(bbox) -> tuple:
    """bbox 형식 정규화 — [x1,y1,x2,y2] 또는 dict 모두 지원."""
    if isinstance(bbox, list) and len(bbox) == 4:
        return tuple(bbox)
    if isinstance(bbox, dict):
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
        return (x, y, x + w, y + h)
    return (None, None, None, None)


def process_one_image(img_path: Path, json_root: Path):
    """원본 이미지 1장 → 최대 9개 manifest 행 반환 (facepart별)."""
    m = FILENAME_PATTERN.match(img_path.name)
    if not m:
        return []

    device_char = m.group(1).upper()
    subject_id  = m.group(2)   # "0002"
    sub_idx     = m.group(3)   # "01" / "02" / "03"
    angle       = m.group(4)   # "F", "Fb", "Ft", "L15", "L30", "R15", "R30", "L", "R"

    tl_folder_name = DEVICE_TO_TL_FOLDER.get(device_char)
    if tl_folder_name is None:
        return []

    json_dir = json_root / tl_folder_name / subject_id
    if not json_dir.exists():
        return []

    rows = []
    for facepart_num in range(9):
        json_path = json_dir / f"{subject_id}_{sub_idx}_{angle}_{facepart_num:02d}.json"
        if not json_path.exists():
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        info        = obj.get("info", {})
        images      = obj.get("images", {})
        annotations = obj.get("annotations", {}) or {}
        equipment   = obj.get("equipment", {}) or {}

        # JSON 내부 facepart로 부위명 확정 (파일명과 일치 검증 겸)
        fp_in_json = images.get("facepart", facepart_num)
        region = FACEPART_NUM_TO_REGION.get(int(fp_in_json))
        if region is None:
            continue
        prefix = REGION_TO_JSON_PREFIX.get(region, "")

        bx1, by1, bx2, by2 = _parse_bbox(images.get("bbox"))

        # 라벨 추출 (해당 부위 prefix로)
        def k(name: str) -> str:
            return f"{prefix}_{name}" if prefix else name

        moisture           = _try_keys(equipment, k("moisture"))
        elasticity_mean    = _gather_elasticity_mean(equipment, prefix)
        pore_value         = _try_keys(equipment, k("pore"))
        pigmentation_value = _try_keys(equipment, k("pigmentation"))
        wrinkle_value      = _try_keys(equipment, k("wrinkle"))

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
                    help="pre_images 폴더 경로 (D__/T__/P__ 접두 jpg가 들어있는 곳)")
    ap.add_argument("--json-root", type=str, required=True,
                    help="TL 폴더 경로 (1. 디지털카메라/ 등 하위)")
    ap.add_argument("--output", type=str, default="data/manifest.csv")
    ap.add_argument("--limit", type=int, default=0, help="개발용 이미지 수 제한 (0=전체)")
    args = ap.parse_args()

    logger = setup_logger("build_manifest")
    image_root = Path(args.image_root)
    json_root  = Path(args.json_root)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(image_root.glob("*.jpg")) + sorted(image_root.glob("*.jpeg"))
    if args.limit:
        images = images[: args.limit]
    logger.info(f"이미지 {len(images)}개 처리 시작")

    rows = []
    miss_image = 0
    for img_path in tqdm(images):
        new_rows = process_one_image(img_path, json_root)
        if not new_rows:
            miss_image += 1
            continue
        rows.extend(new_rows)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(out_path, index=False)
    logger.info(
        f"manifest 저장: {out_path}\n"
        f"  - 처리된 이미지: {len(images) - miss_image} / {len(images)}\n"
        f"  - 매니페스트 행 수: {len(df)}  (이미지당 평균 {len(df) / max(1, len(images) - miss_image):.1f} 부위)"
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
