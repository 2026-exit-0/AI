"""유틸리티: 시드 고정, 디바이스 자동 감지, 로거.

노트북 환경 호환:
  - NVIDIA CUDA → 'cuda'
  - Apple Silicon (M1/M2/M3) → 'mps'
  - 그 외 → 'cpu'
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """재현 가능한 실험을 위한 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 완전 deterministic은 성능 손실이 큼 → 학습 단계에서는 benchmark 활용 권장
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """CUDA > MPS > CPU 우선순위로 자동 감지."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_supports_amp(device: torch.device) -> bool:
    """Mixed Precision (AMP) 사용 가능 여부 — 현재는 CUDA에서만 안정적."""
    return device.type == "cuda"


def setup_logger(name: str = "damda", log_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ===== 9개 부위 ↔ facepart 번호 ↔ JSON prefix =====
# 매핑은 실제 데이터(JSON 내부) 기준. JSON에서 facepart=N 인 파일이 어떤 prefix
# (예: l_cheek_*, forehead_*) 를 쓰는지 확인하여 정함.
REGION_TO_ID = {
    "PART_0":   0,   # 전체 (라벨 없음, 이미지만)
    "FOREHEAD": 1,
    "GLABELLA": 2,
    "L_EYE":    3,
    "R_EYE":    4,
    "L_CHEEK":  5,   # JSON 실측: facepart=5 → l_cheek_*
    "R_CHEEK":  6,
    "LIP":      7,
    "CHIN":     8,
}
ID_TO_REGION = {v: k for k, v in REGION_TO_ID.items()}

# 부위명 → AI-Hub JSON 라벨 키 prefix
# 실제 prefix가 다르면 build_manifest 실행 후 결측률 보고 보정.
REGION_TO_JSON_PREFIX = {
    "PART_0":   "",
    "FOREHEAD": "forehead",      # 실측 확인됨
    "GLABELLA": "glabellus",     # 실측 확인됨 (Latin form, 주의)
    "L_EYE":    "l_perocular",   # 실측 확인됨
    "R_EYE":    "r_perocular",   # 실측 확인됨
    "L_CHEEK":  "l_cheek",       # 실측 확인됨
    "R_CHEEK":  "r_cheek",
    "LIP":      "lip",
    "CHIN":     "chin",
}

# ===== 디바이스 prefix ↔ TL 폴더 매핑 =====
# 이미지 파일명: {DEVICE}__{ID}_{SUB}_{ANGLE}.jpg
#   D = 디지털카메라, T = 스마트패드, P = 스마트폰
DEVICE_TO_TL_FOLDER = {
    "D": "1. 디지털카메라",
    "T": "2. 스마트패드",
    "P": "3. 스마트폰",
}
DEVICE_TO_ID = {"D": 0, "T": 1, "P": 2}
