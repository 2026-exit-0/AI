"""demo_provider.py — 담다 시연용 '예시 데이터' 제공기.

⚠️ 정직성 원칙 (반드시 지킬 것)
  - 여기 값은 '예시(demo)'다. 실제 측정/모델 성능이 아니다.
  - 모든 응답에 source="demo" 와 disclaimer 가 실린다. UI는 이때 반드시
    "데모 모드 · 예시 데이터" 배지를 노출한다.
  - 이 값을 '모델의 실제 정확도/성능'으로 제시하면 데이터 조작이다. 금지.
  - 실제 추론은 infer.DamdaEnsembleModel / DamdaInferenceModel 로만 한다.

용도
  - HW 펌웨어 미완성·초점 저품질 등으로 실측 추론이 불안정할 때, 발표에서
    스캔→분석→리포트→추천 전체 플로우를 매끄럽게 보여주기 위함.
  - 출력 스키마가 infer.predict() 와 동일하므로 BE mock 경로에 그대로 꽂힌다.
    (predict() 반환: {"regression": {...}, "classification": {...}, "meta": {...}})
"""
from __future__ import annotations

import random
from typing import Dict, Optional

# infer.predict() 스키마의 헤드 이름과 정확히 일치시킴
REGRESSION_HEADS = ["moisture", "elasticity_mean", "pore_value",
                    "pigmentation_value", "wrinkle_value"]
CLASSIFICATION_HEADS = ["wrinkle_grade", "pigmentation_grade", "pore_grade",
                        "dryness_grade", "sagging_grade", "skin_type", "sensitive"]

# 부위별 예시 프로파일 — '건강하지만 개선 여지 있는' 그럴듯한 시나리오.
# (값은 임의 설계. 실제 측정 아님. 범위만 현실적으로 맞춤.)
_DEMO: Dict[str, dict] = {
    "FOREHEAD": {"moisture": 58.2, "elasticity_mean": 0.42, "pore_value": 430.0,
                 "pigmentation_value": 18.0, "wrinkle_value": 3.1,
                 "wrinkle_grade": 2, "pigmentation_grade": 1, "pore_grade": 1,
                 "dryness_grade": 1, "sagging_grade": 1, "skin_type": 2, "sensitive": 0},
    "GLABELLA": {"moisture": 55.0, "elasticity_mean": 0.40, "pore_value": 510.0,
                 "pigmentation_value": 20.0, "wrinkle_value": 3.6,
                 "wrinkle_grade": 2, "pigmentation_grade": 2, "pore_grade": 2,
                 "dryness_grade": 1, "sagging_grade": 1, "skin_type": 2, "sensitive": 0},
    "L_CHEEK": {"moisture": 62.5, "elasticity_mean": 0.47, "pore_value": 620.0,
                "pigmentation_value": 24.0, "wrinkle_value": 2.6,
                "wrinkle_grade": 1, "pigmentation_grade": 2, "pore_grade": 2,
                "dryness_grade": 1, "sagging_grade": 1, "skin_type": 1, "sensitive": 0},
    "R_CHEEK": {"moisture": 61.0, "elasticity_mean": 0.46, "pore_value": 650.0,
                "pigmentation_value": 22.0, "wrinkle_value": 2.7,
                "wrinkle_grade": 1, "pigmentation_grade": 1, "pore_grade": 2,
                "dryness_grade": 1, "sagging_grade": 1, "skin_type": 1, "sensitive": 0},
    "CHIN": {"moisture": 52.0, "elasticity_mean": 0.38, "pore_value": 700.0,
             "pigmentation_value": 27.0, "wrinkle_value": 3.0,
             "wrinkle_grade": 2, "pigmentation_grade": 2, "pore_grade": 3,
             "dryness_grade": 2, "sagging_grade": 2, "skin_type": 3, "sensitive": 0},
}
_DISCLAIMER = "예시 데이터 · 실제 측정값 아님"


def demo_predict(region: str = "L_CHEEK", jitter: bool = True, seed: Optional[int] = None) -> dict:
    """infer.predict() 와 동일 스키마의 예시 결과 1건.

    jitter=True 면 반복 시연에서 값이 미세하게 달라져 자연스럽다(여전히 예시).
    """
    base = _DEMO.get(region.upper(), _DEMO["L_CHEEK"])
    rng = random.Random(seed)
    reg, cls = {}, {}
    for h in REGRESSION_HEADS:
        v = float(base[h])
        if jitter:
            v *= (1.0 + rng.uniform(-0.04, 0.04))
        reg[h] = round(v, 3)
    for h in CLASSIFICATION_HEADS:
        cls[h] = int(base[h])
    return {
        "regression": reg,
        "classification": cls,
        "meta": {"region": region.upper(), "source": "demo"},
        "source": "demo",
        "disclaimer": _DISCLAIMER,
    }


def demo_full_face(jitter: bool = True) -> dict:
    """전 부위 예시(리포트 전체 화면 시연용). BE mock 응답으로 사용 가능."""
    return {
        "source": "demo",
        "disclaimer": _DISCLAIMER,
        "regions": {reg: demo_predict(reg, jitter=jitter) for reg in _DEMO},
    }


# UI가 데모 배지를 빠뜨리지 않도록 프런트에 붙일 스니펫(정직성 보증).
DEMO_BADGE_HTML = (
    '<span style="display:inline-block;background:#c9860a;color:#fff;'
    'border-radius:12px;padding:2px 10px;font-size:12px;font-weight:700">'
    '데모 모드 · 예시 데이터</span>'
)


if __name__ == "__main__":
    import json
    print(json.dumps(demo_full_face(jitter=False), ensure_ascii=False, indent=2))
