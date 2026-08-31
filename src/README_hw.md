# HW 매크로 스캔 경로 (surface_features / demo_provider)

`scan_images`(ESP32-CAM 실측) 관련 두 모듈. `PROGRESS.md` §6 "시연 후 작업"의
ESP32 도메인 대응 라인에 해당한다.

## 배경
실측 스캔은 **접사(macro)** 라 AI-Hub 얼굴부위 학습(=v3+v5.1 앙상블, `infer.py`)과
화각이 근본적으로 다르다. `scanner_aug`(v5/v5.1)는 해상도·압축·색만 흉내낼 뿐 화각을
못 메우므로, 기존 앙상블이 매크로 이미지에 그대로 전이되지 않는다.
→ 매크로에서 실제 추출 가능한 표면 지표를 별도 경로로 뽑는다.

## surface_features.py — 실측 표면 지표
```bash
pip install opencv-python-headless numpy
python -m src.surface_features --root <scan_images> --out surface_scores.csv
```
- WHITE+UV 페어(부위별 최선명 프레임) → 색소/불균일/질감/홍조/모공/포르피린 지수.
- `utils.REGION_TO_ID` 재사용, 스캔의 `NOSE` 폴더는 `GLABELLA`로 정규화.
- 라벨 없음 → `fit_cohort_thresholds` + `to_band` 로 코호트 상대 3-band(양호/보통/주의).
- 지표별 신뢰도 자동 판정(초점/과노출). 실측 표본(2026-08) 기준:
  - `pigment`,`heterogeneity`(UV): **ok** (현재 가장 쓸 만한 신호)
  - `texture`(WHITE): 초점 좋을 때만 ok
  - `pore`,`porphyrin`: 대부분 **unusable** (초점 부족 / 현 데이터 미검출)
- 약한 지표를 강조하지 않는 것이 설계 원칙.

## demo_provider.py — 시연용 예시 데이터
`infer.predict()` 와 **동일 스키마**(regression 5 + classification 7)로 출력.
BE mock 경로에 그대로 꽂아 FE 전체 플로우를 매끄럽게 시연.
```python
from src.demo_provider import demo_full_face, demo_predict, DEMO_BADGE_HTML
resp = demo_predict("L_CHEEK")          # {"regression":..,"classification":..,"source":"demo",..}
```
**정직성 규칙 (필수)**
- 모든 응답에 `source="demo"` + `"예시 데이터 · 실제 측정값 아님"` 이 실린다.
- UI는 데모 응답일 때 `DEMO_BADGE_HTML`(데모 배지)을 반드시 표시.
- ✅ 예시 데이터로 시연(배지 표시) — 정상 시연 기법.
- ❌ 예시 수치를 "모델 실제 정확도/성능"으로 제시 — 데이터 조작. 금지.

## 다음
- 실측 검증: `infer.DamdaEnsembleModel` 로 v3+v5.1 을 scan_images 에 추론 →
  PROGRESS §6 의 "AI-Hub vs ESP32 도메인 비교" 실제 수치 확보(=B 필요성 실증).
- 초점/LED 확산 개선 후 재촬영 → `texture`/`pore` 신뢰도 승격 확인.
- 소량 약지도 확보 시 매크로 전용 경량 회귀로 승격.
