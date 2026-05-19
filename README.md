# damda — AI

담다(DAMDA) 프로젝트의 AI 모델링 코드.
한국인 피부상태 측정 데이터(AI-Hub)를 활용해 이미지 기반 다중 출력 피부 상태 추정 모델을 학습한다.

## Phase 전략

| Phase | 입력 | 출력 | 시점 |
| --- | --- | --- | --- |
| **Phase 1** (지금) | 이미지 + 부위 ID | 수분·탄력·모공·색소침착 + 등급·피부타입 | AI-Hub 데이터로 본 학습 |
| **Phase 2** | + ESP32 센서값(수분·조도) | 동일 | 하드웨어 데이터 누적 후 fine-tune |
| **Phase 3** | + ESP32-CAM 도메인 적응 재전처리 | 동일 | 데모 직전 |

`src/model.py`의 `forward(image, region_id, sensor=None)` 시그니처는 Phase 2 확장을 전제로 설계되어 있어, sensor 분기만 채우면 된다.

## 디렉터리

```
AI/
├── README.md
├── requirements.txt
├── configs/
│   └── baseline.yaml
├── src/
│   ├── __init__.py
│   ├── utils.py               # 시드, 디바이스 감지(CUDA/MPS/CPU), 로거, 부위 매핑
│   ├── build_manifest.py      # raw JSON → manifest.csv 변환
│   ├── dataset.py             # PyTorch Dataset (마스크 기반 다중 라벨)
│   ├── model.py               # ResNet-50 + 부위 임베딩 + 멀티헤드 (센서 확장 자리)
│   ├── losses.py              # 마스크 회귀 + ignore_index 분류
│   └── train.py               # 학습 루프 (시드·AMP·체크포인트·TensorBoard)
└── data/
    └── manifest.csv           # build_manifest.py 결과물
```

## 데이터 다운로드 전략

전체 데이터셋(125,424건, 수십~수백 GB 추정)을 한 번에 받지 말 것. 노트북에서는 단계적 확장을 권장한다.

| 단계 | 규모 | 목적 |
| --- | --- | --- |
| A. **Baseline Validation** | 300~500장 | 파이프라인 동작 확인 |
| B. **Mini-Train** | 1~2만 건 (10~15%) | 데모용 학습 모델 확보 |
| C. **Full-Train** | 전체 | 공모전·논문 정량 평가용 (선택) |

A → B → C 순서로 확장. 단계마다 학습/평가를 끝낸 뒤 다음 단계 진입.

## 학습 환경

팀 내 두 가지 환경을 모두 지원한다.

### A. Google Colab (무료 T4 GPU)

`notebooks/colab_train.ipynb`을 Colab에서 열어 그대로 실행. Drive 마운트 → 경로 설정 → manifest 빌드 → Baseline Validation → TensorBoard 흐름이 모두 포함되어 있다.

**준비**: 코드(AI/)와 데이터를 Drive에 다음 구조로 업로드.

```
MyDrive/2026-damda/
├── AI/         # 이 코드 폴더 통째로 업로드
└── data/
    ├── images/{REGION}/*.jpg
    └── labels/*.json
```

런타임 메뉴에서 `T4 GPU` 선택 후 노트북 첫 셀부터 순서대로 실행.

### B. NVIDIA GPU 로컬 (팀원용)

```powershell
cd "<프로젝트 경로>\2026-damda\AI"
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

# CUDA 12.4 빌드 (드라이버 확인 후 cu121, cu118 등으로 바꿔도 됨)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# 확인
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

VRAM별 `configs/baseline.yaml` 권장값:

| VRAM | batch_size | image_size |
| --- | --- | --- |
| 4GB | 8 | 192 |
| 6GB | 16 | 224 |
| 8GB | 24 | 224 |
| 12GB+ | 32~64 | 224 |

### C. CPU 전용 (비권장)

`src/utils.py`의 `get_device()`가 CPU로 자동 폴백한다. Baseline Validation까지는 가능하나 본 학습은 비현실적이므로 Colab 사용 권장.

---

`src/utils.py`의 `get_device()`는 CUDA → MPS(Apple Silicon) → CPU 순으로 자동 감지하고, AMP(Mixed Precision)는 CUDA에서만 활성화된다.

## 실행 순서

### 0. 환경 준비 (1회)

```powershell
cd "C:\Users\YSB\OneDrive\Desktop\2026-damda\AI"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch는 환경에 따라 별도 설치가 필요할 수 있다.
- NVIDIA CUDA: https://pytorch.org/get-started/locally/ 에서 명령 복사
- 그 외: `pip install torch torchvision` (CPU 버전)

### 1. 데이터 매니페스트 생성

AI-Hub raw JSON을 단일 manifest.csv로 정규화. 이미지 폴더 구조는 부위명(`L_CHEEK`, `FOREHEAD` 등)별로 나뉘어 있다고 가정한다.

```powershell
python -m src.build_manifest `
  --image-root "샘플_이미지_루트_경로" `
  --json-root  "샘플_JSON_루트_경로" `
  --output     data/manifest.csv
```

처음 실행 시 `--limit 200`을 붙여 빠르게 동작 확인 권장.

manifest.csv 컬럼:
`image_path, region, region_id, subject_id, gender, age, skin_type, sensitive, moisture, elasticity_mean, pore_value, pigmentation_value, wrinkle_value, wrinkle_grade, pigmentation_grade, pore_grade, dryness_grade, sagging_grade` (없는 값은 NaN)

### 2. Baseline Validation

500장 이하로 학습 → loss 감소 여부 확인.

```powershell
python -m src.train --config configs/baseline.yaml --validation-mode
```

통과 기준 — epoch마다 train loss 감소 / val loss 진동 없음. 이 단계 실패 시 전처리·라벨 매핑부터 점검.

### 3. 본 학습

```powershell
python -m src.train --config configs/baseline.yaml
```

체크포인트 → `checkpoints/`, TensorBoard 로그 → `runs/`.

## 라벨 매핑 (AI-Hub → manifest)

| AI-Hub raw | 정규화 컬럼 | 타입 |
| --- | --- | --- |
| `{region}_moisture` | `moisture` | 회귀 |
| `{region}_elasticity_R0` ~ `R9` 평균 | `elasticity_mean` | 회귀 |
| `{region}_elasticity_Q0` ~ `Q3` | (현재 미사용) | — |
| `{region}_pore` (equipment) | `pore_value` | 회귀 |
| `{region}_pigmentation` (equipment, 있을 때) | `pigmentation_value` | 회귀 |
| `{region}_wrinkle` (equipment, 있을 때) | `wrinkle_value` | 회귀 |
| `{region}_pore` (annotations) | `pore_grade` | 분류 |
| `{region}_pigmentation` (annotations) | `pigmentation_grade` | 분류 |
| `{region}_wrinkle` (annotations) | `wrinkle_grade` | 분류 |
| `{region}_dryness` (annotations) | `dryness_grade` | 분류 |
| `{region}_sagging` (annotations) | `sagging_grade` | 분류 |
| `info.skin_type` | `skin_type` | 분류 |

부위마다 사용 가능한 라벨이 다르므로 `dataset.py`가 라벨 마스크를 함께 반환하고, `losses.py`는 마스크가 1인 라벨만 손실에 포함한다.

## TODO

- [ ] Q0~Q3 elasticity 활용 방안 결정 (다중 측정값 의미 파악 후)
- [ ] `info.sensitive` 헤드 추가
- [ ] Domain adaptation: ESP32-CAM 시뮬레이션 전처리 (JPEG q10, downsample, 색온도 매칭)
- [ ] FastAPI 추론 endpoint
