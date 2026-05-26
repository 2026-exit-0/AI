# damda AI 학습 진행 기록

> 졸업 프로젝트 "damda" 의 AI 모델 학습 일지. 각 버전마다 가설 → 변경 → 결과 → 발견 순으로 기록.
> 마지막 업데이트: 2026-05-26 (v4 재개 — epoch 8 부터 진행 중, evaluate.py 작성 완료)

---

## 0. 프로젝트 개요

### 목표
스마트폰/디지털카메라로 촬영한 얼굴 사진에서 **부위별 피부 상태**를 자동 측정.
사용자가 별도 측정 장비 없이 본인의 피부 상태 (수분/탄력/모공/색소/주름)를 정량 수치와 전문가 등급으로 받아볼 수 있도록 함.

### 데이터셋
**AI-Hub 028. 한국인 피부상태 측정 데이터**
- 11,154장 얼굴 이미지 (디지털카메라 D / 스마트패드 T / 스마트폰 P)
- 한 이미지당 9개 facepart JSON 라벨 (PART_0, 이마, 미간, 좌/우 안주변, 좌/우 볼, 입술, 턱)
- manifest 통합 후 100,386행 (이미지 × 부위)

### 접근법: 멀티태스크 + 부위 임베딩
ResNet-50 백본으로 부위별 crop 이미지를 인코딩하고, 어떤 부위인지를 임베딩으로 같이 넣은 뒤 공유 trunk 를 거쳐 **5개 회귀 헤드 + 7개 분류 헤드** 가 동시에 출력. 각 라벨은 부위마다 존재 여부가 다르므로 마스크 기반 손실로 결측은 학습에서 제외.

회귀 (정량 측정값):
- moisture (수분)
- elasticity_mean (탄력, Cutometer R0~R9 평균)
- pore_value (모공 측정값)
- pigmentation_value (색소반점 개수, PART_0 한정)
- wrinkle_value (주름 평균 거칠기 Ra, 양 안주변 한정)

분류 (전문가 등급):
- wrinkle_grade (0~6)
- pigmentation_grade (0~5)
- pore_grade (0~5)
- dryness_grade (0~4)
- sagging_grade (0~6)
- skin_type (0~5)
- sensitive (binary)

---

## 1. 모델 아키텍처

```
              [얼굴 부위 이미지 (224×224)]
                        │
                ResNet-50 backbone
                  (ImageNet pretrained)
                        │
                  [feature 2048]                  [region_id (정수)]
                        │                                 │
                        │                          Embedding(9, 16)
                        │                                 │
                        └──────────┬──────────────────────┘
                                   │ concat
                              [feature 2064]
                                   │
                       Linear(2064 → 512) + ReLU + Dropout
                                   │
                            shared trunk (512)
                          /                     \
              regression_head                 classification_heads (ModuleDict)
              Linear(512, 5)                  Linear(512, K) × 7
              ↓                               ↓
          {moisture, elasticity_mean,         {wrinkle_grade, pigmentation_grade,
           pore_value, pigmentation_value,     pore_grade, dryness_grade,
           wrinkle_value}                      sagging_grade, skin_type, sensitive}
```

**설계 의도**
- 부위가 강한 사전정보이므로 별도 임베딩으로 명시. CNN이 부위를 암묵적으로 추론하기보다는 직접 알려주는 게 효율적.
- 공유 trunk 는 부위 간 표현 공유에 도움 (예: 이마/볼의 피부 결 특징이 부분적으로 유사).
- Phase 2 확장 여지: sensor 입력(온/습도, 광학 센서값 등)을 fusion concat 할 수 있도록 `sensor_branch` 자리 비워둠.

---

## 2. 손실 함수 설계

### 회귀: 마스크 기반 SmoothL1 (Huber)
```
diff       = SmoothL1(pred - target)        # (B, R)
masked     = diff * mask                    # 결측은 mask=0 → 0으로 cancel
loss       = sum(masked) / sum(mask)        # 유효 라벨만 카운트해 평균
```
SmoothL1 선택 이유: outlier 에 강하면서도 작은 오차에서는 L2 처럼 부드럽게 수렴. 측정값에 노이즈가 섞일 수 있는 실측 데이터 특성과 부합.

### 분류: CE with ignore_index + (선택) class_weights
결측 라벨은 target=-1로 표시해 `CrossEntropyLoss(ignore_index=-1)` 로 자동 제외.
v2 에서 sklearn 'balanced' 방식 class_weights 를 도입했으나 역효과 판정 (아래 v2 기록 참고).

### 전체 손실 결합
```
loss = regression_weight × reg_loss  +  classification_weight × cls_avg
     = 1.0 × reg_loss                +  0.5 × mean(cls_loss_per_head)
```
회귀가 분류보다 가중 큰 이유: 측정값이 본 서비스의 핵심 deliverable (등급은 보조). 또 회귀 손실 스케일이 작아 자연스럽게 균형이 잡힘.

---

## 3. 학습 환경

| 항목 | 사양 |
|---|---|
| 졸프실 PC | RTX 5060 8GB VRAM |
| 노트북 (작업/SSH) | OneDrive 동기화, git-bash + SSH 클라이언트 |
| Python | 3.10+ |
| 주요 라이브러리 | torch, torchvision, pandas, tqdm, tensorboard |
| AMP (mixed precision) | CUDA 환경에서 자동 활성화 |
| 배치 사이즈 | 16 (8GB VRAM 안전치) |
| 학습 시간 | epoch당 약 27분, 50 epoch 약 22시간 |

워크플로우: 노트북에서 코드 수정 → git push → 졸프실 PC 에서 git pull → cmd 에서 학습 실행. SSH 는 진단/모니터링 용도.

---

## 4. 버전별 실험 기록

### v1 — Baseline (2026-05-21, 30 epoch)

**가설**
ResNet-50 + 부위 임베딩 + 마스크 멀티태스크 손실 만으로도 회귀/분류가 동시에 학습 가능한지 확인.

**설정**
- lr 0.001, batch 16, Adam, cosine 스케줄
- augmentation: Resize 224, RandomHorizontalFlip, ToTensor + ImageNet Normalize (최소 증강)
- regression_weight=1.0, classification_weight=0.5
- class_weights 미사용

**결과**
- **best val/loss/total = 0.8765 @ epoch 28**
- 추세: epoch 1 (~0.97) → epoch 14 (0.92, plateau 시작) → epoch 28 (0.8765)
- overfitting 신호 없음 (val 계속 감소)
- val/loss/regression 0.3524, val/loss/classification 1.0481

**문제점 (사후 발견)**
- per-head 손실 로깅이 없어서 각 헤드 학습 상태 파악 불가
- (v2 분석 후 알게 됨) pigmentation_value, wrinkle_value 회귀 헤드가 manifest 키 매칭 실패로 **사실상 학습 안 되고 있었음** — total 손실에는 표시 안 됐을 뿐 실제로는 backbone 일부가 무관 gradient 받는 중

**교훈**
- 분포 분석 + per-head 로깅이 baseline 단계부터 필수
- 멀티태스크에서 헤드 1~2개가 학습 안 돼도 total 손실 평균은 그럴듯하게 나옴 → 숫자 하나로 판단하면 안 됨

---

### v2 — Class weights + augmentation 강화 (2026-05-22, 50 epoch)

**가설**
1. v1 plateau 가 epoch 14부터 시작했으니 augmentation 을 강화하면 더 길게 의미있게 학습 가능
2. 약한 분류 헤드 3개(`wrinkle_grade`, `skin_type`, `sensitive`) 는 클래스 불균형 때문 — `class_weights` 로 minority 보강하면 개선될 것
3. 학습률을 약간 낮추면 (0.001 → 0.0006) cosine 스케줄을 더 길게 활용 가능

**변경**
- `epochs: 30 → 50`, `lr: 0.001 → 0.0006`
- augmentation 강화: `Resize(248) + RandomCrop(224)`, `RandomRotation(10°)`, `ColorJitter(brightness/contrast 0.15)`, `RandomErasing(p=0.25)`
- `use_class_weights: true` (sklearn 'balanced' 방식, max_weight=5.0, min_count=5)
- `early_stop_patience: 8`
- per-head 손실 로깅 추가 (`losses.py` 에 `return_per_head=True`)
- 학습셋 기준 회귀 정규화 통계 도입 (mean/std)

**결과**
- **best val/loss/total = 0.8913 @ epoch 45** — v1 대비 **+0.0148 악화**
- val/loss/regression: 0.3524 → **0.3446** (개선 ↓)
- val/loss/classification: 1.0481 → **1.0903** (악화 ↑)

| 분류 헤드 | v1 best | v2 best | Δ |
|---|---|---|---|
| cls_dryness_grade | 1.0291 | 1.0610 | +0.0319 |
| cls_pigmentation_grade | 1.0247 | 1.0925 | +0.0678 |
| cls_pore_grade | 0.8929 | 0.9738 | +0.0809 |
| cls_sagging_grade | 0.9717 | 0.9750 | +0.0033 |
| cls_sensitive | 0.6005 | 0.6259 | +0.0254 |
| cls_skin_type | 1.5213 | 1.5279 | +0.0066 |
| cls_wrinkle_grade | 1.2635 | 1.2779 | +0.0144 |

**문제점**
1. **class_weights 가 역효과**: 약한 헤드(wrinkle/skin_type/sensitive)는 개선 못 하고 잘 되던 헤드들(pore, pigmentation, dryness)을 망쳤음. balanced 방식이 minority 에 과집중하면서 majority 학습이 약해진 듯.
2. **🚨 큰 발견: reg_pigmentation_value / reg_wrinkle_value 가 0.0000**
   per-head 로깅 결과 두 헤드는 epoch 1부터 끝까지 손실이 정확히 0. mask 합이 0 (전부 결측) → 데이터가 없는 셈.
3. 클래스 가중치가 처음에 폭주 (238, 117 등) — cap_ratio*mean 로직이 극단값에 의해 무효화됨. 절대 상한 max_weight=5.0 으로 수정했으나 그래도 효과 없음.

**인사이트**
- v2 의 val/loss/total 숫자는 v1 과 직접 비교 불가 (class_weights 가 CE 스케일을 바꿔놓음). per-head 로 봐야 진실이 보임.
- 데이터 측면 문제(pigmentation/wrinkle 결측)가 알고리즘 측면 문제(class_weights)보다 더 큼.

---

### v2.5 — manifest 진단 (2026-05-22 오후)

**가설**
v2 에서 발견된 pigmentation_value, wrinkle_value 전체 결측이 (a) 진짜 결측인지 (b) manifest 빌더 버그인지 확인.

**진단 방법**
1. `inspect_json.py` 작성 — AI-Hub TL 루트 자동 탐색 후 부위별 JSON 1개씩 까서 equipment/annotations 키 목록 출력.
2. manifest 결측률 직접 측정: pigmentation_value 0/100,386 (100%), wrinkle_value 0/100,386 (100%).

**발견**
원본 JSON 에는 데이터가 **있다**. 단, 빌더가 키 이름을 잘못 매칭하고 있었음.

| facepart | 부위 | equipment 실제 키 |
|---|---|---|
| 0 | PART_0 | `pigmentation_count` (← 빌더는 `pigmentation` 찾음) |
| 3 | L_EYE | `l_perocular_wrinkle_Ra/Rmax/Rt/Rz/Rp/Rv/Rq/R3z` (← 빌더는 `l_perocular_wrinkle` 찾음) |
| 4 | R_EYE | 동일 (r_perocular_*) |
| 1 | FOREHEAD | wrinkle/pigmentation 측정값 **없음**, annotations 에 등급만 있음 |
| 5,6 | L/R_CHEEK | pigmentation 측정값 없음, annotations 에 등급만 있음 |

**중요 인사이트: 부위마다 측정 항목이 완전히 다르다.**
AI-Hub 데이터셋은 부위 특화 측정 — 이마는 수분/탄력, 안주변은 주름 거칠기, 볼은 수분/탄력/모공, 전체는 색소반점 개수 등.

**조치**
빌더 수정 — `pigmentation_value` 는 PART_0 에서만 `pigmentation_count` 로, `wrinkle_value` 는 L/R_EYE 에서만 `*_wrinkle_Ra` 로 채움. 다른 부위는 NaN → mask=0 으로 자동 제외.

**manifest 재생성 결과**
- pigmentation_value: 11.1% 존재 (~11K건, PART_0 한 부위만)
- wrinkle_value: 22.2% 존재 (~22K건, L_EYE + R_EYE)
- moisture: 44.4% 존재 (이마/양볼/턱 4 부위)
- pore_value: 22.2% 존재 (양 볼)
- elasticity_mean: 44.4% 존재

---

### v3 — 데이터 정상화 + class_weights 해제 (2026-05-22 ~ 05-23, 50 epoch)

**가설**
1. v2 의 진짜 문제는 알고리즘이 아니라 **데이터 누락** 이었음. manifest 가 정상화된 지금, 회귀 5개 헤드가 모두 살아있는 상태에서 v1 수준 또는 그 이상 성능 기대.
2. class_weights 는 데이터 결측 상태에서도 역효과였으니 정상 데이터에서도 효과 없을 가능성이 높음 → 제거.
3. 강화된 augmentation 은 회귀 개선에 기여했으므로 유지.
4. lr 을 v1(0.001) 과 v2(0.0006) 의 중간값 0.0008 로 조정 — 약간의 안정성 + 충분한 학습 속도.

**변경**
- `build_manifest.py` 부위 특화 키 매칭 (앞 v2.5 참고)
- manifest 재생성 (100K JSON)
- `regression_targets`: 5개 모두 유지 (실제 데이터 들어옴)
- `use_class_weights: false`
- `lr: 0.0006 → 0.0008`
- augmentation, early_stop, per-head 로깅 등 v2 변경분 유지

**결과 — 전체**
- **best val/loss/total = 0.8234 @ epoch 45** (50/50 full run, early_stop 미발동)
- v1 대비 **−0.0531 (−6.1%)**, v2 대비 **−0.0679 (−7.6%)** 개선
- val/loss/regression: 0.3524 → **0.3109** (−0.0415)
- val/loss/classification: 1.0481 → **1.0244** (−0.0237)
- 학습 추세: epoch 30 이후 0.82xx 범위 진동, epoch 45 이후 plateau

**결과 — per-head (v1 → v3 best)**

회귀:
| 헤드 | v1 | v3 | Δ | 비고 |
|---|---|---|---|---|
| reg_elasticity_mean | (미로깅) | 0.3625 @19 | — | 회귀 중 가장 일찍 best |
| reg_moisture | (미로깅) | 0.4100 @42 | — | 5개 중 절대값 가장 높음 |
| reg_pore_value | (미로깅) | 0.1822 @45 | — | |
| reg_pigmentation_value | (미로깅) | **0.1279 @44** | — | **v2 의 0.0000(가짜) → v3 정상 학습** |
| reg_wrinkle_value | (미로깅) | **0.2169 @49** | — | **v2 의 0.0000(가짜) → v3 정상 학습** |

분류:
| 헤드 | v1 best | v3 best | Δ | 비고 |
|---|---|---|---|---|
| cls_wrinkle_grade | 1.2635 | 1.1962 | **−0.0673** | 최대 개선폭 |
| cls_pigmentation_grade | 1.0247 | 0.9950 | −0.0297 | |
| cls_sagging_grade | 0.9717 | 0.9484 | −0.0233 | |
| cls_dryness_grade | 1.0291 | 1.0114 | −0.0177 | |
| cls_skin_type | 1.5213 | 1.5057 | −0.0156 | **여전히 약함** (절대값 최대) |
| cls_pore_grade | 0.8929 | 0.8804 | −0.0125 | |
| cls_sensitive | 0.6005 | 0.5955 | −0.0050 | binary, near uniform |

**발견**
1. **manifest fix 효과 확정**. v2 에서 0.0000 으로 찍히던 pigmentation/wrinkle 회귀가 v3 에서 정상 손실(0.1279, 0.2169)을 보임. masked loss 로직 자체는 정상이었고, 입력 데이터가 비어있던 게 진짜 원인이었다는 v2.5 진단을 검증.
2. **v2 의 val/loss/regression 0.3446 은 일부 가짜였다**. 5개 헤드 중 2개가 0.0000 으로 평균을 끌어내림. v3 는 5개 모두 진짜 학습한 결과 0.3109 — **표면 차이(0.0337)보다 실질 개선이 더 큼.**
3. **class_weights 제거가 모든 분류 헤드에 +효과**. v2 vs v3 에서 7개 분류 헤드 전부 개선(−0.005 ~ −0.10), 단 한 개도 악화 없음. balanced 가중치가 데이터와 무관하게 역효과였음이 확정.
4. **약한 헤드 우선순위 (v4 후보)**:
   - `cls_skin_type` (1.5057): 6 class uniform=1.79 대비 거의 학습 안 됨. v1→v3 개선폭 −0.0156 로 가장 작음. **focal loss 1순위.**
   - `cls_wrinkle_grade` (1.1962): 절대값 두 번째로 높지만 v1→v3 개선폭은 최대. wrinkle_value 회귀 신호가 도움이 된 것으로 추정.
   - `cls_sensitive` (0.5955): binary uniform=0.69 대비 약간만 학습. 데이터 자체가 거의 정보 없음(near-random 라벨) 가능성도 있어 검토 필요.
5. **수렴 상태**. best @ 45, epoch 38~50 사이 0.82xx 진동. epoch 60~70 으로 늘려도 marginal gain 예상 → **다음 버전은 epoch 늘리기보다 다른 손실/데이터 방향이 효율적.**

**비교 정정**

v2 의 0.8913 을 v1 의 0.8765 와 직접 비교하면 안 됨(앞 v2 절 참고). per-head 로 보면 v2 는 분류 전 헤드 악화 + 회귀 2개 헤드 비활성. v3 는 분류 전 헤드 개선 + 회귀 5개 헤드 활성. **v3 가 v1/v2 둘 모두를 명확히 dominant.**

**산출물**
- `runs/main/` TB 이벤트 (50 epoch)
- `runs/main/val_per_head_v3.csv` — 50 epoch × 15 tag CSV
- `checkpoints/epoch045.pt` — best (실제 추론용)
- `checkpoints/epoch050.pt` — last (재학습 resume 용)

---

### v4 — Focal loss 도입 (진행 중, 2026-05-24 시작)

**가설**
약한 분류 헤드 (cls_skin_type 1.5057, cls_wrinkle_grade 1.1962, cls_sensitive 0.5955) 가 v3 에서도 충분히 학습되지 않은 이유는 클래스 불균형 + easy example 에 손실이 묻혀서. Focal loss (γ=2) 가 hard example 에 손실을 집중시켜 약한 헤드를 개선한다. γ=0 이면 CE 와 동일하므로 강한 헤드 악화 위험은 낮음.

**변경 (v3 대비 minimal — ablation 명확성)**
- `src/losses.py` — `focal_ce_loss` 함수 추가 (ignore_index/class_weight 호환)
- `src/losses.py` — `multitask_loss` 에 `classification_loss_type`, `focal_gamma` 인자 추가
- `src/train.py` — cfg 옵션 전달 + 시작 시 분류 손실 종류 로깅
- `configs/baseline.yaml` — `classification_loss: focal`, `focal_gamma: 2.0` 추가
- 그 외 모두 v3 와 동일: lr 0.0008, batch 16, 50 epoch, cosine, augmentation 강화, class_weights off, manifest v3

**적용 범위**
모든 7개 분류 헤드에 동일 γ=2.0 일괄 적용 (헤드별 차등 안 함). 헤드별 γ 차등은 v5 결정.

**예상**
- best val/loss/total: 0.81 ~ 0.82 (v3 0.8234 대비 marginal 개선)
- cls_skin_type: 1.50 → 1.40 ~ 1.45 (큰 개선 기대)
- cls_wrinkle_grade: 1.20 → 1.10 ~ 1.15
- 잘 되던 헤드: 거의 변화 없음 (focal γ=2 는 easy example 감쇠라 잘 학습된 헤드 영향 적음)

**종료 조건 (성공/실패 판정)**
- ✅ 성공: cls_skin_type 또는 cls_wrinkle_grade 가 v3 대비 −0.03 이상 개선 + 다른 헤드 악화 없음 → v5 에서 헤드별 γ 차등 검토
- ⚠️ 부분: 약한 헤드 개선 있으나 강한 헤드 동시 악화 → v5 에서 약한 3개에만 focal 적용
- ❌ 실패: 약한 헤드 변화 없거나 악화 → focal loss 자체가 부적합. v5 는 데이터 측면 (wrinkle Rmax 보조 회귀, acne 추가) 으로 방향 전환

**준비 작업 (학습 시작 전)**
1. v3 산출물 보존:
   ```cmd
   cd C:\damda\AI
   ren runs\main main_v3
   ren checkpoints checkpoints_v3
   ```
2. v3 체크포인트 디스크 정리 (best + last 만 남기기):
   ```cmd
   cd C:\damda\AI\checkpoints_v3
   ren epoch045.pt _BEST_epoch045.pt
   ren epoch050.pt _LAST_epoch050.pt
   del epoch0??.pt
   ren _BEST_epoch045.pt epoch045.pt
   ren _LAST_epoch050.pt epoch050.pt
   ```
   → 약 13GB 회수
3. git pull (코드 변경분 적용)
4. sanity 먼저: `python -m src.train --config configs/baseline.yaml --validation-mode`
   - 로그에 `분류 손실: focal (gamma=2.0)` 가 찍히는지 확인
5. 본 학습: `train_detach.bat` (schtasks 기반, SSH/노트북 종료 무관)

**진행 상황 — epoch 1~7 (2026-05-24, 1차 시도)**

학습 시작 ~7 epoch (약 3시간) 까지의 val/loss/total 추이:

| epoch | val total | 비고 |
|---:|---:|---|
| 1 | 0.7707 | 첫 평가 |
| 2 | 0.7481 | ↓ |
| 3 | 0.6723 | ↓↓ (큰 폭) |
| 4 | 0.6319 | ↓ |
| 5 | 0.6430 | ↑ (wobble 시작) |
| 6 | 0.6510 | ↑ |
| 7 | 0.6258 | **↓ new best**, wobble 회복 |

해석:
- 7 epoch 만에 −0.145 (−19%) 하강. v3 의 7 epoch 시점보다 빠른 수렴 — focal loss 효과로 추정 (확정은 50 epoch 완주 후)
- epoch 5~6 wobble 0.02 정도는 LR scheduler 경계 / augmentation 분산 / val set 사이즈로 충분히 설명 가능. 4~5 epoch 연속 상승이면 빨간불이지만 2 epoch wobble 은 healthy
- val/loss/regression vs val/loss/classification 분해는 학습 종료 후 dump_tb.py 로 확인 예정

**사고 — SSH 끊김 → Tailscale 데몬 사망 → PC 먹통 (2026-05-24 ~ 26)**

타임라인:
1. **05-24, epoch 7 직후 (~21시 추정)**: SSH 세션 끊김. 학습 살았는지 노트북에서 확인 시도
2. **05-25 ~ 26**: 노트북에서 `ssh DS@100.118.240.67` → `Connection timed out` 반복. `ping` 100% 손실. Tailscale admin 콘솔 점 회색
3. **05-26 오전**: `tailscale status` 로 lab PC `last seen 17h ago, tx 5304 rx 0` 확인 → Tailscale 데몬 사망 99% 확정
4. **05-26 오후, 졸프실 방문**: PC 가 켜져있으나 입력 반응 없음 (먹통). 재부팅 단행
5. **복구 후**: 안정성 보장 작업 (powercfg 절전 차단 + Tailscale/sshd Automatic + sc.exe failure 재시작 + 매일 4시 강제 재시작 보험) 적용 → `train_detach_resume.bat` 실행 → **epoch 8 부터 정상 재개**

근본 원인 (NOTES.md 트러블슈팅 절 상세):
- Windows 가 학습 중 절전 진입 시도 → GPU context 손상 → python 좀비화 → 시스템 hang → Tailscale 데몬 동반 사망
- 체크포인트는 매 epoch 저장이라 손실 없음. epoch 8 부터 깨끗하게 이어짐

**부가 산출물 (2026-05-26)**
- `_train_payload_resume.bat`, `train_detach_resume.bat` — SSH/노트북 종료 안전 resume 런처 (사고 직후 작성)
- `src/evaluate.py` — held-out test set 평가 스크립트 (v4 종료 직후 실행 예정)
- NOTES.md 안정성 보장 절 + 트러블슈팅 절 보강

**진행률 (2026-05-26 기준)**
- epoch 7 / 50 완료 후 사고. 재부팅 후 epoch 8 부터 재개. 남은 ~42 epoch × 28분 ≈ 20시간 예상 → 05-27 오후 ~ 05-28 새벽 종료 예상

---

## 5. 향후 계획 (v5 후보)

### 약한 분류 헤드 처치 — v3 결과로 우선순위 확정
v3 결과에서 약한 헤드의 절대값과 v1→v3 개선폭이 명확해짐.

**1순위: `cls_skin_type` (1.5057, 개선폭 −0.0156)**
- 6 class uniform=1.79 대비 거의 학습 안 됨. v1→v3 개선폭이 7개 헤드 중 가장 작음.
- 우선 클래스 분포 확인 → 1~2개 클래스로 쏠려있다면 focal loss(γ=2) 가 효과 있을 가능성.
- 분포가 균등한데 학습이 안 된다면 라벨 자체가 시각적으로 구분 불가능한 (annotator agreement 낮은) 라벨일 수 있어 평가 방식 재검토 필요.

**2순위: `cls_wrinkle_grade` (1.1962, 개선폭 −0.0673)**
- v3 에서 wrinkle_value 회귀가 살아나면서 분류도 같이 개선됨 → 회귀-분류 보조 신호의 시너지가 확인됨.
- 추가로 wrinkle 의 8개 거칠기 파라미터(Rmax/Rt/Rz/Rp/Rv/Rq/R3z) 를 회귀 보조 헤드로 추가하면 더 좋아질 가능성.

**3순위: `cls_sensitive` (0.5955, 개선폭 −0.0050)**
- binary uniform=0.69 대비 약간만 학습. 데이터 자체에 정보가 거의 없거나, 1개 부위에만 라벨이 있을 가능성. 분포 점검 먼저.

**알고리즘 후보**
- **Focal loss** (γ=2): skin_type 1순위 적용 대상.
- **헤드별 학습률 차등**: skin_type/wrinkle_grade 에 lr×1.5.
- **Stage-2 fine-tuning**: backbone 동결 후 약한 헤드만 추가 학습.

### 데이터 활용 확장
- **PART_0 의 `acne` annotation** — 현재 빌더 누락. 추가하면 여드름 분류/회귀 헤드 신설 가능.
- **wrinkle 8개 거칠기 파라미터** (Rmax, Rt, Rz, ...) — 현재 Ra 만 사용. 다중 회귀 또는 단일 압축 지표로 활용.
- **elasticity Q0~Q3** (점탄성 위상각) — 현재 R0~R9 만 평균. Q계열은 다른 의미.

### 평가 기반 확장
- ✅ `src/evaluate.py` 작성 완료 (2026-05-26) — test set 으로 부위별/헤드별 정확도/MAE 측정. JSON + Markdown + Confusion CSV + per-region 슬라이스 + v3/v4 diff 모드까지. v4 종료 직후 실행 예정.
- ⬜ 회귀 산점도 (예측 vs 실제) — evaluate.py 의 `--save-predictions` CSV 를 matplotlib 으로 시각화. 졸업논문 figure 후보.
- ⬜ 부위별 약점 표 정리 — evaluate.py per_region 출력을 LaTeX 표로 변환.

### 모델 아키텍처
- 헤드별 trunk 분리 — 현재 단일 trunk 512 인데, 회귀와 분류는 trunk 가 달라야 더 잘 학습될 가능성. 다만 파라미터 늘어서 데이터 양 대비 검토 필요.
- 부위별 backbone freeze 비율 — 부위 임베딩이 약하게 작용하면 backbone 이 부위에 따라 적응 못 함. 임베딩 차원 16 → 32 검토.

---

## 6. 메타: 학습/실험 워크플로우 개선

- ✅ per-head 손실 자동 로깅 (`losses.py`)
- ✅ TB 이벤트 → CSV 덤프 (`dump_tb.py`)
- ✅ AI-Hub JSON 키 자동 진단 (`inspect_json.py`)
- ✅ early_stopping
- ✅ 부위 특화 manifest 빌더
- ✅ test set 평가 분리 (`evaluate.py`) — 2026-05-26
- ✅ SSH/노트북 종료 안전 학습 런처 (`train_detach.bat`, schtasks 기반) — v3 종료 시점
- ✅ SSH/노트북 종료 안전 resume 런처 (`train_detach_resume.bat`) — 2026-05-26 사고 대응
- ✅ Tailscale/sshd 자동시작 + 절전 차단 일괄 정리 (NOTES.md "안정성 보장" 블록) — 2026-05-26
- ⬜ 매 epoch 정확도/F1 도 train.log 에 기록 (현재 val total 만)
- ⬜ best ckpt 자동 별도 보관 (`best.pt`) — 현재는 매 epoch 저장이라 best 찾으려면 epoch 번호 알아야 함
- ⬜ python 프로세스 자동 재시작 (NSSM 서비스화 또는 watcher) — 현재는 schtasks 1회 실행만
- ⬜ Wake-on-LAN 으로 원격 PC 복구 — 현재는 졸프실 물리방문 필수
