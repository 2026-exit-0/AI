# damda AI 프로젝트 노트

> 졸프 진행하며 쌓이는 핵심 메모와 명령어. 새로 알게 된 사실이나 자주 쓰는 명령이 생기면 이 파일에 갱신.
> 마지막 업데이트: 2026-05-24 (v4 시작 — focal loss γ=2.0)

---

## 1. 데이터셋 핵심 사실 (AI-Hub 028 한국인 피부상태)

- **이미지 11,154장** × 9 facepart = manifest **100,386행**
- 디바이스: D(디카) 54,054 / T(패드) 23,166 / P(폰) 23,166
- bbox 결측 4.3% (일부 JSON에 bbox 없음 — 학습 시 crop 생략)

### ⚠ 부위별 측정 항목이 **모두 다름** (큰 함정)

equipment(측정값) ≠ annotations(전문가 등급). 부위마다 둘 다 있을 수도, 한쪽만 있을 수도 있음.

| facepart | 부위 | equipment | annotations |
|---|---|---|---|
| 0 | PART_0 | `pigmentation_count` | `acne` |
| 1 | FOREHEAD | `moisture`, `elasticity_R0~R9` | `pigmentation`, `wrinkle` |
| 2 | GLABELLA | (없음) | `wrinkle` |
| 3 | L_EYE | `*_wrinkle_Ra/Rmax/Rt/Rz/Rp/Rv/Rq/R3z` | `wrinkle` |
| 4 | R_EYE | 동일 (r_perocular_*) | `wrinkle` |
| 5 | L_CHEEK | `moisture`, `elasticity`, `pore` | `pigmentation`, `pore` |
| 6 | R_CHEEK | 동일 | `pigmentation`, `pore` |
| 7 | LIP | (없음) | `dryness` |
| 8 | CHIN | `moisture`, `elasticity` | `sagging` |

### 회귀 헤드와 부위 매핑 (v3 빌더 기준)

| 회귀 타겟 | 채워지는 부위 | 결측률 | 키 |
|---|---|---|---|
| moisture | FOREHEAD, L/R_CHEEK, CHIN | 55.6% | `{prefix}_moisture` |
| elasticity_mean | 동일 | 55.6% | `{prefix}_elasticity_R0~R9` 평균 |
| pore_value | L/R_CHEEK | 77.8% | `{prefix}_pore` |
| pigmentation_value | PART_0 만 | 88.9% | `pigmentation_count` |
| wrinkle_value | L/R_EYE 만 | 77.8% | `{prefix}_wrinkle_Ra` |

> mask 기반 손실이라 결측 행은 자동 제외됨. 추론 시에도 부위 확인 후 해당 헤드만 사용해야 함.

---

## 2. 자주 쓰는 명령어

### 학습 (졸프실 PC cmd, SSH 권장)

**가장 단순 — `.bat` 래퍼 사용**:

| 래퍼 | 용도 | 노트북 꺼도 OK? |
|---|---|---|
| `train_detach.bat` | 새 학습 시작 (**SSH/노트북과 완전 분리, 권장**) | ✅ 가능 |
| `train_main.bat` | 새 학습 시작 (같은 cmd에서 `start /B`) | ⚠ SSH 종료 시 학습 죽을 수 있음 |
| `train_resume.bat` | 끊긴 학습 이어가기 (`--resume`) | ⚠ 동일 |

```cmd
cd C:\damda\AI

REM ★ 권장: 노트북 꺼도 학습 유지 (schtasks 사용)
train_detach.bat

REM 같은 세션 안에서만 백그라운드 (간단 테스트용)
train_main.bat

REM 끊긴 학습 이어가기
train_resume.bat
```

**`train_detach.bat` 작동 원리**
- Windows 작업 스케줄러(`schtasks`)에 "damda-train" task 등록 후 즉시 실행
- 학습은 SSH cmd 의 자식이 아니라 schtasks 컨텍스트의 자식이 됨 → 부모 SSH 가 죽어도 영향 없음
- 실제 학습 진입점은 `_train_payload.bat` (venv 자동 활성화 + python 직접 호출, `start /B` 없음)
- task 관리:
  ```cmd
  REM 상태 확인 (Status: Running 이면 정상)
  schtasks /query /tn "damda-train" /v /fo LIST

  REM 학습 강제 중단
  schtasks /end /tn "damda-train"

  REM task 삭제 (학습 완전 끝난 뒤)
  schtasks /delete /tn "damda-train" /f
  ```

**직접 명령으로 하고 싶을 때**:
```cmd
cd C:\damda\AI
myvenv\Scripts\activate.bat

REM 본 학습 (foreground)
python -m src.train --config configs/baseline.yaml > train_console.log 2>&1

REM 백그라운드 분리 실행 (SSH 끊김 영향 최소화)
start "damda-train" /B python -m src.train --config configs/baseline.yaml > train_console.log 2>&1

REM 1차 sanity (validation-mode, 약 10분)
python -m src.train --config configs/baseline.yaml --validation-mode

REM 끊긴 학습 이어가기
python -m src.train --config configs/baseline.yaml --resume
python -m src.train --config configs/baseline.yaml --resume-from checkpoints\epoch020.pt
```

**SSH 끊김 대응**
- 체크포인트가 매 epoch 저장되므로 끊겨도 최대 ~27분 손실
- 다시 접속 후 `train_resume.bat` 한 줄
- SSH 끊김 ≠ 학습 종료 (Windows OpenSSH는 자식 종료 안 시키는 경우 多). `nvidia-smi` 와 train.log tail 로 살아있는지 먼저 확인

**졸프실 PC 절전 비활성화** (한 번만, 관리자 cmd 필요)
```cmd
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change disk-timeout-ac 0
REM 디스플레이도 안 끄려면: powercfg /change monitor-timeout-ac 0

REM 설정 확인
powercfg /query SCHEME_CURRENT SUB_SLEEP
```
PC 가 절전 들어가면 GPU 멈춰서 학습 중단됨. 노트북 sleep 은 학습에 무관.

### 학습 중 진행 확인
```cmd
REM val 손실 추세
findstr /C:"val total" runs\main\train.log

REM 마지막 5줄 (UTF-8)
powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"

REM GPU (정상이면 70%+)
nvidia-smi

REM 파일 크기 변화로 학습 살아있는지 확인
dir train_console.log
```

### per-head 손실 분석 (v1/v2/v3 비교)
```cmd
REM 단일 run 요약
python -m src.dump_tb runs\main

REM 두 run 비교 (delta 자동 계산)
python -m src.dump_tb runs\main_v2 runs\main

REM CSV 내보내기
python -m src.dump_tb runs\main --csv runs\main\val_per_head.csv
```

### 데이터셋 진단
```cmd
REM AI-Hub JSON 키 자동 탐색
python -m src.inspect_json
python -m src.inspect_json --n 20

REM manifest 결측률
python -c "import pandas as pd; df=pd.read_csv('data/manifest.csv'); print((df.isna().mean()*100).round(1))"

REM 특정 컬럼 통계
python -c "import pandas as pd; df=pd.read_csv('data/manifest.csv'); print(df['moisture'].describe())"
```

### manifest 재생성
```cmd
python -m src.build_manifest ^
    --image-root "C:\damda\dataset\028.한국인 피부상태 측정 데이터\3.개방데이터\1.데이터\Training\01.원천데이터\TS" ^
    --json-root  "C:\damda\dataset\028.한국인 피부상태 측정 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터\TL" ^
    --output     data\manifest.csv
```

### 새 버전 시작 시 정리
```cmd
REM 직전 결과 보존 + 체크포인트 초기화
if exist runs\main ren runs\main main_v2
if exist checkpoints rmdir /s /q checkpoints
```

### git 워크플로우 (노트북 ↔ 졸프실 PC)
```bash
# 노트북 git-bash
cd /c/Users/YSB/OneDrive/Desktop/2026-damda
git add AI/...
git commit -m "..."
git push
```
```cmd
REM 졸프실 PC cmd
cd C:\damda\AI
git pull
```

### 커밋 메시지 / PR body 규칙

**커밋 메시지** — `<type>: <한국어 요약 50자 내외>` 형식
- `docs:` 문서만 (PROGRESS, NOTES, README)
- `feat:` 새 기능 (새 헤드, 새 손실, 새 스크립트)
- `fix:` 버그 수정
- `refactor:` 기능 변화 없는 구조 개선
- `chore:` 빌드/설정/배치 스크립트
- `experiment:` 실험 설정만 변경 (yaml)

예: `experiment: v4 focal loss 도입 (skin_type 타겟)`

**PR body** — 첫 줄은 위 커밋 메시지, 빈 줄, 그 다음 본문. Claude 가 작성해주는 PR body 도 항상 이 형식을 따름.
```markdown
<type>: <요약>

## <상세 제목>
... 본문 ...
```

---

## 3. 트러블슈팅 (이미 겪은 것들)

### 한글 깨짐 (mojibake)
cmd/PowerShell 기본 인코딩이 cp949. 두 가지 해결:
```cmd
chcp 65001
```
```ps1
Get-Content runs\main\train.log -Encoding UTF8
```

### `findstr` 가 한글 검색 못 함
findstr 는 cp949 모드라 UTF-8 한글 매칭 실패. 영문 토큰으로 검색하거나 PowerShell 사용:
```cmd
REM ✗ 안 됨
findstr "회귀 정규화" runs\main\train.log

REM ✓ 됨 (영문)
findstr "mean=" runs\main\train.log
findstr "moisture" runs\main\train.log

REM ✓ 됨 (PowerShell)
powershell -Command "Get-Content runs\main\train.log -Encoding UTF8 | Select-String '회귀'"
```

### PowerShell 가상환경 활성화 스크립트 차단
```ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
또는 cmd 에서 `.bat` 사용: `myvenv\Scripts\activate.bat`

### Ctrl+C 후 터미널 먹통
DataLoader `num_workers > 0` 워커 정리 실패. 순서:
1. Ctrl+C 다시 한 번
2. Ctrl+Break
3. 다른 cmd에서 `taskkill /F /IM python.exe`
4. 안 되면 창 닫고 새로 열기

근본 회피용: `configs/baseline.yaml` 에서 `num_workers: 0` (디버깅 한정)

### git 인식 안 됨 (PATH 문제, git 자체는 설치돼 있음)
임시 (현재 세션만):
```ps1
$env:Path += ";C:\Program Files\Git\cmd"
```
영구:
```ps1
[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Program Files\Git\cmd", "User")
```

### SSH 셸이 cmd (bash 아님)
Windows OpenSSH 서버 기본 셸이 cmd. heredoc(`<<PY`) 안 통함.
→ 한 줄 `python -c "..."` 또는 `.py` 파일을 git pull 받아 실행.

### `head`, `tail` 같은 유닉스 명령 없음
cmd에는 없음. 대체:
```cmd
REM head -5
powershell -Command "Get-Content file.txt -Head 5"
REM tail -5
powershell -Command "Get-Content file.txt -Tail 5"
```

### 체크포인트 덮어쓰기 사고
validation-mode 가 같은 `checkpoints/` 폴더에 쓰므로 본 학습 ckpt 위에 덮어쓸 위험.
→ 본 학습 끝나면 즉시 `ren checkpoints checkpoints_v?` 백업할 것.

---

## 4. 실험 이력

### v1 (baseline, 2026-05-21, 30 epoch)
- lr 0.001, augmentation 약함, class_weights 없음
- **best val/loss/total: 0.8765 @ epoch 28**
- 회귀 5개 헤드 중 2개(pigmentation_value, wrinkle_value)가 빌더 키 매칭 실패로 학습 안 됨 (당시엔 모름)

### v2 (2026-05-22, 50 epoch)
- lr 0.0006, augmentation 강화, class_weights ON
- **best val/loss/total: 0.8913 @ epoch 45** (v1보다 +0.015 나빠짐)
- val/loss/regression 은 개선 (0.3524 → 0.3446) but val/loss/classification 악화 (1.0481 → 1.0903)
- 진단: class_weights 가 약한 헤드도 개선 못 하고 잘 되던 헤드들을 망쳤음 → 역효과
- per-head reg 로깅으로 pigmentation_value, wrinkle_value 가 통째 0 인 것 발견 → 빌더 버그 추적의 시작

### v4 (진행 중, 2026-05-24 시작)
- v3 동일 + **classification_loss: focal, focal_gamma: 2.0** (7개 헤드 일괄)
- 타겟: cls_skin_type (1.5057) / cls_wrinkle_grade (1.1962) 개선
- 종료 조건: 약한 헤드 −0.03 개선 + 강한 헤드 악화 없음 → 성공

### v3 (2026-05-22 ~ 05-23, 50 epoch)
- 빌더 수정: pigmentation_value 는 PART_0의 `pigmentation_count`, wrinkle_value 는 L/R_EYE의 `*_wrinkle_Ra`
- class_weights OFF, lr 0.0008
- 회귀 헤드 5개 모두 실제 데이터로 학습
- **best val/loss/total: 0.8234 @ epoch 45** (v1 대비 −0.053 / −6.1%, v2 대비 −0.068 / −7.6%)
- 회귀 0.3524 → 0.3109, 분류 1.0481 → 1.0244 (모든 7개 헤드 일제히 개선, 단 한 개도 악화 없음)
- 새로 살아난 헤드: reg_pigmentation_value 0.1279, reg_wrinkle_value 0.2169 (v2 의 0.0000 은 manifest 결측이었음이 검증)
- 약한 헤드 우선순위 확정: skin_type (1.5057, 거의 학습 안 됨, v4 focal loss 1순위) > wrinkle_grade (1.1962, 개선폭 최대) > sensitive (0.5955, 정보량 부족 의심)
- 수렴 상태: epoch 38~50 사이 0.82xx 진동. epoch 더 늘리기보다 다른 손실/데이터 방향 권장

---

## 5. 알려진 이슈 / 다음 후보

- [ ] **test set 평가 스크립트** (별도 `src/evaluate.py`) 미작성
- [ ] **약한 분류 헤드** (v3 best 기준: skin_type 1.5057, wrinkle_grade 1.1962, sensitive 0.5955) — class_weights 로 해결 안 됨. v4 후보: focal loss (skin_type 1순위) / head 별 lr 차등 / sensitive 라벨 분포 점검
- [ ] **체크포인트 디스크 정리** — 50 epoch × 295MB = 14.7GB. best(epoch045) + last(epoch050) 외 삭제 권장
- [ ] **PART_0 의 `acne` annotation** 빌더에서 누락. 별도 회귀/분류 헤드로 확장 가능
- [ ] **wrinkle 8개 거칠기 파라미터** 중 Ra 만 사용. Rmax/Rt 등 7개도 정보가 있음
- [ ] **`elasticity_Q0~Q3`** (점탄성 위상각) 빌더가 R0~R9 만 평균. Q 계열도 별도 헤드 검토 가능
- [ ] **부위별 학습률/손실 가중치 차등** — 부위가 한쪽으로 쏠리면 부위 임베딩이 그쪽으로 편향될 가능성

---

## 6. 환경 메모

### venv 위치 (PC별 다름)
- **졸프실 PC**: `C:\damda\.venv` (AI 폴더의 **부모**)
- **노트북**: `C:\Users\YSB\OneDrive\Desktop\2026-damda\AI\myvenv` (AI 폴더 안)

수동 활성화 (필요 시):
```cmd
REM 졸프실 PC
call C:\damda\.venv\Scripts\activate.bat

REM 노트북
call myvenv\Scripts\activate.bat
```

`train_main.bat` / `train_resume.bat` 는 양쪽 위치 모두 자동 감지함.

---

## 7. 파일 구조 요약

```
AI/
  configs/baseline.yaml      ← 학습 설정 (lr, epochs, regression_targets 등)
  data/manifest.csv          ← 단일 manifest (build_manifest.py 결과, gitignored)
  src/
    build_manifest.py        ← AI-Hub JSON → manifest 통합
    dataset.py               ← PyTorch Dataset + collate + class_weights + reg_stats
    model.py                 ← ResNet-50 + region embedding + 다중 헤드
    losses.py                ← 마스크 회귀 + class_weights CE
    train.py                 ← 학습 루프
    utils.py                 ← REGION_TO_ID, REGION_TO_JSON_PREFIX, logger 등
    inspect_json.py          ← AI-Hub JSON 키 진단 (인자 없이 자동 탐색)
    dump_tb.py               ← TB 이벤트 파일에서 per-head 손실 추출 + 비교
  runs/                      ← TensorBoard 로그 (main, baseline_val, main_v1, main_v2 ...)
  checkpoints/               ← 학습 중 epoch별 ckpt (gitignored)
  myvenv/                    ← 가상환경 (gitignored)
```
