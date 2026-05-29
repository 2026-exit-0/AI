# damda AI 프로젝트 노트

> 졸프 진행하며 쌓이는 핵심 메모와 명령어. 새로 알게 된 사실이나 자주 쓰는 명령이 생기면 이 파일에 갱신.
> 마지막 업데이트: 2026-05-29 (v5.1 채택 / v5.5 실패 / evaluate.py 이종 ensemble 지원)

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
| `train_detach_resume.bat` | **끊긴 학습 이어가기 (SSH/노트북과 완전 분리)** | ✅ 가능 |
| `train_main.bat` | 새 학습 시작 (같은 cmd에서 `start /B`) | ⚠ SSH 종료 시 학습 죽을 수 있음 |
| `train_resume.bat` | 끊긴 학습 이어가기 (`--resume`, 같은 cmd) | ⚠ 동일 |

```cmd
cd C:\damda\AI

REM ★ 권장: 노트북 꺼도 학습 유지 (schtasks 사용)
train_detach.bat

REM ★ 끊긴 학습 이어가기 — SSH/노트북과 완전 분리
train_detach_resume.bat

REM 같은 세션 안에서만 백그라운드 (간단 테스트용)
train_main.bat

REM 끊긴 학습 이어가기 (같은 cmd, SSH 끊김 위험)
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

**졸프실 PC 안정성 보장 — 무인 학습 전 필수 (한 번만, 관리자 권한)**

야간 무인 학습 사고 (2026-05-24~26) 의 직접 원인이 ① 절전 진입 + ② Tailscale 데몬 사망이었음. 셋 다 묶어서 처리.

```cmd
:: ① 절전/최대절전 완전 차단 (관리자 cmd)
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change disk-timeout-ac 0
powercfg /h off

:: 설정 확인
powercfg /query SCHEME_CURRENT SUB_SLEEP
```

```powershell
# ② Tailscale 자동시작 + 죽으면 자동 재시작 (관리자 PowerShell)
Set-Service Tailscale -StartupType Automatic
sc.exe failure Tailscale reset= 86400 actions= restart/60000/restart/60000/restart/60000

# ③ sshd 자동시작 + 방화벽 22번 허용
Set-Service sshd -StartupType Automatic
Start-Service sshd
Enable-NetFirewallRule -Name "OpenSSH-Server-In-TCP"

# 상태 확인 (둘 다 Running 이어야 함)
Get-Service Tailscale, sshd
```

```cmd
:: (선택) 매일 새벽 4시 Tailscale 강제 재시작 — 데몬 사망 보험
schtasks /create /tn "tailscale-restart" /tr "powershell -Command Restart-Service Tailscale" /sc daily /st 04:00 /ru SYSTEM /rl HIGHEST
```

PC 가 절전 들어가면 GPU context 손상 + Tailscale 데몬 죽음 + 학습 프로세스 좀비화 콤보로 PC 가 통째로 먹통됨. 노트북 sleep 은 학습에 무관.

**자동 재시작이 보장되는 것 / 안 되는 것**

| 항목 | 자동 복구? |
|---|---|
| Tailscale 데몬 | ✅ (위 ② 설정 시) |
| sshd | ✅ (위 ③ 설정 시) |
| PC 절전 진입 | ❌ 진입 자체가 차단 (위 ① 설정 시) |
| **학습 python 프로세스** | **❌ NO** — OOM/CUDA crash 시 그대로 죽음. schtasks 는 1회 실행만 |
| PC 재부팅 후 학습 | ❌ 누군가 `train_detach_resume.bat` 다시 돌려야 함 |

### 학습 중 진행 확인

**4종 세트** — 역할이 다 다름. 학습 살았는지 의심될 때 위에서부터 차례로:

| 명령 | 보여주는 것 |
|---|---|
| `schtasks /query /tn "damda-train" /fo LIST \| findstr Status` | task 자체 (`Running` / `Ready`(=끝)) |
| `powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"` | epoch 진행, val total 추세 |
| `nvidia-smi` | GPU 가 일하고 있나 (VRAM 점유 + Util%) |
| `dir checkpoints` | epoch 가 실제로 끝났나 (27분마다 ckpt 추가) |

```cmd
REM val 손실 추세
findstr /C:"val total" runs\main\train.log

REM 마지막 5줄 (UTF-8)
powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"

REM GPU (정상이면 70%+. 시작 직후 1~2분은 0% 일 수 있음 — 데이터로딩 워밍업)
nvidia-smi

REM 파일 크기 변화로 학습 살아있는지 확인
dir train_console.log
```

**학습 완료 신호**
- `schtasks /query` Status: `Ready` (실행 끝남)
- `train.log` 마지막 줄: `학습 완료. best val loss = X.XXXX @ epochNN`
- `checkpoints/` 에 epochNNN.pt 가 의도한 마지막 epoch 까지 다 있음

**끄기 전 sanity check (학습 시작 직후 1~2분 후 권장)**
```cmd
schtasks /query /tn "damda-train" /fo LIST | findstr Status
powershell -Command "Get-Content C:\damda\AI\runs\main\train.log -Tail 10 -Encoding UTF8"
nvidia-smi
```
이 셋 다 정상이면 SSH/노트북 꺼도 안전.

### per-head 손실 분석 (학습 중 val loss 곡선)
```cmd
REM 단일 run 요약
python -m src.dump_tb runs\main

REM 두 run 비교 (delta 자동 계산)
python -m src.dump_tb runs\main_v2 runs\main

REM CSV 내보내기
python -m src.dump_tb runs\main --csv runs\main\val_per_head.csv
```

### held-out test set 평가 (학습 종료 후)

`dump_tb.py` 는 학습 중 val loss 만 보여줌. evaluate.py 는 학습 끝난 후 test split 에서 **MAE / Accuracy / Macro F1 / Confusion matrix** 같은 사람용 메트릭을 뽑음 (졸업논문 표용).

```cmd
REM 기본 — test split 평가, JSON+MD+confusion CSV 자동 생성
python -m src.evaluate ^
    --config configs/baseline.yaml ^
    --checkpoint checkpoints\epoch030.pt ^
    --split test ^
    --config-version v4

REM v3 ckpt 도 평가 후 diff 표 자동 생성
python -m src.evaluate --config configs/baseline.yaml ^
    --checkpoint checkpoints\v3_epoch045.pt --split test --config-version v3

python -m src.evaluate --config configs/baseline.yaml ^
    --checkpoint checkpoints\epoch030.pt --split test --config-version v4 ^
    --compare-to runs/eval/v3_epoch045_test.json

REM 오류 사례 수동 분석용 — 샘플별 예측/정답 CSV
python -m src.evaluate ... --save-predictions runs/eval/v4_preds.csv

REM 부위 슬라이스 끄기 (기본은 ON)
python -m src.evaluate ... --no-per-region
```

기본 출력: `runs/eval/<ckpt_stem>_<split>.{json,md}` + `runs/eval/<ckpt_stem>_<split>_cm/<head>.csv`. 학습 split (seed=42, split_by='id') 와 100% 동일하게 재현되므로 train data 누설 없음.

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

### evaluate.py `--ensemble` 이종 architecture 에러 (2026-05-29)

증상 1 — sensor mismatch:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x0 and 1x32)
  → model.py 의 sensor_branch(sensor) 에서 발생
```

증상 2 — regression head 수 mismatch:
```
RuntimeError: The size of tensor a (5) must match the size of tensor b (4) at non-singleton dimension 1
  → run_inference 의 reg_accum + o["regression"] 에서 발생
```

원인: ensemble 멤버들의 architecture 가 다른데 (예: v3 는 sensor 없음, v5.1 은 moisture sensor / v3 는 5 회귀 헤드, v5.1 은 4 헤드), dataset / metric 이 첫 ckpt 기준이라 mismatch.

해결 (evaluate.py 의 구조):
- **sensor / categorical**: 모든 멤버의 **union** 으로 dataset 구성. 각 모델은 자기 sensor_inputs_list 에 맞춰 column slice (모델별 `_sensor_inputs_list`, `_categorical_inputs_dict` 메타 부착)
- **regression / classification 헤드**: 모든 멤버의 **intersection** (공통) 만 평균. dataset 도 intersection 으로 만들어 라벨 shape 일치. 각 모델 출력에서 `_regression_targets_list` index 로 column slice
- 평가 결과의 composite / mean MAE / mean F1 은 intersection 헤드만 반영 → 단일 모델 평가 결과와 직접 비교 시 헤드 수 차이 주의

검증 흐름:
```cmd
:: v3 (5 reg, sensor 없음) + v5.1 (4 reg, moisture sensor)
python -m src.evaluate --config configs/baseline.yaml ^
    --ensemble checkpoints_v3\epoch045.pt,checkpoints_v5.1\epoch048.pt ^
    --split test --tta
:: 로그 확인:
::   ensemble union: sensor=['moisture'], categorical={}
::   ensemble 공통 회귀 헤드: ['elasticity_mean', 'pore_value', 'pigmentation_value', 'wrinkle_value']
::   ensemble 공통 분류 헤드: ['wrinkle_grade', 'pigmentation_grade', 'pore_grade', 'dryness_grade', 'sagging_grade', 'skin_type', 'sensitive']
```

### 야간 학습 중 SSH 끊김 → 17h 뒤 PC 먹통 (2026-05-24~26 사고)

증상 순서:
1. SSH 세션 갑자기 끊김 (학습 시작 후 ~몇 시간)
2. 노트북에서 `ssh DS@100.118.240.67` → `Connection timed out` (refused 아님)
3. `tailscale status` 로 보니 lab PC `last seen 17h ago, tx 5304 rx 0` — Tailscale 데몬 사망
4. 졸프실 가서 확인 → PC 가 켜져있지만 입력 반응 없음 (먹통)

근본 원인 (추정 99%):
- Windows 가 학습 도중 절전 진입 시도 → GPU context 손상 → python 프로세스 좀비화
- 좀비 프로세스가 GPU 락 + RAM 잡고 안 죽음 → 시스템 전반 hang
- 그 와중에 Tailscale 데몬도 같이 죽음 (자동 재시작 미설정)

해결:
1. PC 재부팅 (먹통 상태에선 답이 이것뿐 — checkpoint 는 디스크에 안전)
2. **위 "안정성 보장" 블록 전부 적용** (powercfg + Tailscale/sshd 자동시작 + 매일 4시 재시작 보험)
3. `train_detach_resume.bat` 로 마지막 체크포인트부터 재개

교훈:
- "PC 가 켜져있다" ≠ "OS 가 정상 동작" — Tailscale `tx N rx 0` 패턴이 결정적 단서
- 100.x.x.x 대역으로 ping 100% 손실 → 십중팔구 Tailscale 데몬 죽음 (네트워크 케이블/공유기 문제 아님)
- `Connection timed out` (sshd 죽음) vs `Connection refused` (sshd 살아있지만 거부) 구별 중요

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

### v4 (2026-05-24~26, focal γ=2 — ❌ 실패)
- v3 동일 + **classification_loss: focal, focal_gamma: 2.0** (7개 헤드 일괄)
- 시도: cls_skin_type (1.5057) / cls_wrinkle_grade (1.1962) 개선 목표
- **결과**: epoch 12 val/loss/total best 0.6096 후 plateau, epoch 20 early_stop. **test set composite 0.5909 (v3 0.6461 대비 −0.0552)**
- **12개 헤드 중 9개 악화, 회귀 4개 악화. focal 1순위 타겟이었던 skin_type 이 가장 크게 악화 (F1 0.190 → 0.168, −11.6%)**
- 함정: val loss 0.6096 < v3 0.8234 라 학습 중엔 좋아 보였음. 그러나 focal 의 loss scale 이 CE 대비 ~25× 작은 게 원인이었고, 손실 무관 메트릭(MAE/F1) 으로 보면 일제히 regression. **다음부턴 손실 종류 바뀌면 val loss 비교 무효 — evaluate.py 의 test set metric 으로만 평가**
- 폐기: focal 일괄 적용. v5 는 CE 로 회귀 + scanner_aug + sensor_input
- 상세는 PROGRESS.md §4 v4 절 참고

### v5 (2026-05-27, scanner_aug + sensor_input + CE — ❌ 실패)
- v3 동일 + **augment_mode: scanner** (LowRes / Blur / ColorJitter 강화 / JPEG / Noise) + **sensor_inputs: [moisture]** + regression_targets 에서 moisture 제거
- **결과**: epoch 36 best (val 0.8895 plateau), epoch 44 early_stop. **test composite 0.5513 (v3 -0.0948, v4 보다도 약함)**
- **패턴 매우 명확**: 고주파 디테일 의존 헤드 큰 폭 악화 (pore_value MAE +38%, pigmentation_value +45%, wrinkle_grade F1 -25%), 거시적 정보 의존 헤드는 보존 (elasticity, pigmentation_grade, sensitive)
- **scanner_aug 의 LowResSimulate(low_range=(64,128), p=0.6) 가 주범 확정** — 다운→업샘플로 미세 텍스처 정보 파괴
- 다른 aug (Blur p=0.5 / JPEG p=0.7 / Noise p=0.6 / ColorJitter 강화) 도 기여. 한 이미지당 평균 ~2.4개 aug 적용 → 학습 데이터 망가짐
- sensor_input 의 효과는 단독 검증 안 됨 (ablation 안 함)
- v5.1 으로 aug 강도 약화 시도
- 상세는 PROGRESS.md §4 v5 절 참고

### v5.1 (2026-05-27 시작, scanner_aug 강도 ~40% 약화 — 진행 중)
- v5 와 동일하되 dataset.py 의 scanner aug 모든 강도/p 약화. LowResSimulate 가장 크게 약화 (p 0.6→0.3, low_range 64-128→128-192)
- 가설: 디테일 헤드 회복 + scanner robustness 일부 유지
- 종료 조건: composite ≥ 0.62 (v3 0.6461 ±5%) + pore_value MAE 회복
- 상세는 PROGRESS.md §4 v5.1 절 참고

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

- [x] **test set 평가 스크립트** (`src/evaluate.py`) — 2026-05-26 작성. MAE/RMSE/Accuracy/F1/Confusion + per-region 슬라이스 + v3/v4 diff 모드. v4 종료 직후 실행 예정
- [ ] **약한 분류 헤드** (v3 test F1 기준: dryness_grade 0.160, skin_type 0.190, pore_grade 0.188) — class_weights (v2) 와 focal γ=2 일괄 (v4) 둘 다 실패. v5 본 학습엔 포함 안 함. v5.5+ 후보: 약한 헤드 한정 focal γ=0.5 / 헤드별 lr 차등 / per-head trunk 분리 / 라벨 annotator agreement 검증. 자세한 우선순위는 PROGRESS.md §5 참고
- [ ] **체크포인트 디스크 정리** — 50 epoch × 295MB = 14.7GB. best(epoch045) + last(epoch050) 외 삭제 권장
- [ ] **PART_0 의 `acne` annotation** 빌더에서 누락. 별도 회귀/분류 헤드로 확장 가능
- [ ] **wrinkle 8개 거칠기 파라미터** 중 Ra 만 사용. Rmax/Rt 등 7개도 정보가 있음
- [ ] **`elasticity_Q0~Q3`** (점탄성 위상각) 빌더가 R0~R9 만 평균. Q 계열도 별도 헤드 검토 가능
- [ ] **부위별 학습률/손실 가중치 차등** — 부위가 한쪽으로 쏠리면 부위 임베딩이 그쪽으로 편향될 가능성
- [ ] **학습 프로세스 자동 재시작** — 현재 schtasks 는 1회 실행만. python OOM/CUDA crash 시 부활 안 됨. NSSM 으로 서비스화 또는 watcher 스크립트 검토 (v5 이후)
- [ ] **Wake-on-LAN** — PC 가 꺼진 뒤 원격으로 깨우기. 현재는 졸프실 물리방문 필수

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
  configs/baseline.yaml          ← 학습 설정 (lr, epochs, regression_targets 등)
  data/manifest.csv              ← 단일 manifest (build_manifest.py 결과, gitignored)
  src/
    build_manifest.py            ← AI-Hub JSON → manifest 통합
    dataset.py                   ← PyTorch Dataset + collate + class_weights + reg_stats
    model.py                     ← ResNet-50 + region embedding + 다중 헤드
    losses.py                    ← 마스크 회귀 + CE/focal CE + class_weights
    train.py                     ← 학습 루프 (+ --resume, --resume-from)
    evaluate.py                  ← held-out test set 평가 (v3/v4 diff 모드 포함)
    utils.py                     ← REGION_TO_ID, ID_TO_REGION, logger 등
    inspect_json.py              ← AI-Hub JSON 키 진단 (인자 없이 자동 탐색)
    dump_tb.py                   ← TB 이벤트 파일에서 per-head 손실 추출 + 비교
  train_detach.bat               ← 새 학습 (schtasks 기반, SSH/노트북 분리)
  train_detach_resume.bat        ← 끊긴 학습 이어가기 (schtasks 기반)
  _train_payload.bat             ← train_detach 가 호출하는 페이로드
  _train_payload_resume.bat      ← train_detach_resume 가 호출하는 페이로드
  train_main.bat / train_resume.bat  ← 같은 cmd 안 백그라운드 (간단 테스트용)
  runs/                          ← TensorBoard 로그 (main, baseline_val, main_v1, main_v2 ...)
  runs/eval/                     ← evaluate.py 결과 (JSON + MD + confusion CSV)
  checkpoints/                   ← 학습 중 epoch별 ckpt (gitignored)
  myvenv/                        ← 가상환경 (gitignored)
```

---

## 8. 시연 환경 — ESP32-CAM 스캐너 (Phase 2, 2026-06 중순 시연)

### 하드웨어 구성 (중간 발표 기록)

| 분류 | 부품 | 비고 |
|---|---|---|
| 메인 컨트롤러 | ESP32-CAM (AI Thinker) | Wi-Fi + OV2640 카메라 내장 |
| 카메라 모듈 | OV2640 | 실용 해상도 **640×480 ~ 800×600** (메모리/대역폭 한계) |
| 수분 센서 | FDC2112 (Moisture Click) | I²C 0x2A, 정전용량 방식 — AI-Hub `moisture` 와 원리 동일 |
| 조도 센서 | VEML7700 | I²C 0x10, 조도 + 자외선 간접 추정 |
| 조명 | 백색 LED ×3 + UV LED (395nm) ×1 | UV 는 색소반점 검출 보조 (의학적 원리) |
| 촬영 트리거 | 택트스위치 (IO2) | user trigger |
| 전원 | TP4056 + 3.7V LiPo + 안정화 커패시터 | |
| 통신 | I²C (IO14/IO15) + Wi-Fi 웹서버 | 고정 IP 10.105.206.100 |
| 펌웨어 | CH340 MB 보드 (USB 시리얼) | |

### AI-Hub ↔ ESP32-CAM 도메인 갭

| 차원 | AI-Hub 학습 | ESP32-CAM 시연 | 갭 | 대응 |
|---|---|---|---|---|
| 해상도 | 디카 수천만 px → 224×224 crop | 640×480 → 부위 crop ~100-200px | 🔴 매우 큼 | scanner_aug (저해상도 시뮬) |
| JPEG 압축 | 약함 (원본 RAW 가능) | 강함 (대역폭 절감) | 🔴 큼 | scanner_aug (JPEG quality 30~70) |
| 색 정확도 | DSLR 표준 | OV2640 저정확도 | 🟡 중 | ColorJitter 강화 |
| 노이즈 | 낮음 | 높음 (저조도시 더) | 🟡 중 | GaussianNoise aug |
| 조명 균일성 | 스튜디오 균일 | LED ×3 균일 | 🟢 작음 (오히려 유리) | 별도 대응 불필요 |
| 부위 식별 | JSON 메타 명시 | 스캐너는 모름 | 🔴 큼 | **UI 에서 user 가 부위 명시** (mediapipe over-engineering) |
| 센서 입력 | 없음 (image only) | FDC2112 + VEML7700 | 신규 차원 | model.py sensor_branch 활용 |

### 시연 시나리오 (3주 후)

```
사용자 UI 흐름:
  1. 부위 선택 ("이마 측정 시작")
  2. 스캐너 댐 → 택트스위치
  3. 백색 LED 점등 → 사진 1장 + FDC2112 수분값 + VEML7700 조도값 캡처
  4. (선택) UV LED 점등 → UV 사진 1장 추가 (색소반점 보조 시각화용, 모델 입력엔 X)
  5. Wi-Fi 로 노트북/PC 에 전송
  6. infer.py → 부위별 측정값 (수분/탄력/모공) + 등급 (주름/색소/모공 등) 반환
  7. UI 에 결과 표시
```

핵심: **부위 자동 인식 없음** — user 가 측정 전에 명시. 시연 안정성 ↑ + 구현 비용 ↓.

### Mitigation 전략 (시연 3주 작업 분할)

**Week 1 (현재)**
1. scanner-matched augmentation 파이프라인 — dataset.py `build_transforms(augment_mode='scanner')` 옵션
2. 센서 입력 학습 파이프라인 — model.py sensor_branch 활성화 (sensor_dim=2)
3. `src/infer.py` — 시연용 단일 추론 entry
4. (50건 데이터 도착 대기) — ESP32-CAM 실측 페어 데이터

**Week 2**
5. v5 본 학습 — scanner_aug ON + sensor_input ON + 헤드별 focal γ
6. 50건 ESP32-CAM 데이터로 **fine-tune** (마지막 5 epoch 만, lr ×0.1)
   - lab PC 학습 데이터: 100K AI-Hub + 50 ESP32-CAM (oversample 10× = 500)
   - 또는 stage-2: AI-Hub 학습 끝난 ckpt 를 50건 ESP32-CAM 으로 fine-tune

**Week 3**
7. Gradio UI + ESP32 Wi-Fi 연동 (HTTP polling 또는 WebSocket)
8. End-to-end 리허설 — 부위별 측정 흐름 5번 반복 안정성 확인

### 센서 활용 매핑

| 센서 | AI-Hub 매칭 컬럼 | 모델 활용 |
|---|---|---|
| FDC2112 (수분) | `moisture` (corneometer) | **회귀 타겟 → 입력 feature** ablation. 단위 캘리브레이션 필요 (TBD: 같은 사람 corneometer vs FDC2112 측정 비교) |
| VEML7700 (조도) | 없음 | 환경 메타 입력 — 모델이 광원 변화 보정에 활용 가능 |
| UV LED 395nm 이미지 | 없음 (AI-Hub 가시광만) | 시연 시 화면에 별도 표시 (의학적 시각화), 모델 입력 X |

### 50건 데이터 확보 계획

- 수일 내 ESP32-CAM 으로 50건 이상 페어 데이터 (image + 센서값) 수집 가능 (사용자 보고)
- 라벨: 전문가 등급은 불가능 (전문가 필요). 측정값 (FDC2112 수분) 만 ground truth
- 활용:
  - **검증 set**: 모델 이미지 예측 moisture vs FDC2112 실측 moisture 상관관계
  - **fine-tune set**: 회귀 헤드 (moisture) 만 한정 supervision, 분류는 self-distill 또는 fix
  - **augmentation 보정**: scanner_aug 가 실제 ESP32-CAM 분포를 얼마나 정확히 시뮬하는지 KL divergence 측정
