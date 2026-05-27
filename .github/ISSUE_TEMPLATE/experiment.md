---
name: 실험 (Experiment)
about: 새 학습 버전 / ablation / 가설 검증
title: 'experiment: '
labels: ['experiment']
assignees: ''
---

## 배경 / 동기

<!-- 이전 버전 결과 / 사고 / 발견에서 출발한 가설 -->

## 가설

<!-- 무엇을 검증할 것인가. 정량 가능하게 -->

## 변경 / 작업 범위

- `configs/...`
- `src/...`

## 예상 결과

| 지표 | baseline | 목표 |
|---|---:|---:|
|  |  |  |

## 종료 조건 (성공/실패 판정)

- ✅ 성공: ...
- ⚠️ 부분: ...
- ❌ 실패: ...

## 작업 체크리스트

- [ ] config 변경
- [ ] sanity check (`--validation-mode`)
- [ ] 본 학습
- [ ] evaluate.py 평가
- [ ] PROGRESS.md 결과 절 추가

## 의존성

- 🟢 / 🟡 / 🔴 ...

## 참고

- `PROGRESS.md` §N
- `NOTES.md` §N
- 관련 PR #N
