# ChokePoint Face Attendance Evaluation (`P2E_S1_C3`)

이 디렉터리는 ChokePoint 데이터셋(`P2E_S1_C3` 시퀀스)을 대상으로 얼굴 인식 기반 출석 모델의 예측 성능을 평가하고, 95% 신뢰구간(Confidence Interval)을 산출하는 예제를 포함합니다.

## 1. 설치할 것 (Prerequisites)

- **Python:** 3.10 이상
- **패키지 관리 도구:** `uv` (또는 `pip`)
- **필수 파이썬 패키지:** `pandas`, `numpy`, `matplotlib`, `ipykernel`

```bash
uv venv --python 3.10
uv pip install pandas numpy matplotlib ipykernel