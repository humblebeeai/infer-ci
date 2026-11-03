
[English](README.md) | [한국어](README.ko.md)

# 평가 지표를 위한 신뢰구간 (Confidence Intervals for evaluation metrics)
![logo](docs/images/infer.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

이 도구는 Accuracy, MSE와 같은 일반적인 머신러닝 평가 지표를 계산하고, 부트스트래핑(bootstrapping) 등의 방법을 사용하여 해당 지표의 신뢰구간(Confidence Intervals)을 반환하는 표준화된 평가 도구입니다.

## 신뢰할 수 있는 AI 모델 평가를 위해 신뢰구간이 왜 필요한가요?

실무에서 팀들은 작거나 계속 변하는 테스트 세트에서 계산된 단일 숫자 지표를 바탕으로 모델을 배포하거나 반려하는 경우가 많습니다. 지표에 노이즈가 있을 때 데이터, 랜덤 시드 또는 샘플링의 아주 작은 변화만으로도 실제가 아닌 "성능 저하(regressions)"나 "획기적 발전(breakthroughs)"처럼 보일 수 있습니다. 이는 불필요한 반복 작업을 유발하고, 이해관계자와의 신뢰를 무너뜨리며, 불안정한 배포 결정을 초래합니다.

실제 사례: 몇 년 전 HumbleBee.AI를 창립하기 전에, Jumabek Alikhan의 동료가 독일 고객을 위해 YOLO 기반 탐지기를 구축하고 있었습니다. 작은 테스트 세트에서의 초기 평가에서는 mAP가 약 40%로 보고되었습니다. 하지만 데이터를 더 수집하고 훈련/테스트 세트를 확장한 후 보고된 mAP가 약 36%로 떨어졌고, 고객은 모델 성능이 저하되었다고 판단했습니다. Jumabek이 이 문제를 조사하기 위해 투입되었습니다. 문제는 YOLO의 "버그"가 아니라 통계적 불확실성이었습니다. 원래의 40%는 매우 넓은 95% 신뢰구간(약 20%~60%)을 가진 작은 테스트 세트에서 나온 반면, 더 큰 테스트 세트에서의 36%는 훨씬 더 좁고 신뢰할 수 있는 구간(약 30%~42%)을 가지고 있었습니다. 이러한 불확실성을 소통함으로써 분쟁이 해결되었고, 팀은 모델의 실제 성능 궤적에 대해 의견을 일치시킬 수 있었습니다.

이러한 사고방식, 즉 평가를 단일 점수가 아닌 통계적 추론으로 다루는 것은 훗날 Jumabek이 HumbleBee.AI를 설립하는 방향성을 제시했습니다. 낙관적인 대시보드 수치가 아닌 근거를 설명할 수 있는 신뢰성 있는 측정을 통해 프로덕션 ML을 제공하는 것입니다. `infer-ci`는 이러한 원칙을 오픈소스로 구현한 것으로, 어떤 팀이든 이미 보고하고 있는 지표에 신뢰구간을 첨부하고 안정적인 증거를 바탕으로 배포 결정을 내릴 수 있도록 돕습니다.

## 개발 동기 (The motivation)

성능 추정치를 제공할 때 신뢰구간(Confidence Intervals)을 계산하는 것은 그동안 충분히 연구되지 않았습니다. 수년에 걸쳐 연구자들은 의학 통계, AutoML, 텍스트 분류 등 다양한 분야에서 여러 CI 추정 방법을 제안해 왔습니다. 각 방법은 특정한 가정, 장점 및 한계를 가지고 있습니다. 그러나 대부분의 접근 방식은 단일 지표나 좁은 범위의 모델 클래스에 맞춰 설계되어 있어 적용 범위가 제한적입니다.

우리는 다양한 CI 계산 방법(부트스트래핑, Wald, Wilson 등)을 포함하여 모델 성능 지표와 신뢰구간을 함께 계산하고 명확한 시각화 그래프를 생성하는 표준화된 평가 도구를 개발했습니다. 이 도구는 ML, CV, LLM을 모두 지원합니다.


## 설치 (Installation)

### PyPI를 통한 설치 (권장)

```bash
pip install infer-ci

```

### 소스 코드로부터 설치 (개발용)

```bash
git clone https://github.com/humblebeeai/infer-ci.git
cd infer-ci
pip install -e ".[dev]"

```

## 시작하기

```python
# All the possible imports:
from infer_ci import MetricEvaluator
from infer_ci import accuracy_score, precision_score, tpr_score, f1_score
from infer_ci import mae, mse, rmse, r2_score, iou

# Classification example:
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

evaluator = MetricEvaluator()

accuracy, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
)
```

## 신뢰구간 계산 방법과 기본값

`MetricEvaluator.evaluate()`의 기본 방법은 `method='bootstrap_bca'`입니다. BCa 부트스트랩은 추정값의 편향과 비대칭성을 보정합니다. 지표 함수를 직접 호출하면 함수별 기본값이 적용됩니다. 예를 들어 `accuracy_score()`는 Wilson, `f1_score()`는 Takahashi 방법을 사용합니다. 모든 지표가 모든 신뢰구간 계산 방법을 지원하는 것은 아닙니다.

선택한 지표가 지원하는 방법을 지정하세요. 예를 들어 정확도에는 `method='wilson'`, 부트스트랩 재표집에는 `method='bootstrap_percentile'`을 사용할 수 있습니다. 회귀 지표는 `method='jackknife'`도 지원합니다. 같은 데이터와 라이브러리 버전에서 부트스트랩 결과를 재현하려면 `random_state`를 지정하세요.

## 지원되는 방법 (Supported Methods)

* **분석적 방법 (Analytical Methods)**: Wilson, Normal, Agresti-Coull, Beta, Jeffreys, Binomial test
* **부트스트랩 방법 (Bootstrap Methods)**: Bootstrap Basic, Bootstrap Percentile, Bootstrap Bias-Corrected and Accelerated (기본값)
* **잭나이프 방법 (Jackknife Method)**: Jackknife

**MetricEvaluator를 통해 사용 가능한 핵심 지표:**

**분류 (Classification):**

* `accuracy`: 전체 분류 정확도 (Overall classification accuracy)
* `precision`: 정밀도·양성 예측도 (Positive Predictive Value, PPV)
* `recall`: 재현율·민감도 (True Positive Rate/Sensitivity, TPR)
* `specificity`: 특이도 / 참 음성 비율 (True Negative Rate, TNR)
* `npv`: 음성 예측도 (Negative Predictive Value)
* `fpr`: 위양성률 (False Positive Rate)
* `f1`: F1 점수 (평균화 지원)
* `precision_takahashi`: Takahashi 방법을 사용한 정밀도
* `recall_takahashi`: Takahashi 방법을 사용한 재현율
* `auc`: ROC AUC 점수

**회귀 (Regression):**

* `mae`: 평균 절대 오차 (Mean Absolute Error)
* `mse`: 평균 제곱 오차 (Mean Squared Error)
* `rmse`: 평균 제곱 오차의 제곱근 (Root Mean Squared Error)
* `r2`: 결정 계수 (Coefficient of Determination)
* `mape`: 평균 절대 백분율 오차 (Mean Absolute Percentage Error)
* `iou`: IoU (Intersection Over Union)

**객체 탐지 (Object Detection - 전체 및 클래스별 지원):**

* `map`: mean Average Precision 0.5:0.95
* `map50`: mean Average Precision 0.5
* `precision`: 정밀도 (Precision)
* `recall`: 재현율 (Recall)

📖 **객체 탐지 지표 사용 예제는 다음을 참조하세요:** [Object Detection Metrics Usage Guide](docs/object-detection-metrics-usage.md)

📚 **온라인 버전 체험하기:** [https://infer.humblebee.ai](https://infer.humblebee.ai)

사용 가능한 방법과 지표를 코드로 확인할 수도 있습니다. 반환 목록은 작업 유형별 요약이며, 개별 지표마다 지원 범위가 다를 수 있습니다.

```python
from infer_ci import MetricEvaluator

evaluator = MetricEvaluator()
for task in ['classification', 'regression', 'detection']:
    print(f"{task} methods:", evaluator.get_available_methods(task))
    print(f"{task} metrics:", evaluator.get_available_metrics(task))
```

여기서 사용하는 분석적 계산은 아래 참고 문헌에 있는 Takahashi 등의 2022년 논문을 바탕으로 합니다. 해당 논문은 micro 평균의 재현율과 정밀도를 유도합니다. 이 프로젝트는 델타 방법을 사용해 macro F1 등의 신뢰구간 계산도 구현합니다.

## 분류 보고서 만들기

`classification_report_with_ci`는 주요 분류 지표와 신뢰구간을 포함하는 텍스트 보고서를 만듭니다. 먼저 각 클래스를 이진 분류 문제로 취급합니다. 클래스별 정밀도와 재현율의 기본 신뢰구간 계산 방법은 Wilson이고, F1은 이진 Takahashi 방법을 사용합니다. 이후 다중 클래스의 micro·macro 지표를 Takahashi 방법으로 계산합니다.

```python
from infer_ci import classification_report_with_ci

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)
```

출력 예시:

```text
     Class  Precision  Recall  F1-Score    Precision CI       Recall CI     F1-Score CI  Support
0  Class 0      0.600   1.000     0.750  (0.231, 0.882)    (0.439, 1.0)  (0.408, 1.092)        3
1  Class 1      0.889   1.000     0.941   (0.565, 0.98)    (0.676, 1.0)  (0.796, 1.086)        8
2  Class 2      1.000   0.667     0.800     (0.61, 1.0)  (0.354, 0.879)  (0.562, 1.038)        9
3    micro      0.850   0.850     0.850  (0.694, 1.006)  (0.694, 1.006)  (0.694, 1.006)       20
4    macro      0.830   0.889     0.830  (0.702, 0.958)  (0.775, 1.002)  (0.548, 1.113)       20
```

## F1의 binary·macro·micro 평균

`average='binary'`에는 이진 레이블을 사용하세요. 정밀도와 재현율도 평균 방식을 지원하며, 선택한 지표에서 지원하는 계산 방법을 확인해야 합니다.

```python
from infer_ci import f1_score

# Synthetic data for demonstrating the averaging API.
y_true = [0, 1, 1, 0, 1, 0, 1, 0] * 10
y_pred = [0, 1, 0, 0, 1, 1, 1, 0] * 10

for average in ['binary', 'macro', 'micro']:
    value, ci = f1_score(
        y_true, y_pred, confidence_level=0.95,
        average=average, method='takahashi'
    )
    print(f"{average}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

bootstrap_binary_f1, ci = f1_score(
    y_true, y_pred, confidence_level=0.95, average='binary',
    method='bootstrap_bca', n_resamples=5000, random_state=42
)
```

## 스크립트 (Scripts)

이 문서는 본 프로젝트에 포함된 다음 스크립트들에 대한 개요와 사용 방법을 제공합니다:

* **[`clean.sh`](docs/dev/scripts/clean.sh.md)**
* **[`get-version.sh`](docs/dev/scripts/get-version.sh.md)**
* **[`test.sh`](docs/dev/scripts/test.sh.md)**
* **[`bump-version.sh`](docs/dev/scripts/bump-version.sh.md)**
* **[`build.sh`](docs/dev/scripts/build.sh.md)**
* **[`release.sh`](docs/dev/scripts/release.sh.md)**
* **[`changelog.sh`](docs/dev/scripts/changelog.sh.md)**
* **[`diagrams.sh`](docs/dev/scripts/diagrams.sh.md)**
* **[`docs.sh`](docs/dev/scripts/docs.sh.md)**

모든 스크립트는 **`scripts`** 디렉터리에 위치해 있습니다:

```txt
scripts/
├── build.sh
├── bump-version.sh
├── changelog.sh
├── clean.sh
├── diagrams.sh
├── docs.sh
├── get-version.sh
├── release.sh
└── test.sh

```

이 스크립트들은 Linux 또는 macOS 환경에서 사용하도록 설계되었습니다. Windows 환경에서도 적절한 도구를 설치하면 작동할 수 있지만, 정상적인 동작을 완벽히 보장하지는 않습니다.

---

## 인용 (Citation)

연구에 이 도구를 사용하신다면 아래와 같이 인용해 주세요. 다음은 BibTeX 항목 예시입니다:

```
@misc{infer-ci,
  title={Confidence intervals for evaluation metrics},
  author={Humblebee ai},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/humblebeeai/infer-ci}},
}
```

---

## 참고 문헌 (References)

신뢰구간을 위한 Python 라이브러리:
https://github.com/jacobgil/confidenceinterval

AutoML 환경에서의 예측 성능 신뢰구간 추정:
https://arxiv.org/abs/2406.08099v1

이항 신뢰구간 계산은 statsmodels 패키지를 사용합니다:
https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html

Yandex data school의 빠른 delong 알고리즘 구현:
https://github.com/yandexdataschool/roc_comparison

상관된 ROC 곡선 아래 면적 비교를 위한 DeLong 알고리즘의 빠른 구현 (X. Sun and W. Xu):
https://ieeexplore.ieee.org/document/6851192

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP2

마이크로 평균 F1 및 매크로 평균 F1 점수에 대한 신뢰구간
`Kanae Takahashi,1,2 Kouji Yamamoto,3 Aya Kuchiba,4,5 and Tatsuki Koyama6`

부트스트랩 소개 (B. Efron and R. J. Tibshirani):
An Introduction to the Bootstrap, Chapman & Hall/CRC, Boca Raton, FL, USA (1993)

http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
`Nathaniel E. Helwig, “Bootstrap Confidence Intervals”`

부트스트래핑(통계학), 위키백과:
https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29
