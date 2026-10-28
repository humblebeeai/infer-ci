# 빠른 시작

5분 안에 infer-ci를 설치하고 사용해 봅니다.

## 설치

```bash
pip install infer-ci
```

infer-ci는 Python 3.10 이상이 필요합니다. 가상환경 설정과 다른 설치 방법은 [설치](installation.ko.md) 문서를 참고하세요.

## 첫 번째 평가

`MetricEvaluator.evaluate()`는 정답값과 모델 예측값을 입력받아, 지표값과 그 신뢰구간(CI)을 함께 반환합니다.

### 입력값

| 인자 | 설명 |
|---|---|
| `y_true` | 정답값. 1차원 NumPy 배열 또는 리스트입니다. 이진 분류에서는 `0`/`1` 레이블을, 회귀에서는 숫자를 사용합니다. |
| `y_pred` | 모델 예측값. `y_true`와 같은 순서, 같은 길이여야 합니다. 분류에서는 확률이 아닌 예측 레이블을 사용합니다. |
| `task` | 문제 유형: `'classification'`, `'regression'`, `'detection'` 중 하나입니다. |
| `metric` | 계산할 지표. 예: `'accuracy'`, `'mae'`. [사용 가능한 지표](#available-metrics)를 참고하세요. |

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `method` | `'bootstrap_bca'` | 신뢰구간 계산 방법입니다. `'wilson'` 같은 분석적 방법은 실행할 때마다 같은 결과를 냅니다. Bootstrap 방법(`'bootstrap_*'`)은 데이터를 무작위로 다시 뽑습니다. [사용 가능한 신뢰구간 방법](#available-ci-methods)을 참고하세요. |
| `confidence_level` | `0.95` | 신뢰수준입니다. `0.95`이면 95% 신뢰구간을 계산합니다. |
| `n_resamples` | `9999` | Bootstrap 재표집 횟수입니다. Bootstrap 방법에서만 사용합니다. |
| `random_state` | `None` | Bootstrap 방법의 난수 시드입니다. 값을 지정하면(예: `random_state=42`) 실행할 때마다 같은 신뢰구간이 나옵니다. |
| `compute_ci` | `True` | `False`로 지정하면 지표값만 반환합니다. |

### 반환값

`compute_ci=True`(기본값)이면 `evaluate()`는 튜플을 반환합니다.

```python
value, (lower, upper) = evaluator.evaluate(...)
```

- `value`: 입력 데이터로 계산한 지표값
- `(lower, upper)`: 신뢰구간의 하한과 상한

아래 예제에서는 신뢰구간을 `ci`로 받아 `ci[0]`(하한)과 `ci[1]`(상한)로 읽습니다. `compute_ci=False`이면 `value`만 반환합니다.

### 분류 예제

```python
from infer_ci import MetricEvaluator
import numpy as np

# Sample binary classification data
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

# Initialize evaluator
evaluator = MetricEvaluator()

# Compute accuracy with Wilson interval
accuracy, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
)

print(f"Accuracy: {accuracy:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**출력:**
```
Accuracy: 0.900
95% CI: [0.596, 0.982]
```

Wilson 구간은 공식으로 계산하므로, 이 예제는 실행할 때마다 같은 결과를 출력합니다.

### 회귀 예제

```python
from infer_ci import MetricEvaluator
import numpy as np

# Sample regression data
y_true = np.array([3.0, -0.5, 2.0, 7.0, 5.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0, 4.5])

# Initialize evaluator
evaluator = MetricEvaluator()

# Compute MAE with bootstrap
mae, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='regression',
    metric='mae',
    method='bootstrap_bca',
    n_resamples=2000,
    random_state=42
)

print(f"MAE: {mae:.3f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**출력:**
```
MAE: 0.500
95% CI: [0.200, 0.800]
```

Bootstrap 방법은 데이터를 무작위로 다시 뽑습니다. `random_state=42`를 지정하면 실행할 때마다 같은 신뢰구간이 나옵니다. `random_state`를 지정하지 않으면 MAE 값은 같지만 신뢰구간은 실행마다 조금씩 달라질 수 있습니다.

Bootstrap이 실행되는 동안 `Bootstrap CI (BCA): 100%|...| 2006/2006` 같은 진행 표시줄도 함께 출력됩니다. BCa 방법은 구간을 보정하기 위한 계산을 추가로 수행하므로, 전체 횟수는 `n_resamples + 표본 수 + 1`(여기서는 2000 + 5 + 1)입니다.

## 다른 방법: 함수 직접 호출

지표 함수를 직접 호출할 수도 있습니다.

```python
from infer_ci import accuracy_score, mae
import numpy as np

# Classification
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 0])

acc, ci = accuracy_score(y_true, y_pred, method='wilson')
print(f"Accuracy: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

# Regression
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mae_val, ci = mae(y_true, y_pred)
print(f"MAE: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 자주 쓰는 작업 흐름

### 1. 여러 지표 평가하기

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()

metrics = ['accuracy', 'precision', 'recall', 'f1']

for metric in metrics:
    value, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric=metric,
        method='wilson'
    )
    print(f"{metric.capitalize():12s}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**출력:**
```
Accuracy    : 0.900 [0.597, 0.983]
Precision   : 0.857 [0.485, 0.981]
Recall      : 1.000 [0.661, 1.000]
F1          : 0.923 [0.640, 0.989]
```

### 2. 신뢰구간 방법 비교하기

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

methods = ['wilson', 'normal', 'bootstrap_bca']

for method in methods:
    acc, ci = accuracy_score(y_true, y_pred, method=method, n_resamples=2000)
    width = ci[1] - ci[0]
    print(f"{method:15s}: [{ci[0]:.3f}, {ci[1]:.3f}] (width: {width:.3f})")
```

**출력:**
```
wilson         : [0.519, 0.938] (width: 0.419)
normal         : [0.596, 1.004] (width: 0.408)
bootstrap_bca  : [0.533, 0.933] (width: 0.400)
```

### 3. 다중 클래스 분류

```python
from infer_ci import classification_report_with_ci
import numpy as np

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)
```

**출력:**
```
     Class  Precision  Recall  F1-Score    Precision CI       Recall CI     F1-Score CI  Support
0  Class 0      0.600   1.000     0.750  (0.231, 0.882)    (0.439, 1.0)  (0.408, 1.092)        3
1  Class 1      0.889   1.000     0.941   (0.565, 0.98)    (0.676, 1.0)  (0.796, 1.086)        8
2  Class 2      1.000   0.667     0.800     (0.61, 1.0)  (0.354, 0.879)  (0.562, 1.038)        9
3    micro      0.850   0.850     0.850  (0.694, 1.006)  (0.694, 1.006)  (0.694, 1.006)       20
4    macro      0.830   0.889     0.830  (0.702, 0.958)  (0.775, 1.002)  (0.548, 1.113)       20
```

## 사용 가능한 지표 {#available-metrics}

### 분류 지표

```python
from infer_ci import MetricEvaluator

evaluator = MetricEvaluator()
print(evaluator.get_available_metrics('classification'))
```

- `accuracy` - 정확도
- `precision` - 정밀도(양성 예측도, Positive Predictive Value)
- `recall` - 재현율(참양성률, True Positive Rate)
- `f1` - F1 점수
- `specificity` - 특이도(참음성률, True Negative Rate)
- `npv` - 음성 예측도(Negative Predictive Value)
- `fpr` - 위양성률(False Positive Rate)
- `auc` - ROC AUC 점수

### 회귀 지표

```python
print(evaluator.get_available_metrics('regression'))
```

- `mae` - 평균 절대 오차(Mean Absolute Error)
- `mse` - 평균 제곱 오차(Mean Squared Error)
- `rmse` - 평균 제곱근 오차(Root Mean Squared Error)
- `r2` - 결정계수(R² Score)
- `mape` - 평균 절대 백분율 오차(Mean Absolute Percentage Error)
- `iou` - IoU(Intersection over Union)

### 객체 탐지 지표

```python
print(evaluator.get_available_metrics('detection'))
```

- `map` - mAP @0.5:0.95(mean Average Precision)
- `map50` - mAP @0.5
- `precision` - 탐지 정밀도
- `recall` - 탐지 재현율

## 사용 가능한 신뢰구간 방법 {#available-ci-methods}

### 분류

```python
print(evaluator.get_available_methods('classification'))
```

**분석적 방법:**
- `wilson` - Wilson 점수 구간(권장)
- `normal` - 정규 근사
- `agresti_coull` - Agresti-Coull 구간
- `beta` - Clopper-Pearson 구간
- `jeffreys` - Jeffreys 구간

**Bootstrap 방법:**
- `bootstrap_bca` - BCa(권장)
- `bootstrap_percentile` - 백분위수(Percentile)
- `bootstrap_basic` - 기본(Basic)

**기타:**
- `jackknife` - Jackknife 재표집
- `takahashi` - F1/정밀도/재현율용(다중 클래스)

### 회귀

```python
print(evaluator.get_available_methods('regression'))
```

- `bootstrap_bca` - Bootstrap BCa(기본값)
- `bootstrap_percentile` - Bootstrap 백분위수
- `jackknife` - Jackknife 재표집

## 실전 예제

### scikit-learn 모델 평가하기

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from infer_ci import MetricEvaluator
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Binary classification (class 0 vs rest)
y_binary = (y == 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate with confidence intervals
evaluator = MetricEvaluator()

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    value, ci = evaluator.evaluate(
        y_true=y_test,
        y_pred=y_pred,
        task='classification',
        metric=metric,
        method='wilson'
    )
    print(f"{metric.capitalize():12s}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 다음 단계

- [예제](examples.md)
- [객체 탐지 가이드](../object-detection-metrics-usage.md)
- [MetricEvaluator API](../api-docs/MetricEvaluator.md)
- [API 레퍼런스](../api-docs/index.md)

## 공통 매개변수

대부분의 함수는 다음 매개변수를 받습니다.

- `method` - 신뢰구간 계산 방법(기본값: 'bootstrap_bca')
- `confidence_interval` - 신뢰수준(기본값: 0.95)
- `n_resamples` - Bootstrap 재표집 횟수(기본값: 2000)
- `random_state` - 재현성을 위한 난수 시드
- `compute_ci` - 신뢰구간 계산 여부(기본값: True)
- `plot` - 시각화 생성(기본값: False, Bootstrap 전용)

## 팁

1. 이진 분류 지표에는 **Wilson 방법**을 권장합니다.
2. **Bootstrap BCa**는 기본값이며 대부분의 경우에 잘 맞습니다.
3. 결과를 재현하려면 `random_state`를 사용하세요.
4. 더 안정적인 신뢰구간이 필요하면 `n_resamples`를 늘리세요(예: 5000).
5. Bootstrap 방법에서 `plot=True`를 사용하면 분포를 시각화할 수 있습니다.

## 도움 받기

- 📖 [전체 문서](https://infer.humblebee.ai)
- 💬 [GitHub Discussions](https://github.com/humblebeeai/infer-ci/discussions)
- 🐛 [이슈 제보](https://github.com/humblebeeai/infer-ci/issues)
