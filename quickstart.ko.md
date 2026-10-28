
# 빠른 시작

5분 안에 infer-ci를 시작해 보세요.

## 설치

pip를 사용하여 infer-ci를 설치합니다.

```bash
pip install infer-ci
```

## 첫 번째 평가

### 분류(Classification) 예제

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()

accuracy, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='classification',
    metric='accuracy',
    method='wilson'
)

print(f"정확도: {accuracy:.3f}")
print(f"95% 신뢰구간: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### 회귀(Regression) 예제

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([3.0, -0.5, 2.0, 7.0, 5.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0, 4.5])

evaluator = MetricEvaluator()

mae, ci = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task='regression',
    metric='mae',
    method='bootstrap_bca',
    n_resamples=2000
)

print(f"평균 절대 오차(MAE): {mae:.3f}")
print(f"95% 신뢰구간: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 함수 직접 호출하기

```python
from infer_ci import accuracy_score, mae
import numpy as np

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 1, 0, 0])

acc, ci = accuracy_score(y_true, y_pred, method='wilson')
print(f"정확도: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

mae_val, ci = mae(y_true, y_pred)
print(f"MAE: {mae_val:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 여러 지표 평가하기

하나의 모델에 대해 여러 분류 지표를 평가할 수 있습니다.

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    value, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task='classification',
        metric=metric,
        method='wilson'
    )
    print(f"{metric}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 신뢰구간 계산 방법 비교하기

```python
from infer_ci import accuracy_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

methods = ['wilson', 'normal', 'bootstrap_bca']

for method in methods:
    acc, ci = accuracy_score(
        y_true, y_pred, method=method, n_resamples=2000
    )
    width = ci[1] - ci[0]
    print(f"{method:15s}: [{ci[0]:.3f}, {ci[1]:.3f}] (너비: {width:.3f})")
```

## 다중 클래스 분류

```python
from infer_ci import classification_report_with_ci

y_true = [0, 1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1]
y_pred = [0, 1, 0, 0, 2, 1, 1, 1, 0, 2, 2, 1, 0, 1, 2, 1, 2, 2, 1, 1]

classification_report_with_ci(y_true, y_pred)
```

## 사용 가능한 지표

### 분류 지표

- `accuracy` - 정확도
- `precision` - 정밀도
- `recall` - 재현율
- `f1` - F1 점수
- `specificity` - 특이도
- `npv` - 음성 예측도
- `fpr` - 거짓 양성률
- `auc` - ROC 곡선 아래 면적

### 회귀 지표

- `mae` - 평균 절대 오차
- `mse` - 평균 제곱 오차
- `rmse` - 평균 제곱근 오차
- `r2` - 결정계수
- `mape` - 평균 절대 백분율 오차
- `iou` - 교집합 대비 합집합 비율

### 객체 탐지 지표

- `map` - IoU 임계값 0.5~0.95에서의 평균 정밀도
- `map50` - IoU 0.5에서의 평균 정밀도
- `precision` - 탐지 정밀도
- `recall` - 탐지 재현율

## 사용 가능한 신뢰구간 계산 방법

### 분류

- `wilson` - Wilson 점수 구간
- `normal` - 정규 근사 방법
- `agresti_coull` - Agresti-Coull 구간
- `beta` - Clopper-Pearson 구간
- `jeffreys` - Jeffreys 구간
- `bootstrap_bca` - 편향 보정 및 가속 부트스트랩
- `bootstrap_percentile` - 백분위수 부트스트랩
- `bootstrap_basic` - 기본 부트스트랩 구간
- `jackknife` - 잭나이프 재표본추출
- `takahashi` - 다중 클래스 정밀도, 재현율 및 F1을 위한 방법

### 회귀

- `bootstrap_bca` - BCa 부트스트랩 구간
- `bootstrap_percentile` - 백분위수 부트스트랩 구간
- `jackknife` - 잭나이프 재표본추출

## 실제 예제

### Scikit-learn 모델 평가하기

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from infer_ci import MetricEvaluator

iris = load_iris()
X, y = iris.data, iris.target

# 이진 분류: 클래스 0과 나머지 클래스 구분
y_binary = (y == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

evaluator = MetricEvaluator()

for metric in ['accuracy', 'precision', 'recall', 'f1']:
    value, ci = evaluator.evaluate(
        y_true=y_test,
        y_pred=y_pred,
        task='classification',
        metric=metric,
        method='wilson'
    )
    print(f"{metric}: {value:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

## 다음 단계

- [분류 지표 가이드](classification.md)
- [회귀 지표 가이드](regression.md)
- [객체 탐지 가이드](../object-detection-metrics-usage.md)
- [신뢰구간 계산 방법](ci-methods.md)
- [API 레퍼런스](../api-docs/index.md)

## 주요 매개변수

- `method` - 신뢰구간 계산 방법
- `confidence_interval` - 신뢰 수준 (기본값: `0.95`)
- `n_resamples` - 부트스트랩 반복 횟수 (기본값: `2000`)
- `random_state` - 재현 가능한 결과를 위한 난수 시드
- `compute_ci` - 신뢰구간 계산 여부 (기본값: `True`)
- `plot` - 시각화 생성 여부 (기본값: `False`)

## 사용 팁

1. 이진 분류 지표에는 `wilson` 방법을 사용하세요.
2. 부트스트랩 신뢰구간에는 `bootstrap_bca`를 사용하세요.
3. 결과를 재현하려면 `random_state`를 설정하세요.
4. 부트스트랩 추정값을 더 안정적으로 얻으려면 `n_resamples`를 늘리세요.
5. 지원되는 부트스트랩 방법에서 `plot=True`를 사용하면 분포를 시각화할 수 있습니다.

## 도움말

- [전체 문서](https://infer.humblebee.ai)
- [GitHub 토론](https://github.com/humblebeeai/infer-ci/discussions)
- [문제 신고](https://github.com/humblebeeai/infer-ci/issues)
