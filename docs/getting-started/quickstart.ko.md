
# 빠른 시작

5분 안에 infer-ci를 시작해 보세요.

## 설치

```bash
pip install infer-ci
```

## 첫 번째 평가

### 분류(Classification) 예제

```python
from infer_ci import MetricEvaluator
import numpy as np

# 실제 레이블과 예측 레이블 준비
y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

# 평가기 생성 및 평가 실행
evaluator = MetricEvaluator()
results = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="classification"
)

print(results)
```

### 회귀(Regression) 예제

```python
from infer_ci import MetricEvaluator
import numpy as np

# 실제 값과 예측값 준비
y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
y_pred = np.array([2.8, 5.2, 2.7, 6.5, 4.8])

# 회귀 모델 평가
evaluator = MetricEvaluator()
results = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="regression"
)

print(results)
```

## 직접 메서드 호출하기

평가기를 사용하지 않고 개별 메트릭을 직접 호출할 수도 있습니다.

### 분류 메트릭

```python
from infer_ci import accuracy, precision, recall, f1_score
import numpy as np

y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])

print(f"정확도: {accuracy(y_true, y_pred)}")
print(f"정밀도: {precision(y_true, y_pred)}")
print(f"재현율: {recall(y_true, y_pred)}")
print(f"F1 점수: {f1_score(y_true, y_pred)}")
```

### 회귀 메트릭

```python
from infer_ci import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.2, 2.7, 6.5])

print(f"평균 제곱 오차(MSE): {mean_squared_error(y_true, y_pred)}")
print(f"평균 절대 오차(MAE): {mean_absolute_error(y_true, y_pred)}")
print(f"결정 계수(R²): {r2_score(y_true, y_pred)}")
```

## 여러 메트릭 한 번에 계산하기

여러 메트릭을 지정해 한 번에 평가할 수 있습니다.

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

evaluator = MetricEvaluator()
results = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="classification",
    metrics=["accuracy", "precision", "recall", "f1"]
)

for metric, value in results.items():
    print(f"{metric}: {value}")
```

## 모델 간 신뢰구간 비교

infer-ci를 사용하면 평가 메트릭의 신뢰구간을 계산하고 모델 간 차이를 비교할 수 있습니다.

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
model_a = np.array([0, 1, 1, 0, 0, 0, 1, 1])
model_b = np.array([0, 1, 0, 0, 1, 0, 1, 0])

evaluator = MetricEvaluator()

results_a = evaluator.evaluate(
    y_true=y_true,
    y_pred=model_a,
    task="classification"
)

results_b = evaluator.evaluate(
    y_true=y_true,
    y_pred=model_b,
    task="classification"
)

print("모델 A:", results_a)
print("모델 B:", results_b)
```

> 참고: 실제로 사용할 수 있는 인자와 메서드는 설치된 infer-ci 버전에 따라 다를 수 있습니다. 자세한 내용은 API 레퍼런스를 확인하세요.

## 다중 클래스 분류

다중 클래스 분류 문제에서도 평가 메트릭을 계산할 수 있습니다.

```python
from infer_ci import MetricEvaluator
import numpy as np

y_true = np.array([0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1])

evaluator = MetricEvaluator()
results = evaluator.evaluate(
    y_true=y_true,
    y_pred=y_pred,
    task="classification"
)

print(results)
```

## 다음 단계

- [설치 안내](installation.ko.md)에서 설치 방법을 확인하세요.
- [API 레퍼런스](../api/index.md)에서 자세한 기능을 확인하세요.
- 프로젝트의 README에서 추가 사용 예제를 확인하세요.

## 문제 해결

### 설치 오류

다음 명령어로 패키지를 설치하거나 업데이트하세요.

```bash
pip install --upgrade infer-ci
```

### 도움이 필요한 경우

문제가 계속되면 GitHub 저장소의 Issues에 문의해 주세요.
