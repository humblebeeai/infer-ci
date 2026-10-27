```python
#!/usr/bin/env python
"""
이진 분류 신뢰구간 평가 예제

infer-ci 패키지를 사용하여 분류 평가 지표와 신뢰구간을 계산합니다.
"""

import numpy as np
from infer_ci import MetricEvaluator, accuracy_score


def main():
    print("신뢰구간을 활용한 이진 분류 평가")
    print("=" * 50)

    # 실제 정답과 모델의 예측 결과 예시
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

    print(f"\n데이터 개수: {len(y_true)}개")
    print(f"실제 정답: {y_true}")
    print(f"모델 예측: {y_pred}")

    # 방법 1: MetricEvaluator 사용
    print("\n" + "=" * 50)
    print("방법 1: MetricEvaluator를 사용한 평가")
    print("=" * 50)

    evaluator = MetricEvaluator()

    # Wilson 방법으로 정확도와 신뢰구간 계산
    acc, ci = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task="classification",
        metric="accuracy",
        method="wilson"
    )

    print(f"\n정확도: {acc:.3f}")
    print(f"95% 신뢰구간 (Wilson): [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Bootstrap BCA 방법으로 F1 점수와 신뢰구간 계산
    f1, ci_f1 = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        task="classification",
        metric="f1",
        method="bootstrap_bca",
        n_resamples=1000
    )

    print(f"\nF1 점수: {f1:.3f}")
    print(f"95% 신뢰구간 (Bootstrap BCA): "
          f"[{ci_f1[0]:.3f}, {ci_f1[1]:.3f}]")

    # 방법 2: 정확도 계산 함수 직접 사용
    print("\n" + "=" * 50)
    print("방법 2: 평가 함수 직접 호출")
    print("=" * 50)

    acc2, ci2 = accuracy_score(y_true, y_pred, method="wilson")

    print(f"\n정확도: {acc2:.3f}")
    print(f"95% 신뢰구간: [{ci2[0]:.3f}, {ci2[1]:.3f}]")

    # 여러 평가 지표를 한 번에 계산
    print("\n" + "=" * 50)
    print("여러 평가 지표 비교")
    print("=" * 50)

    metrics = ["accuracy", "precision", "recall", "f1"]

    metric_names = {
        "accuracy": "정확도",
        "precision": "정밀도",
        "recall": "재현율",
        "f1": "F1 점수"
    }

    for metric in metrics:
        value, ci = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            task="classification",
            metric=metric,
            method="wilson"
        )

        print(
            f"{metric_names[metric]:8s}: {value:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        )

    print("\n신뢰구간은 평가 지표의 불확실성을 이해하는 데 도움을 줍니다.")
    print("평가가 완료되었습니다!")


if __name__ == "__main__":
    main()
```
