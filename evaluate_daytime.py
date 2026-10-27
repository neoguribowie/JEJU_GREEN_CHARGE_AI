
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. 기존 예측 결과 읽기
file_path = Path("results/solar_predictions.csv")

df = pd.read_csv(file_path)

# 2. 필요한 컬럼 확인
required = ["시간", "actual", "predicted"]

for col in required:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {col}")

# 3. 성능 평가 함수
def evaluate(data, name):
    if len(data) < 2:
        print(f"{name}: 평가 데이터 부족")
        return

    y_true = data["actual"]
    y_pred = data["predicted"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)

    return {
        "평가구간": name,
        "데이터수": len(data),
        "MAE_kWh": round(mae, 4),
        "RMSE_kWh": round(rmse, 4),
        "R2": round(r2, 4)
    }

# 4. 평가할 시간대 설정
periods = {
    "전체 시간": df,
    "09~17시": df[df["시간"].between(9, 17)],
    "10~16시": df[df["시간"].between(10, 16)],
    "11~15시": df[df["시간"].between(11, 15)]
}

# 5. 성능 계산
results = []

for name, subset in periods.items():
    result = evaluate(subset, name)
    if result is not None:
        results.append(result)

result_df = pd.DataFrame(results)

print("\n=== EXP-004 낮 시간대 성능 평가 ===")
print(result_df.to_string(index=False))

# 6. 결과 저장
Path("results").mkdir(exist_ok=True)

result_df.to_csv(
    "results/daytime_evaluation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n결과 저장 완료!")
