
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. 데이터 불러오기
file_path = Path("data/raw/sun/제주#1.csv")

for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError("CSV 인코딩을 확인하세요.")

# 2. 금악태양광 발전소 선택
df = df[df["발전기코드"] == "AJ85"].copy()

df["거래일"] = pd.to_datetime(df["거래일"])
df = df.sort_values(["거래일", "시간"])

# 3. 시간 관련 변수 생성
df["월"] = df["거래일"].dt.month

df["hour_sin"] = np.sin(
    2 * np.pi * (df["시간"] % 24) / 24
)
df["hour_cos"] = np.cos(
    2 * np.pi * (df["시간"] % 24) / 24
)
df["month_sin"] = np.sin(
    2 * np.pi * df["월"] / 12
)
df["month_cos"] = np.cos(
    2 * np.pi * df["월"] / 12
)

# 4. 기상변수
weather_features = [
    "기온", "강수량", "풍속", "풍향",
    "습도", "일조", "일사", "지면온도"
]

# 5. 모델별 입력변수
experiments = {
    "EXP-001": weather_features + ["월", "시간"],
    "EXP-002": weather_features,
    "EXP-003": weather_features + [
        "월", "시간",
        "hour_sin", "hour_cos",
        "month_sin", "month_cos"
    ]
}

# 6. 시간순 학습 / 테스트 분리
train = df[df["거래일"] < "2025-01-01"]
test = df[
    (df["거래일"] >= "2025-01-01") &
    (df["거래일"] < "2026-01-01")
]

if train.empty or test.empty:
    raise ValueError("학습 또는 테스트 데이터가 없습니다.")

print("학습 데이터:", len(train))
print("테스트 데이터:", len(test))

results = []

# 7. 세 모델을 같은 조건에서 학습
for name, features in experiments.items():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[features],
        train["계량전력량"]
    )

    y_true = test["계량전력량"]
    y_pred = model.predict(test[features])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)

    # 8. 일조시간 기준 주간 성능 평가
    # 일조 > 0은 주간 전체가 아니라
    # 직사광선이 관측된 시간의 근사 기준임
    day_mask = test["일조"] > 0

    if day_mask.sum() > 1:
        day_r2 = r2_score(
            y_true[day_mask],
            y_pred[day_mask]
        )
        day_mae = mean_absolute_error(
            y_true[day_mask],
            y_pred[day_mask]
        )
    else:
        day_r2 = np.nan
        day_mae = np.nan

    results.append({
        "실험": name,
        "MAE_kWh": round(mae, 4),
        "RMSE_kWh": round(rmse, 4),
        "R2": round(r2, 4),
        "일조시간_MAE_kWh": round(day_mae, 4),
        "일조시간_R2": round(day_r2, 4)
    })

# 9. 결과 출력 및 CSV 저장
result_df = pd.DataFrame(results)

print("\n=== 모델 성능 비교 ===")
print(result_df.to_string(index=False))

Path("results").mkdir(exist_ok=True)
result_df.to_csv(
    "results/model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n모델 비교 결과 저장 완료!")
