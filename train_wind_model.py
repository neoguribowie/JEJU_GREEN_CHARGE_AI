
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# =============================
# 1. 파일과 결과 폴더 설정
# =============================
DATA_PATH = Path("data/processed/wind_weather_2020_2023.csv")
RESULT_DIR = Path("results/wind")
MODEL_DIR = Path("models")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["trade_date"] = pd.to_datetime(df["trade_date"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(["trade_date", "trade_hour"]).reset_index(drop=True)

# =============================
# 2. 입력변수 생성
# =============================

# 시간·월의 주기적 특징
df["hour_sin"] = np.sin(2 * np.pi * df["trade_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["trade_hour"] / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

features = ["hour_sin", "hour_cos", "month_sin", "month_cos"]

stations = ["jeju", "gosan", "seongsan", "seogwipo"]

for station in stations:
    # 풍향 360°와 0°는 같은 방향이므로 Sin/Cos로 변환
    direction = pd.to_numeric(
        df[f"{station}_wind_direction"], errors="coerce"
    )

    radians = np.deg2rad(direction)
    df[f"{station}_direction_sin"] = np.sin(radians)
    df[f"{station}_direction_cos"] = np.cos(radians)

    features += [
        f"{station}_wind_speed",
        f"{station}_temperature",
        f"{station}_humidity",
        f"{station}_pressure",
        f"{station}_direction_sin",
        f"{station}_direction_cos",
    ]

# 숫자형 변환 및 무한대 처리
for col in features + ["wind_mwh"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

# 원래 발전량 데이터는 결측이 없어야 함
assert df["wind_mwh"].notna().all()
assert set(df["year"].unique()) == {2020, 2021, 2022, 2023}

# =============================
# 3. 공통 평가 함수
# =============================
def calculate_metrics(actual, predicted):
    return {
        "MAE_MWh": mean_absolute_error(actual, predicted),
        "RMSE_MWh": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
    }

# =============================
# 4. 두 실험
# =============================
experiments = {
    "EXP-W001": [2021, 2022],
    "EXP-W002": [2020, 2021, 2022],
}

all_metrics = []
all_monthly = []
all_predictions = []

for name, train_years in experiments.items():
    print(f"\n===== {name} =====")
    print("학습 연도:", train_years)
    print("평가 연도: 2023")

    train = df[df["year"].isin(train_years)].copy()
    test = df[df["year"] == 2023].copy()

    X_train = train[features]
    y_train = train["wind_mwh"]

    X_test = test[features]
    y_test = test["wind_mwh"]

    # 동일 설정 사용
    # 결측치 중앙값은 학습 데이터에서만 계산됨
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=1,
        )),
    ])

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    # 연간 평가
    metrics = calculate_metrics(y_test, predicted)

    all_metrics.append({
        "experiment": name,
        "train_years": ",".join(map(str, train_years)),
        "test_year": 2023,
        "train_rows": len(train),
        "test_rows": len(test),
        **metrics,
    })

    print("학습 행 수:", len(train))
    print("평가 행 수:", len(test))
    print(f"MAE  : {metrics['MAE_MWh']:.4f} MWh")
    print(f"RMSE : {metrics['RMSE_MWh']:.4f} MWh")
    print(f"R²   : {metrics['R2']:.4f}")

    # 예측값 저장용
    pred_df = test[
        ["trade_date", "trade_hour", "timestamp", "month"]
    ].copy()

    pred_df["experiment"] = name
    pred_df["actual_mwh"] = y_test.to_numpy()
    pred_df["predicted_mwh"] = predicted

    all_predictions.append(pred_df)

    # 월별 평가
    for month, group in pred_df.groupby("month"):
        monthly_metrics = calculate_metrics(
            group["actual_mwh"],
            group["predicted_mwh"],
        )

        all_monthly.append({
            "experiment": name,
            "month": month,
            "rows": len(group),
            **monthly_metrics,
        })

    # 첫 7일 실제값·예측값 그래프
    plot_data = pred_df.head(168)

    plt.figure(figsize=(14, 5))
    plt.plot(
        plot_data["actual_mwh"].to_numpy(),
        label="Actual",
        linewidth=1.5,
    )
    plt.plot(
        plot_data["predicted_mwh"].to_numpy(),
        label="Predicted",
        linewidth=1.2,
        alpha=0.85,
    )
    plt.xlabel("Hour (first 7 days of 2023)")
    plt.ylabel("Wind electricity (MWh)")
    plt.title(f"{name}: Actual vs Predicted")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / f"{name}_prediction.png",
        dpi=150,
    )
    plt.close()

    # 학습된 파이프라인 저장
    joblib.dump(
        {
            "model": model,
            "features": features,
            "train_years": train_years,
            "target": "wind_mwh",
            "target_unit": "MWh",
        },
        MODEL_DIR / f"{name}_wind_rf.pkl",
    )

# =============================
# 5. 결과 비교 및 저장
# =============================
metrics_df = pd.DataFrame(all_metrics)
monthly_df = pd.DataFrame(all_monthly)
predictions_df = pd.concat(
    all_predictions, ignore_index=True
)

metrics_df.to_csv(
    RESULT_DIR / "wind_model_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

monthly_df.to_csv(
    RESULT_DIR / "wind_monthly_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

predictions_df.to_csv(
    RESULT_DIR / "wind_predictions_2023.csv",
    index=False,
    encoding="utf-8-sig",
)

# 월별 MAE 그래프
pivot = monthly_df.pivot(
    index="month",
    columns="experiment",
    values="MAE_MWh",
)

pivot.plot(kind="bar", figsize=(12, 5))
plt.title("Monthly Wind Forecast MAE - 2023")
plt.xlabel("Month")
plt.ylabel("MAE (MWh)")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(
    RESULT_DIR / "wind_monthly_mae.png",
    dpi=150,
)
plt.close()

print("\n===== 전체 실험 결과 =====")
print(metrics_df.round(4).to_string(index=False))

print("\n===== 월별 평가 결과 =====")
print(monthly_df.round(4).to_string(index=False))

print("\n결과 저장 완료:", RESULT_DIR)
