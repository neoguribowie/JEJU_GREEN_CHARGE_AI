
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. 설정
# ==========================================
OUT = Path("results/demand_d002")
OUT.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

DEMAND_FILE = Path(
    "data/processed/jeju_demand_hourly_2021_2023.csv"
)
WEATHER_FILE = Path(
    "data/processed/wind_weather_2020_2023.csv"
)
FORECAST_FILE = Path(
    "data/processed/jeju_weather_forecast_4stations.csv"
)

STATIONS = ["jeju", "gosan", "seongsan", "seogwipo"]
KEYS = ["trade_date", "trade_hour"]


def add_features(df, forecast_mode=False):
    """과거 관측자료 또는 미래 예보의 입력변수 구성."""
    df = df.copy()
    features = []

    for station in STATIONS:
        for variable in ["temperature", "humidity", "wind_speed"]:
            features.append(f"{station}_{variable}")

        direction_col = f"{station}_wind_direction"
        direction = pd.to_numeric(df[direction_col], errors="coerce")
        radians = np.deg2rad(direction)

        df[f"{station}_direction_sin"] = np.sin(radians)
        df[f"{station}_direction_cos"] = np.cos(radians)

        features.extend([
            f"{station}_direction_sin",
            f"{station}_direction_cos"
        ])

    hour = pd.to_numeric(df["trade_hour"])
    month = pd.to_datetime(df["trade_date"]).dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    features.extend([
        "hour_sin", "hour_cos",
        "month_sin", "month_cos"
    ])

    # ====================================
    # EXP-D002: 요일 및 주말 정보 추가
    # ====================================
    day_of_week = pd.to_datetime(
        df["trade_date"]
    ).dt.dayofweek

    # 월요일=0, 화요일=1, ..., 일요일=6
    df["weekday_sin"] = np.sin(
        2 * np.pi * day_of_week / 7
    )
    df["weekday_cos"] = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # 토요일(5), 일요일(6)은 주말
    df["is_weekend"] = (
        day_of_week >= 5
    ).astype(int)

    features.extend([
        "weekday_sin",
        "weekday_cos",
        "is_weekend"
    ])

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[features] = df[features].replace(
        [np.inf, -np.inf], np.nan
    )

    return df, features


# ==========================================
# 2. 과거 데이터 불러오기
# ==========================================
demand = pd.read_csv(DEMAND_FILE)
weather = pd.read_csv(WEATHER_FILE)

for frame in [demand, weather]:
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["trade_hour"] = pd.to_numeric(
        frame["trade_hour"]
    ).astype(int)

    assert not frame.duplicated(KEYS).any()

df = demand.merge(
    weather,
    on=KEYS,
    how="left",
    validate="one_to_one",
    suffixes=("", "_weather")
)

assert len(df) == 26280
assert df["demand_mwh"].notna().all()

df, features = add_features(df)

assert len(features) == 27

df["year"] = df["trade_date"].dt.year

train = df[df["year"].isin([2021, 2022])].copy()
test = df[df["year"] == 2023].copy()

print("===== EXP-D002 =====")
print("입력변수 개수:", len(features))
print("학습 데이터:", len(train))
print("평가 데이터:", len(test))

# ==========================================
# 3. Random Forest 학습
# ==========================================
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(
    train[features],
    train["demand_mwh"]
)

# ==========================================
# 4. 2023년 성능 평가
# ==========================================
pred = model.predict(test[features])
actual = test["demand_mwh"].to_numpy()

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)

print("\n===== 2023년 전체 성능 =====")
print(f"MAE  : {mae:.4f} MWh")
print(f"RMSE : {rmse:.4f} MWh")
print(f"R²   : {r2:.4f}")

# 낮 시간대 성능도 별도 확인
day_mask = test["trade_hour"].between(10, 16).to_numpy()

day_mae = mean_absolute_error(
    actual[day_mask], pred[day_mask]
)
day_rmse = np.sqrt(mean_squared_error(
    actual[day_mask], pred[day_mask]
))
day_r2 = r2_score(
    actual[day_mask], pred[day_mask]
)

print("\n===== 10~16시 성능 =====")
print(f"MAE  : {day_mae:.4f} MWh")
print(f"RMSE : {day_rmse:.4f} MWh")
print(f"R²   : {day_r2:.4f}")

metrics = pd.DataFrame([
    {
        "period": "all_hours",
        "MAE_MWh": mae,
        "RMSE_MWh": rmse,
        "R2": r2
    },
    {
        "period": "10_to_16",
        "MAE_MWh": day_mae,
        "RMSE_MWh": day_rmse,
        "R2": day_r2
    }
])

metrics.to_csv(
    OUT / "metrics_2023.csv",
    index=False,
    encoding="utf-8-sig"
)

predictions = test[KEYS].copy()
predictions["actual_demand_mwh"] = actual
predictions["predicted_demand_mwh"] = pred
predictions["error_mwh"] = actual - pred

predictions.to_csv(
    OUT / "predictions_2023.csv",
    index=False,
    encoding="utf-8-sig"
)

# 모델 저장
MODEL_PATH = Path("models/demand_rf_d002.pkl")

joblib.dump({
    "model": model,
    "features": features,
    "train_years": [2021, 2022],
    "test_year": 2023,
    "target_unit": "MWh",
    "model_version": "EXP-D002"
}, MODEL_PATH)

print("\n모델 저장:", MODEL_PATH)

# ==========================================
# 5. 실제 미래 기상예보 기반 전력수요 예측
# ==========================================
forecast = pd.read_csv(FORECAST_FILE)
forecast["forecast_datetime"] = pd.to_datetime(
    forecast["forecast_datetime"]
)

forecast = forecast[
    forecast["station"].isin(STATIONS)
].copy()

required = ["TMP", "REH", "WSD", "VEC"]

for col in required:
    if col not in forecast.columns:
        raise ValueError(f"예보 필수 컬럼 누락: {col}")

    forecast[col] = pd.to_numeric(
        forecast[col], errors="coerce"
    )

if forecast.duplicated(
    ["station", "forecast_datetime"]
).any():
    raise ValueError("지점·시간별 중복 예보가 있습니다.")

# 네 지역 예보가 모두 존재하는 시각만 선택
counts = forecast.groupby(
    "forecast_datetime"
)["station"].nunique()

complete_times = counts[counts == 4].index

forecast = forecast[
    forecast["forecast_datetime"].isin(complete_times)
].copy()

# 세로형 데이터 -> 시간별 가로형 데이터
parts = []

for station in STATIONS:
    part = forecast[
        forecast["station"] == station
    ][["forecast_datetime"] + required].copy()

    part = part.rename(columns={
        "TMP": f"{station}_temperature",
        "REH": f"{station}_humidity",
        "WSD": f"{station}_wind_speed",
        "VEC": f"{station}_wind_direction"
    })

    parts.append(part.set_index("forecast_datetime"))

future = pd.concat(parts, axis=1).reset_index()

# 기존 C005와 동일한 거래시간 정렬 가정
# 예보 00시는 전날 거래 24시로 처리
future["trade_hour"] = (
    future["forecast_datetime"].dt.hour.replace({0: 24})
)

future["trade_date"] = (
    future["forecast_datetime"]
    - pd.to_timedelta(
        (future["forecast_datetime"].dt.hour == 0).astype(int),
        unit="D"
    )
).dt.normalize()

future, future_features = add_features(future)

assert future_features == features

# 미래 예보값 누락은 임의로 채우지 않음
if future[features].isna().any().any():
    missing = future[features].isna().sum()
    raise ValueError(
        "미래 예보값 누락:\n"
        + missing[missing > 0].to_string()
    )

# 이번 프로젝트의 충전 검토 시간대
future = future[
    future["forecast_datetime"].dt.hour.between(10, 16)
].copy()

if future.empty:
    raise RuntimeError("낮 시간대 미래 예보가 없습니다.")

future["predicted_demand_mwh"] = model.predict(
    future[features]
)

future["model_version"] = "EXP-D002"
future["forecast_issued_at"] = forecast["issued_at"].iloc[0]

future_output = future[[
    "forecast_datetime",
    "forecast_issued_at",
    "predicted_demand_mwh",
    "model_version"
]].sort_values("forecast_datetime")

future_output.to_csv(
    OUT / "future_demand_forecast.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 미래 전력수요 시험 예측 =====")
print(
    future_output.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\n미래 예측 시점:", len(future_output))
print("저장 완료:", OUT)
