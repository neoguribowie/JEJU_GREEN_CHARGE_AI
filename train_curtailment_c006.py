
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier
)
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)

from refresh_c005_forecast import prepare_features

OUT = Path("results/curtailment_c006")
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["trade_date", "trade_hour"]
STATIONS = ["jeju", "gosan", "seongsan", "seogwipo"]

# ======================================
# 1. 저장된 모델과 입력변수 확인
# ======================================
c005_saved = joblib.load("models/curtailment_rf_c005.pkl")
d002_saved = joblib.load("models/demand_rf_d002.pkl")

weather_features = c005_saved["features"]
demand_features = d002_saved["features"]

assert len(weather_features) == 24
assert len(demand_features) == 27

C006_FEATURES = weather_features + ["predicted_demand_mwh"]

# ======================================
# 2. 공통 입력변수 생성
# ======================================
def add_features(df):
    df = df.copy()

    for station in STATIONS:
        direction = pd.to_numeric(
            df[f"{station}_wind_direction"],
            errors="coerce"
        )

        radians = np.deg2rad(direction)
        df[f"{station}_direction_sin"] = np.sin(radians)
        df[f"{station}_direction_cos"] = np.cos(radians)

    date = pd.to_datetime(df["trade_date"])
    hour = pd.to_numeric(df["trade_hour"])

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)

    weekday = date.dt.dayofweek
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    df["is_weekend"] = (weekday >= 5).astype(int)

    required = set(weather_features + demand_features)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력변수 누락: {sorted(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[list(required)] = df[list(required)].replace(
        [np.inf, -np.inf], np.nan
    )

    return df


def new_demand_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])


def new_curtailment_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])


# ======================================
# 3. 과거 데이터 결합
# ======================================
weather = pd.read_csv(
    "data/processed/wind_weather_2020_2023.csv"
)
target = pd.read_csv(
    "data/processed/curtailment_hourly_2021_2023.csv"
)
demand = pd.read_csv(
    "data/processed/jeju_demand_hourly_2021_2023.csv"
)

for frame in [weather, target, demand]:
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["trade_hour"] = frame["trade_hour"].astype(int)
    assert not frame.duplicated(KEYS).any()

df = target.merge(
    weather,
    on=KEYS,
    how="left",
    validate="one_to_one",
    suffixes=("", "_weather")
)

df = df.merge(
    demand[KEYS + ["demand_mwh"]],
    on=KEYS,
    how="left",
    validate="one_to_one"
)

assert len(df) == 26280
assert df["demand_mwh"].notna().all()
assert df["curtailment"].notna().all()

df = add_features(df)
df["year"] = df["trade_date"].dt.year

# ======================================
# 4. 데이터 누출을 방지한 수요 예측
# ======================================

# 2021년 상반기: 수요 모델 기초 학습
early_2021 = df[
    (df["trade_date"] >= "2021-01-01")
    & (df["trade_date"] < "2021-07-01")
].copy()

# 2021년 하반기: 첫 번째 학습용 예측
late_2021 = df[
    (df["trade_date"] >= "2021-07-01")
    & (df["trade_date"] < "2022-01-01")
].copy()

year_2021 = df[df["year"] == 2021].copy()
year_2022 = df[df["year"] == 2022].copy()
year_2023 = df[df["year"] == 2023].copy()

# 상반기로 학습 -> 2021 하반기 예측
demand_model_a = new_demand_model()
demand_model_a.fit(
    early_2021[demand_features],
    early_2021["demand_mwh"]
)

late_2021["predicted_demand_mwh"] = (
    demand_model_a.predict(late_2021[demand_features])
)

# 2021년 전체로 학습 -> 2022년 예측
demand_model_b = new_demand_model()
demand_model_b.fit(
    year_2021[demand_features],
    year_2021["demand_mwh"]
)

year_2022["predicted_demand_mwh"] = (
    demand_model_b.predict(year_2022[demand_features])
)

# 2023년은 저장된 D002 모델로 예측
year_2023["predicted_demand_mwh"] = (
    d002_saved["model"].predict(
        year_2023[demand_features]
    )
)

print("2021년 하반기 학습 행:", len(late_2021))
print("2022년 검증 행:", len(year_2022))
print("2023년 평가 행:", len(year_2023))

# ======================================
# 5. 2022년 임계값 검증
# ======================================
validation_model = new_curtailment_model()

validation_model.fit(
    late_2021[C006_FEATURES],
    late_2021["curtailment"]
)

val_prob = validation_model.predict_proba(
    year_2022[C006_FEATURES]
)[:, 1]

threshold_rows = []

for threshold in np.round(np.arange(0.05, 0.96, 0.05), 2):
    pred = (val_prob >= threshold).astype(int)

    threshold_rows.append({
        "threshold": threshold,
        "precision": precision_score(
            year_2022["curtailment"], pred, zero_division=0
        ),
        "recall": recall_score(
            year_2022["curtailment"], pred, zero_division=0
        ),
        "f1": f1_score(
            year_2022["curtailment"], pred, zero_division=0
        )
    })

validation_table = pd.DataFrame(threshold_rows)
validation_table.to_csv(
    OUT / "threshold_validation_2022.csv",
    index=False, encoding="utf-8-sig"
)

best = validation_table.sort_values(
    ["f1", "threshold"],
    ascending=[False, False]
).iloc[0]

best_threshold = float(best["threshold"])

print("\n2022년 선택 임계값:", best_threshold)
print("2022년 검증 F1:", round(float(best["f1"]), 4))

# ======================================
# 6. C006 최종 학습 및 2023년 평가
# ======================================
c006_train = pd.concat(
    [late_2021, year_2022],
    ignore_index=True
)

model = new_curtailment_model()

model.fit(
    c006_train[C006_FEATURES],
    c006_train["curtailment"]
)

test_prob = model.predict_proba(
    year_2023[C006_FEATURES]
)[:, 1]

y_test = year_2023["curtailment"].astype(int)

result_rows = []

for threshold in [0.5, best_threshold]:
    predicted = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predicted, labels=[0, 1]
    ).ravel()

    result_rows.append({
        "threshold": threshold,
        "precision": precision_score(
            y_test, predicted, zero_division=0
        ),
        "recall": recall_score(
            y_test, predicted, zero_division=0
        ),
        "f1": f1_score(
            y_test, predicted, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_test, test_prob
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

metrics = pd.DataFrame(result_rows)
metrics.to_csv(
    OUT / "metrics_2023.csv",
    index=False, encoding="utf-8-sig"
)

print("\n===== EXP-C006 2023년 평가 =====")
print(metrics.round(4).to_string(index=False))

predictions = year_2023[KEYS + ["curtailment"]].copy()
predictions["predicted_demand_mwh"] = (
    year_2023["predicted_demand_mwh"].to_numpy()
)
predictions["risk_score"] = test_prob
predictions["risk_flag"] = (
    test_prob >= best_threshold
).astype(int)

predictions.to_csv(
    OUT / "predictions_2023.csv",
    index=False, encoding="utf-8-sig"
)

# ======================================
# 7. 미래 기상예보 + D002 + C006
# ======================================
forecast = pd.read_csv(
    "data/processed/jeju_weather_forecast_4stations.csv"
)

# C005와 동일한 미래 예보 입력 전처리
future = prepare_features(
    forecast, weather_features
)

# D002에 필요한 요일·주말 변수 추가
future = add_features(future)

# 최신 미래 예보 중 낮 시간만 선택
future = future[
    future["forecast_datetime"].dt.hour.between(10, 16)
].copy()

if future.empty:
    raise RuntimeError("미래 낮 시간 예보가 없습니다.")

if future[demand_features].isna().any().any():
    raise ValueError("미래 수요 모델 입력변수에 결측이 있습니다.")

# 예측된 미래 수요 사용
future["predicted_demand_mwh"] = (
    d002_saved["model"].predict(
        future[demand_features]
    )
)

# 출력제어 위험 점수 계산
future["risk_score"] = model.predict_proba(
    future[C006_FEATURES]
)[:, 1]

future["risk_flag"] = (
    future["risk_score"] >= best_threshold
).astype(int)

future["threshold"] = best_threshold
future["model_version"] = "EXP-C006"
future["forecast_issued_at"] = forecast["issued_at"].iloc[0]

future_output = future[[
    "forecast_datetime",
    "forecast_issued_at",
    "predicted_demand_mwh",
    "risk_score",
    "risk_flag",
    "threshold",
    "model_version"
]].sort_values("forecast_datetime")

future_output.to_csv(
    OUT / "future_risk_forecast.csv",
    index=False, encoding="utf-8-sig"
)

joblib.dump({
    "model": model,
    "features": C006_FEATURES,
    "threshold": best_threshold,
    "model_version": "EXP-C006",
    "demand_model_version": "EXP-D002",
    "train_period": "2021-07 to 2022-12",
    "test_year": 2023
}, "models/curtailment_rf_c006.pkl")

print("\n===== C006 미래 시험 예측 =====")
print(
    future_output.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n저장 완료:", OUT)
