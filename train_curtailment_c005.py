
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# ==========================================
# 1. 파일 경로 및 모델 설정
# ==========================================
OUT = Path("results/curtailment_c005")
OUT.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)

TARGET_FILE = Path(
    "data/processed/curtailment_hourly_2021_2023.csv"
)
WEATHER_FILE = Path(
    "data/processed/wind_weather_2020_2023.csv"
)
FORECAST_FILE = Path(
    "data/processed/jeju_weather_forecast_4stations.csv"
)

STATIONS = ["jeju", "gosan", "seongsan", "seogwipo"]
KEYS = ["trade_date", "trade_hour"]

# C004와 같은 후보 임계값
THRESHOLDS = np.round(
    np.arange(0.05, 0.96, 0.05), 2
)


def add_time_features(df):
    df = df.copy()

    hour = df["trade_hour"].astype(float)
    month = pd.to_datetime(df["trade_date"]).dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample"
        ))
    ])


def calculate_metrics(y, prob, threshold):
    predicted = (prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y, predicted, zero_division=0
        ),
        "recall": recall_score(
            y, predicted, zero_division=0
        ),
        "f1": f1_score(
            y, predicted, zero_division=0
        )
    }


# ==========================================
# 2. 과거 기상·출력제어 자료 준비
# ==========================================
target = pd.read_csv(TARGET_FILE)
weather = pd.read_csv(WEATHER_FILE)

for frame in [target, weather]:
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"]
    )
    frame["trade_hour"] = pd.to_numeric(
        frame["trade_hour"]
    ).astype(int)

    assert not frame.duplicated(KEYS).any()

df = target.merge(
    weather,
    on=KEYS,
    how="left",
    validate="one_to_one",
    suffixes=("", "_weather")
)

assert len(df) == 26280
assert df["curtailment"].notna().all()

features = []

for station in STATIONS:
    # 미래 예보에 존재하는 변수만 사용
    for variable in [
        "wind_speed", "temperature", "humidity"
    ]:
        features.append(f"{station}_{variable}")

    direction_col = f"{station}_wind_direction"

    direction = pd.to_numeric(
        df[direction_col], errors="coerce"
    )

    df[f"{station}_direction_sin"] = np.sin(
        np.deg2rad(direction)
    )
    df[f"{station}_direction_cos"] = np.cos(
        np.deg2rad(direction)
    )

    features.extend([
        f"{station}_direction_sin",
        f"{station}_direction_cos"
    ])

df = add_time_features(df)

features.extend([
    "hour_sin", "hour_cos",
    "month_sin", "month_cos"
])

assert len(features) == 24

for col in features:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    )

df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

df["year"] = df["trade_date"].dt.year

train_2021 = df[df["year"] == 2021].copy()
valid_2022 = df[df["year"] == 2022].copy()
train_final = df[df["year"].isin([2021, 2022])].copy()
test_2023 = df[df["year"] == 2023].copy()

print("입력변수 개수:", len(features))
print("2021년 학습:", len(train_2021))
print("2022년 검증:", len(valid_2022))
print("2023년 평가:", len(test_2023))


# ==========================================
# 3. 2022년 검증으로 임계값 선택
# ==========================================
validation_model = build_model()

validation_model.fit(
    train_2021[features],
    train_2021["curtailment"]
)

validation_prob = validation_model.predict_proba(
    valid_2022[features]
)[:, 1]

validation_results = pd.DataFrame([
    calculate_metrics(
        valid_2022["curtailment"],
        validation_prob,
        threshold
    )
    for threshold in THRESHOLDS
])

validation_results.to_csv(
    OUT / "threshold_validation_2022.csv",
    index=False,
    encoding="utf-8-sig"
)

# F1 동률이면 더 높은 임계값 선택
best = validation_results.sort_values(
    ["f1", "threshold"],
    ascending=[False, False]
).iloc[0]

best_threshold = float(best["threshold"])

print("\n2022년 선택 임계값:", best_threshold)
print("2022년 검증 F1:", round(float(best["f1"]), 4))


# ==========================================
# 4. 재학습 후 2023년 평가
# ==========================================
model = build_model()

model.fit(
    train_final[features],
    train_final["curtailment"]
)

test_prob = model.predict_proba(
    test_2023[features]
)[:, 1]

y_test = test_2023["curtailment"].astype(int)

result_rows = []

for threshold in [0.5, best_threshold]:
    metrics = calculate_metrics(
        y_test, test_prob, threshold
    )

    prediction = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, prediction, labels=[0, 1]
    ).ravel()

    metrics.update({
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "average_precision": average_precision_score(
            y_test, test_prob
        )
    })

    result_rows.append(metrics)

results = pd.DataFrame(result_rows)

results.to_csv(
    OUT / "metrics_2023.csv",
    index=False,
    encoding="utf-8-sig"
)

predictions = test_2023[
    ["trade_date", "trade_hour", "curtailment"]
].copy()

predictions["probability"] = test_prob
predictions["prediction_selected"] = (
    test_prob >= best_threshold
).astype(int)

predictions.to_csv(
    OUT / "predictions_2023.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== EXP-C005 2023년 평가 =====")
print(results.round(4).to_string(index=False))


# ==========================================
# 5. 미래 기상예보 입력자료 만들기
# ==========================================
forecast = pd.read_csv(FORECAST_FILE)

forecast["forecast_datetime"] = pd.to_datetime(
    forecast["forecast_datetime"]
)

forecast = forecast[
    forecast["station"].isin(STATIONS)
].copy()

# 실제 예보값 -> 기존 모델 입력변수명
rename_map = {
    "TMP": "temperature",
    "REH": "humidity",
    "WSD": "wind_speed",
    "VEC": "wind_direction"
}

for original in rename_map:
    if original not in forecast.columns:
        raise ValueError(
            f"미래 예보에 {original} 컬럼이 없습니다."
        )

for original, renamed in rename_map.items():
    forecast[renamed] = pd.to_numeric(
        forecast[original], errors="coerce"
    )

# 네 지점 모두 존재하는 시각만 사용
counts = forecast.groupby(
    "forecast_datetime"
)["station"].nunique()

complete_times = counts[counts == 4].index

forecast = forecast[
    forecast["forecast_datetime"].isin(complete_times)
].copy()

# 각 시간에 네 지점의 예보값을 가로로 결합
wide_parts = []

for station in STATIONS:
    sub = forecast[
        forecast["station"] == station
    ][[
        "forecast_datetime",
        "temperature",
        "humidity",
        "wind_speed",
        "wind_direction"
    ]].copy()

    sub = sub.rename(columns={
        "temperature": f"{station}_temperature",
        "humidity": f"{station}_humidity",
        "wind_speed": f"{station}_wind_speed",
        "wind_direction": f"{station}_wind_direction"
    })

    wide_parts.append(
        sub.set_index("forecast_datetime")
    )

future = pd.concat(wide_parts, axis=1).reset_index()

# 기존 자료는 1~24시 거래시간.
# 과거 병합 때 사용한 '시간 종료 기준' 가정에 맞춤.
future["trade_hour"] = (
    (future["forecast_datetime"].dt.hour - 1) % 24 + 1
)

# 00시는 전날 24시에 대응
future["trade_date"] = (
    future["forecast_datetime"]
    - pd.to_timedelta(
        (future["forecast_datetime"].dt.hour == 0).astype(int),
        unit="D"
    )
).dt.normalize()

for station in STATIONS:
    direction = future[
        f"{station}_wind_direction"
    ]

    future[f"{station}_direction_sin"] = np.sin(
        np.deg2rad(direction)
    )
    future[f"{station}_direction_cos"] = np.cos(
        np.deg2rad(direction)
    )

future = add_time_features(future)

for col in features:
    future[col] = pd.to_numeric(
        future[col], errors="coerce"
    )

future[features] = future[features].replace(
    [np.inf, -np.inf], np.nan
)

# 예보 필수변수 누락 시 임의 대체하지 않음
if future[features].isna().any().any():
    missing = future[features].isna().sum()
    raise ValueError(
        "미래 예보 입력변수에 결측값이 있습니다:\n"
        + missing[missing > 0].to_string()
    )

# 낮 시간 우선: 10~16시
future = future[
    future["forecast_datetime"].dt.hour.between(10, 16)
].copy()

if future.empty:
    raise RuntimeError("낮 시간 미래 예보가 없습니다.")

future_prob = model.predict_proba(
    future[features]
)[:, 1]

future["risk_score"] = future_prob
future["risk_flag"] = (
    future_prob >= best_threshold
).astype(int)
future["threshold"] = best_threshold
future["model_version"] = "EXP-C005"

# 예보가 발표된 시각 정보 유지
issued_at = forecast["issued_at"].iloc[0]
future["forecast_issued_at"] = issued_at

future_output = future[[
    "forecast_datetime",
    "forecast_issued_at",
    "risk_score",
    "risk_flag",
    "threshold",
    "model_version"
]].sort_values("forecast_datetime")

future_output.to_csv(
    OUT / "future_risk_forecast.csv",
    index=False,
    encoding="utf-8-sig"
)

# 모델 저장
joblib.dump({
    "model": model,
    "features": features,
    "threshold": best_threshold,
    "train_years": [2021, 2022],
    "test_year": 2023,
    "model_version": "EXP-C005"
}, "models/curtailment_rf_c005.pkl")

print("\n===== 미래 시험 예측 =====")
print(
    future_output.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n미래 예보 시점 수:", len(future_output))
print("저장 완료:", OUT)
