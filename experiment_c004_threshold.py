
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score
)

# ============================
# 1. 파일 설정
# ============================
OUT = Path("results/curtailment_c004")
MODELS = Path("models")
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

weather = pd.read_csv(
    "data/processed/wind_weather_2020_2023.csv"
)
target = pd.read_csv(
    "data/processed/curtailment_hourly_2021_2023.csv"
)
demand = pd.read_csv(
    "data/processed/jeju_demand_hourly_2021_2023.csv"
)

keys = ["trade_date", "trade_hour"]

for frame in [weather, target, demand]:
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["trade_hour"] = pd.to_numeric(
        frame["trade_hour"]
    ).astype(int)
    assert not frame.duplicated(keys).any()

# 2021~2023년 자료 통합
df = target.merge(
    weather.drop(columns=["timestamp"], errors="ignore"),
    on=keys, how="left", validate="one_to_one",
    suffixes=("", "_weather")
)

df = df.merge(
    demand[keys + ["demand_mwh"]],
    on=keys, how="left", validate="one_to_one"
)

assert len(df) == 26280
assert df["demand_mwh"].notna().all()
assert df["curtailment"].notna().all()

# ============================
# 2. 입력변수 (기존 모델과 동일)
# ============================
stations = ["jeju", "gosan", "seongsan", "seogwipo"]
features = []

for s in stations:
    features.extend([
        f"{s}_wind_speed",
        f"{s}_temperature",
        f"{s}_humidity",
        f"{s}_pressure"
    ])

    radians = np.deg2rad(
        pd.to_numeric(
            df[f"{s}_wind_direction"],
            errors="coerce"
        )
    )

    df[f"{s}_direction_sin"] = np.sin(radians)
    df[f"{s}_direction_cos"] = np.cos(radians)

    features.extend([
        f"{s}_direction_sin",
        f"{s}_direction_cos"
    ])

df["hour_sin"] = np.sin(
    2 * np.pi * df["trade_hour"] / 24
)
df["hour_cos"] = np.cos(
    2 * np.pi * df["trade_hour"] / 24
)
df["month_sin"] = np.sin(
    2 * np.pi * df["trade_date"].dt.month / 12
)
df["month_cos"] = np.cos(
    2 * np.pi * df["trade_date"].dt.month / 12
)

features.extend([
    "hour_sin", "hour_cos",
    "month_sin", "month_cos"
])

for col in features + ["demand_mwh"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[features + ["demand_mwh"]] = (
    df[features + ["demand_mwh"]]
    .replace([np.inf, -np.inf], np.nan)
)

df["year"] = df["trade_date"].dt.year

# 임계값 후보를 미리 고정
THRESHOLDS = np.round(
    np.arange(0.05, 0.96, 0.05), 2
)

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

def scores(y, prob, threshold):
    pred = (prob >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "predicted_events": int(pred.sum())
    }

# ============================
# 3. C001, C003 모두 실험
# ============================
experiments = {
    "C001": features,
    "C003": features + ["demand_mwh"]
}

all_validation = []
all_test = []

for name, cols in experiments.items():

    print(f"\n===== {name} 임계값 실험 =====")

    train_2021 = df[df["year"] == 2021]
    valid_2022 = df[df["year"] == 2022]
    train_final = df[df["year"].isin([2021, 2022])]
    test_2023 = df[df["year"] == 2023]

    # 2021년 학습
    validation_model = build_model()
    validation_model.fit(
        train_2021[cols],
        train_2021["curtailment"]
    )

    # 2022년 검증
    valid_prob = validation_model.predict_proba(
        valid_2022[cols]
    )[:, 1]

    validation_rows = []

    for t in THRESHOLDS:
        result = scores(
            valid_2022["curtailment"],
            valid_prob,
            t
        )
        result["model"] = name
        validation_rows.append(result)

    valid_table = pd.DataFrame(validation_rows)
    all_validation.append(valid_table)

    # F1 최대인 임계값을 선택
    # 동률이면 더 높은 임계값 선택
    best_row = valid_table.sort_values(
        ["f1", "threshold"],
        ascending=[False, False]
    ).iloc[0]

    best_t = float(best_row["threshold"])

    print("\n2022년 임계값 검증:")
    print(valid_table.round(4).to_string(index=False))
    print(f"\n선택된 임계값: {best_t:.2f}")

    # 기존 C001/C003과 같은 기간으로 재학습
    final_model = build_model()
    final_model.fit(
        train_final[cols],
        train_final["curtailment"]
    )

    # 2023년 후속 평가
    test_prob = final_model.predict_proba(
        test_2023[cols]
    )[:, 1]

    y_test = test_2023["curtailment"].astype(int)

    # 선택한 임계값과 기존 0.5 모두 비교
    for label, threshold in [
        ("baseline_0.5", 0.5),
        ("selected", best_t)
    ]:
        result = scores(y_test, test_prob, threshold)
        result["model"] = name
        result["method"] = label
        result["average_precision"] = (
            average_precision_score(y_test, test_prob)
        )

        pred = (test_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        result.update({
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        })
        all_test.append(result)

    prediction_df = test_2023[
        ["trade_date", "trade_hour", "curtailment"]
    ].copy()

    prediction_df["probability"] = test_prob
    prediction_df["prediction_0_5"] = (
        test_prob >= 0.5
    ).astype(int)
    prediction_df["prediction_selected"] = (
        test_prob >= best_t
    ).astype(int)

    prediction_df.to_csv(
        OUT / f"{name}_predictions_2023.csv",
        index=False, encoding="utf-8-sig"
    )

    joblib.dump({
        "model": final_model,
        "features": cols,
        "threshold": best_t,
        "train_years": [2021, 2022],
        "validation_year": 2022,
        "validation_train_year": 2021,
        "test_year": 2023
    }, MODELS / f"curtailment_{name}_c004.pkl")

# ============================
# 4. 결과 저장
# ============================
validation_df = pd.concat(
    all_validation, ignore_index=True
)
test_df = pd.DataFrame(all_test)

validation_df.to_csv(
    OUT / "threshold_validation_2022.csv",
    index=False, encoding="utf-8-sig"
)

test_df.to_csv(
    OUT / "threshold_comparison_2023.csv",
    index=False, encoding="utf-8-sig"
)

print("\n===== 2023년 최종 비교 =====")
print(test_df.round(4).to_string(index=False))

print("\n저장 완료:", OUT)
