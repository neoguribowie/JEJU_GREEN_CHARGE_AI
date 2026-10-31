"""C005: 기본 실행은 2022년 검증 확률만 생성합니다.

python train_curtailment_c005.py
python train_curtailment_c005.py --full-run

--full-run은 기존 2023년 평가, 미래 예보 및 모델 파일을 덮어씁니다.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)

OUT = Path("results/curtailment_c005")
TARGET_FILE = Path("data/processed/curtailment_hourly_2021_2023.csv")
WEATHER_FILE = Path("data/processed/wind_weather_2020_2023.csv")
FORECAST_FILE = Path("data/processed/jeju_weather_forecast_4stations.csv")
STATIONS = ["jeju", "gosan", "seongsan", "seogwipo"]
KEYS = ["trade_date", "trade_hour"]
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


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
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])


def calculate_metrics(y, prob, threshold):
    predicted = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
    }


def prepare_features(df):
    df = df.copy()
    features = []
    for station in STATIONS:
        features.extend(f"{station}_{v}" for v in
                        ["wind_speed", "temperature", "humidity"])
        direction = pd.to_numeric(df[f"{station}_wind_direction"], errors="coerce")
        df[f"{station}_direction_sin"] = np.sin(np.deg2rad(direction))
        df[f"{station}_direction_cos"] = np.cos(np.deg2rad(direction))
        features.extend([f"{station}_direction_sin", f"{station}_direction_cos"])
    df = add_time_features(df)
    features.extend(["hour_sin", "hour_cos", "month_sin", "month_cos"])
    assert len(features) == 24
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    return df, features


def positive_probability(model, frame):
    classes = list(model.named_steps["rf"].classes_)
    if 1 not in classes:
        raise ValueError("학습 자료에 양성 클래스 (curtailment=1)가 없습니다.")
    return model.predict_proba(frame)[:, classes.index(1)]


def run_full(df, features, validation_results, best_threshold):
    """명시적인 --full-run에서만 기존 산출물을 재생성합니다."""
    validation_results.to_csv(OUT / "threshold_validation_2022.csv",
                              index=False, encoding="utf-8-sig")
    train_final = df[df["year"].isin([2021, 2022])].copy()
    test_2023 = df[df["year"] == 2023].copy()
    model = build_model()
    model.fit(train_final[features], train_final["curtailment"])
    test_prob = positive_probability(model, test_2023[features])
    y_test = test_2023["curtailment"].astype(int)
    result_rows = []
    for threshold in [0.5, best_threshold]:
        metrics = calculate_metrics(y_test, test_prob, threshold)
        prediction = (test_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        metrics.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                        "average_precision": average_precision_score(y_test, test_prob)})
        result_rows.append(metrics)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT / "metrics_2023.csv", index=False, encoding="utf-8-sig")
    predictions = test_2023[KEYS + ["curtailment"]].copy()
    predictions["probability"] = test_prob
    predictions["prediction_selected"] = (test_prob >= best_threshold).astype(int)
    predictions.to_csv(OUT / "predictions_2023.csv", index=False, encoding="utf-8-sig")
    print("\n===== EXP-C005 2023년 평가 =====")
    print(results.round(4).to_string(index=False))

    forecast = pd.read_csv(FORECAST_FILE)
    forecast["forecast_datetime"] = pd.to_datetime(forecast["forecast_datetime"])
    forecast = forecast[forecast["station"].isin(STATIONS)].copy()
    rename_map = {"TMP": "temperature", "REH": "humidity",
                  "WSD": "wind_speed", "VEC": "wind_direction"}
    for original, renamed in rename_map.items():
        if original not in forecast.columns:
            raise ValueError(f"미래 예보에 {original} 컬럼이 없습니다.")
        forecast[renamed] = pd.to_numeric(forecast[original], errors="coerce")
    counts = forecast.groupby("forecast_datetime")["station"].nunique()
    forecast = forecast[forecast["forecast_datetime"].isin(counts[counts == 4].index)].copy()
    wide_parts = []
    for station in STATIONS:
        sub = forecast[forecast["station"] == station][
            ["forecast_datetime"] + list(rename_map.values())].copy()
        sub = sub.rename(columns={v: f"{station}_{v}" for v in rename_map.values()})
        wide_parts.append(sub.set_index("forecast_datetime"))
    future = pd.concat(wide_parts, axis=1).reset_index()
    # 00시는 전날 24시에 대응하는 시간 종료 기준입니다.
    future["trade_hour"] = (future["forecast_datetime"].dt.hour - 1) % 24 + 1
    future["trade_date"] = (future["forecast_datetime"] - pd.to_timedelta(
        (future["forecast_datetime"].dt.hour == 0).astype(int), unit="D")).dt.normalize()
    future, _ = prepare_features(future)
    if future[features].isna().any().any():
        missing = future[features].isna().sum()
        raise ValueError("미래 예보 입력변수에 결측값이 있습니다:\n"
                         + missing[missing > 0].to_string())
    future = future[future["forecast_datetime"].dt.hour.between(10, 16)].copy()
    if future.empty:
        raise RuntimeError("낮 시간 미래 예보가 없습니다.")
    future_prob = positive_probability(model, future[features])
    future["risk_score"] = future_prob
    future["risk_flag"] = (future_prob >= best_threshold).astype(int)
    future["threshold"] = best_threshold
    future["model_version"] = "EXP-C005"
    future["forecast_issued_at"] = forecast["issued_at"].iloc[0]
    future_output = future[["forecast_datetime", "forecast_issued_at", "risk_score",
                            "risk_flag", "threshold", "model_version"]].sort_values("forecast_datetime")
    future_output.to_csv(OUT / "future_risk_forecast.csv", index=False, encoding="utf-8-sig")
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features, "threshold": best_threshold,
                 "train_years": [2021, 2022], "test_year": 2023,
                 "model_version": "EXP-C005"}, "models/curtailment_rf_c005.pkl")
    print(future_output.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", action="store_true",
                        help="2023년 평가, 미래 예보 및 모델을 재생성하고 기존 파일을 덮어씁니다.")
    args = parser.parse_args()
    output_file = OUT / "predictions_2022_validation.csv"
    # 독점 생성 모드로 기존 검증 CSV도 보호합니다.
    if output_file.exists():
        raise FileExistsError(f"기존 파일을 보호하기 위해 중단합니다: {output_file}")
    target = pd.read_csv(TARGET_FILE)
    weather = pd.read_csv(WEATHER_FILE)
    for frame in [target, weather]:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["trade_hour"] = pd.to_numeric(frame["trade_hour"]).astype(int)
        assert not frame.duplicated(KEYS).any()
    df = target.merge(weather, on=KEYS, how="left", validate="one_to_one",
                      suffixes=("", "_weather"))
    assert len(df) == 26280
    assert df["curtailment"].notna().all()
    df, features = prepare_features(df)
    df["year"] = df["trade_date"].dt.year
    train_2021 = df[df["year"] == 2021].copy()
    valid_2022 = df[df["year"] == 2022].copy()
    for year, frame in [(2021, train_2021), (2022, valid_2022)]:
        if len(frame) != 8760 or not frame["trade_hour"].between(1, 24).all():
            raise ValueError(f"{year}년 자료는 1~24시의 시간별 8760행이어야 합니다.")
        if not frame["curtailment"].isin([0, 1]).all():
            raise ValueError(f"{year}년 curtailment는 0 또는 1이어야 합니다.")
    validation_model = build_model()
    validation_model.fit(train_2021[features], train_2021["curtailment"])
    validation_prob = positive_probability(validation_model, valid_2022[features])
    validation_results = pd.DataFrame([
        calculate_metrics(valid_2022["curtailment"], validation_prob, threshold)
        for threshold in THRESHOLDS])
    best = validation_results.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    best_threshold = float(best["threshold"])
    predictions = valid_2022[KEYS + ["curtailment"]].copy()
    predictions["probability"] = validation_prob
    predictions["prediction_selected"] = (validation_prob >= best_threshold).astype(int)
    predictions["threshold"] = best_threshold
    predictions = predictions.sort_values(KEYS)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, mode="x", index=False, encoding="utf-8-sig")
    print(f"2021년 학습 → 2022년 검증 {len(predictions)}행 저장: {output_file}")
    print(f"2022년 선택 임계값: {best_threshold}, 검증 F1: {float(best['f1']):.4f}")
    if args.full_run:
        run_full(df, features, validation_results, best_threshold)


if __name__ == "__main__":
    main()
