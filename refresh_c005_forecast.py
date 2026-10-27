
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from fetch_weather_forecast import latest_base_time
from fetch_weather_forecast_4stations import (
    STATIONS, fetch_station
)

MODEL_PATH = Path("models/curtailment_rf_c005.pkl")
OUTPUT_DIR = Path("results/curtailment_c005")
FORECAST_PATH = Path(
    "data/processed/jeju_weather_forecast_4stations.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)


def prepare_features(forecast, features):
    """최신 예보를 C005 모델의 입력 형태로 변환."""

    forecast = forecast.copy()

    forecast["forecast_datetime"] = pd.to_datetime(
        forecast["forecast_datetime"]
    )

    needed = ["TMP", "REH", "WSD", "VEC"]

    for column in needed:
        if column not in forecast.columns:
            raise ValueError(f"예보 변수 누락: {column}")

        forecast[column] = pd.to_numeric(
            forecast[column], errors="coerce"
        )

    # 각 시간에 네 지역이 모두 존재하는지 검증
    if forecast.duplicated(
        ["station", "forecast_datetime"]
    ).any():
        raise ValueError("지역별 중복 예보가 있습니다.")

    counts = forecast.groupby(
        "forecast_datetime"
    )["station"].nunique()

    complete_times = counts[counts == 4].index

    forecast = forecast[
        forecast["forecast_datetime"].isin(complete_times)
    ].copy()

    parts = []

    for station in STATIONS:
        sub = forecast[
            forecast["station"] == station
        ][["forecast_datetime"] + needed].copy()

        sub = sub.rename(columns={
            "TMP": f"{station}_temperature",
            "REH": f"{station}_humidity",
            "WSD": f"{station}_wind_speed",
            "VEC": f"{station}_wind_direction"
        })

        parts.append(sub.set_index("forecast_datetime"))

    future = pd.concat(parts, axis=1).reset_index()

    # 기존 C005와 동일한 시간 정렬 가정
    future["trade_hour"] = (
        future["forecast_datetime"].dt.hour
        .replace({0: 24})
    )

    future["trade_date"] = (
        future["forecast_datetime"]
        - pd.to_timedelta(
            (future["forecast_datetime"].dt.hour == 0)
            .astype(int),
            unit="D"
        )
    ).dt.normalize()

    for station in STATIONS:
        direction = future[
            f"{station}_wind_direction"
        ]

        radians = np.deg2rad(direction)

        future[f"{station}_direction_sin"] = np.sin(
            radians
        )
        future[f"{station}_direction_cos"] = np.cos(
            radians
        )

    hour = future["trade_hour"]
    month = future["trade_date"].dt.month

    future["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    future["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )
    future["month_sin"] = np.sin(
        2 * np.pi * month / 12
    )
    future["month_cos"] = np.cos(
        2 * np.pi * month / 12
    )

    missing_columns = [
        c for c in features if c not in future.columns
    ]

    if missing_columns:
        raise ValueError(
            f"모델 입력변수 누락: {missing_columns}"
        )

    future[features] = future[features].replace(
        [np.inf, -np.inf], np.nan
    )

    # 실제 미래 예보값이 없는 시간은 예측하지 않음
    if future[features].isna().any().any():
        missing = future[features].isna().sum()
        raise ValueError(
            "미래 예보값 누락:\n"
            + missing[missing > 0].to_string()
        )

    return future


def main():
    # 1. 기존 학습 모델 로드
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "저장된 C005 모델이 없습니다. "
            "train_curtailment_c005.py를 먼저 실행하세요."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    features = saved["features"]
    threshold = float(saved["threshold"])

    print("C005 모델 불러오기 완료")
    print("입력변수:", len(features))
    print("분류 임계값:", threshold)

    # 2. 최신 기상청 예보 수집
    api_key = st.secrets["KMA_API_KEY"]
    base_date, base_time = latest_base_time()

    print("\n예보 발표 기준:", base_date, base_time)

    results = []

    for station, (lat, lon) in STATIONS.items():
        data = fetch_station(
            station, lat, lon,
            api_key, base_date, base_time
        )

        results.append(data)
        print(f"{station}: {len(data)}개 예보 시점")

    forecast = pd.concat(results, ignore_index=True)

    if forecast.empty:
        raise RuntimeError("기상청 예보가 비어 있습니다.")

    # 3. 모델 입력 형태로 변환
    future = prepare_features(forecast, features)

    # 낮 시간대만 시험 예측
    future = future[
        future["forecast_datetime"]
        .dt.hour.between(10, 16)
    ].copy()

    if future.empty:
        raise RuntimeError(
            "낮 시간대에 해당하는 예보가 없습니다."
        )

    # 4. 기존 모델로 예측 (재학습 없음)
    future["risk_score"] = model.predict_proba(
        future[features]
    )[:, 1]

    future["risk_flag"] = (
        future["risk_score"] >= threshold
    ).astype(int)

    future["threshold"] = threshold
    future["model_version"] = "EXP-C005"

    future["forecast_issued_at"] = (
        f"{base_date} {base_time}"
    )

    output = future[[
        "forecast_datetime",
        "forecast_issued_at",
        "risk_score",
        "risk_flag",
        "threshold",
        "model_version"
    ]].sort_values("forecast_datetime")

    # 모든 작업에 성공한 뒤 파일 교체
    output_path = OUTPUT_DIR / "future_risk_forecast.csv"

    forecast.to_csv(
        FORECAST_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== 최신 미래 시험 예측 =====")
    print(
        output.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n예측 시점 수:", len(output))
    print("예측 저장 완료:", output_path)


if __name__ == "__main__":
    main()
