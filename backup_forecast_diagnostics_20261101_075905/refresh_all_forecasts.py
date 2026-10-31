
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from refresh_c005_forecast import prepare_features

ROOT = Path(__file__).resolve().parent

FORECAST_PATH = ROOT / (
    "data/processed/jeju_weather_forecast_4stations.csv"
)
C005_PATH = ROOT / (
    "results/curtailment_c005/future_risk_forecast.csv"
)
D002_PATH = ROOT / (
    "results/demand_d002/future_demand_forecast.csv"
)
C006_PATH = ROOT / (
    "results/curtailment_c006/future_risk_forecast.csv"
)

D002_MODEL_PATH = ROOT / "models/demand_rf_d002.pkl"
C006_MODEL_PATH = ROOT / "models/curtailment_rf_c006.pkl"

# ---------------------------------------
# 1. 기존 스크립트로 최신 예보 수집 + C005 갱신
# ---------------------------------------
print("STEP 1: 기상청 예보 수집 및 C005 갱신")

subprocess.run(
    [sys.executable, str(ROOT / "refresh_c005_forecast.py")],
    cwd=ROOT,
    check=True
)

# ---------------------------------------
# 2. 기존 학습 모델 불러오기
# ---------------------------------------
print("\nSTEP 2: 저장된 D002 및 C006 모델 불러오기")

d002_saved = joblib.load(D002_MODEL_PATH)
c006_saved = joblib.load(C006_MODEL_PATH)

d002_model = d002_saved["model"]
c006_model = c006_saved["model"]

d002_features = list(d002_saved["features"])
c006_features = list(c006_saved["features"])

threshold = float(c006_saved["threshold"])

# ---------------------------------------
# 3. 동일한 예보 자료로 입력변수 생성
# ---------------------------------------
forecast = pd.read_csv(FORECAST_PATH)
c005 = pd.read_csv(C005_PATH)

if forecast.empty or c005.empty:
    raise ValueError("최신 예보 또는 C005 결과가 비어 있습니다.")

# C006의 기상 입력변수 24개
weather_features = [
    col for col in c006_features
    if col != "predicted_demand_mwh"
]

if len(weather_features) != 24:
    raise ValueError("C006의 기상 입력변수 구성을 확인하세요.")

future = prepare_features(forecast, weather_features)

# D002에서 추가했던 요일·주말 변수 생성
trade_date = pd.to_datetime(future["trade_date"])
weekday = trade_date.dt.dayofweek

future["weekday_sin"] = np.sin(
    2 * np.pi * weekday / 7
)
future["weekday_cos"] = np.cos(
    2 * np.pi * weekday / 7
)
future["is_weekend"] = (weekday >= 5).astype(int)

# C005와 동일한 낮 시간대만 사용
future = future[
    future["forecast_datetime"].dt.hour.between(10, 16)
].copy()

future = future.sort_values("forecast_datetime")

if future.empty:
    raise ValueError("예측 가능한 낮 시간대가 없습니다.")

missing = [
    col for col in d002_features
    if col not in future.columns
]
if missing:
    raise ValueError(f"D002 입력변수 누락: {missing}")

if future[d002_features].isna().any().any():
    raise ValueError("D002 미래 입력변수에 결측값이 있습니다.")

# ---------------------------------------
# 4. D002 전력수요 예측
# ---------------------------------------
print("\nSTEP 3: D002 미래 전력수요 예측")

future["predicted_demand_mwh"] = d002_model.predict(
    future[d002_features]
)

# ---------------------------------------
# 5. C006 출력제어 위험 점수 예측
# ---------------------------------------
print("STEP 4: C006 미래 위험 점수 예측")

missing = [
    col for col in c006_features
    if col not in future.columns
]
if missing:
    raise ValueError(f"C006 입력변수 누락: {missing}")

if future[c006_features].isna().any().any():
    raise ValueError("C006 미래 입력변수에 결측값이 있습니다.")

future["risk_score"] = c006_model.predict_proba(
    future[c006_features]
)[:, 1]

future["risk_flag"] = (
    future["risk_score"] >= threshold
).astype(int)

# ---------------------------------------
# 6. 발표 기준 및 예측 시각 일치 확인
# ---------------------------------------
issued_values = forecast["issued_at"].dropna().astype(str).unique()

if len(issued_values) != 1:
    raise ValueError("예보 발표 기준이 한 개가 아닙니다.")

issued_at = issued_values[0]

c005_issued = c005[
    "forecast_issued_at"
].astype(str).unique()

if len(c005_issued) != 1 or c005_issued[0] != issued_at:
    raise ValueError(
        "C005와 신규 예보의 발표 기준이 다릅니다."
    )

c005_times = set(
    pd.to_datetime(c005["forecast_datetime"])
)
future_times = set(future["forecast_datetime"])

if c005_times != future_times:
    raise ValueError(
        "C005·D002·C006의 예측 시점이 다릅니다."
    )

# ---------------------------------------
# 7. 결과 CSV 저장
# ---------------------------------------
future["forecast_issued_at"] = issued_at

d002_output = future[[
    "forecast_datetime",
    "forecast_issued_at",
    "predicted_demand_mwh"
]].copy()

d002_output["model_version"] = "EXP-D002"

c006_output = future[[
    "forecast_datetime",
    "forecast_issued_at",
    "predicted_demand_mwh",
    "risk_score",
    "risk_flag"
]].copy()

c006_output["threshold"] = threshold
c006_output["model_version"] = "EXP-C006"

D002_PATH.parent.mkdir(parents=True, exist_ok=True)
C006_PATH.parent.mkdir(parents=True, exist_ok=True)

d002_output.to_csv(
    D002_PATH, index=False, encoding="utf-8-sig"
)

c006_output.to_csv(
    C006_PATH, index=False, encoding="utf-8-sig"
)

print("\n===== 통합 예측 갱신 완료 =====")
print("발표 기준:", issued_at)
print("예측 시점 수:", len(future))
print("C005:", C005_PATH)
print("D002:", D002_PATH)
print("C006:", C006_PATH)

print("\n===== D002·C006 예측 미리보기 =====")
print(
    c006_output.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)
