
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"인코딩 오류: {path}")

# 1. 풍력 데이터
wind_files = sorted((BASE / "wind").glob("*.csv"))
weather_files = sorted((BASE / "weather_wind").glob("OBS_ASOS_TIM_*.csv"))

assert len(wind_files) == 4, "풍력 CSV 파일 4개를 확인하세요."
assert len(weather_files) == 4, "기상 CSV 파일 4개를 확인하세요."

wind = pd.concat([read_csv(p) for p in wind_files], ignore_index=True)
wind["FUEL_CD"] = pd.to_numeric(wind["FUEL_CD"], errors="coerce")
wind = wind.loc[wind["FUEL_CD"] == 16].copy()

wind["trade_date"] = pd.to_datetime(
    wind["TRAD_YMD"].astype(str), errors="raise"
)
wind["trade_hour"] = pd.to_numeric(
    wind["TRAD_HH"], errors="raise"
).astype(int)
wind["wind_mwh"] = pd.to_numeric(wind["AMGO"], errors="coerce")

assert wind["trade_hour"].between(1, 24).all()
assert not wind.duplicated(["trade_date", "trade_hour"]).any()

# 거래시간은 종료 시각 기준
wind["timestamp"] = wind["trade_date"] + pd.to_timedelta(
    wind["trade_hour"], unit="h"
)

expected_hours = pd.date_range(
    "2020-01-01 01:00", "2024-01-01 00:00", freq="h"
)
assert len(wind) == len(expected_hours)
assert set(wind["timestamp"]) == set(expected_hours)

# 2. 기상 데이터
weather = pd.concat(
    [read_csv(p) for p in weather_files],
    ignore_index=True
)

weather["timestamp"] = pd.to_datetime(
    weather["일시"], errors="raise"
)
weather["지점"] = pd.to_numeric(
    weather["지점"], errors="raise"
).astype(int)

weather = weather.loc[
    weather["지점"].isin([184, 185, 188, 189])
].copy()

assert not weather.duplicated(["timestamp", "지점"]).any()

# 기본 입력변수
weather_columns = {
    "풍속(m/s)": "wind_speed",
    "풍향(16방위)": "wind_direction",
    "기온(°C)": "temperature",
    "습도(%)": "humidity",
    "현지기압(hPa)": "pressure"
}

parts = []

# 3. 관측지점별 기상변수를 가로로 결합
for station, name in [
    (184, "jeju"),
    (185, "gosan"),
    (188, "seongsan"),
    (189, "seogwipo")
]:
    sub = weather.loc[
        weather["지점"] == station,
        ["timestamp"] + list(weather_columns)
    ].copy()

    sub = sub.rename(columns={
        old: f"{name}_{new}"
        for old, new in weather_columns.items()
    })

    parts.append(sub.set_index("timestamp"))

weather_wide = pd.concat(parts, axis=1).reset_index()

# 4. 시간 기준 병합
merged = wind[
    ["trade_date", "trade_hour", "timestamp", "wind_mwh"]
].merge(
    weather_wide,
    on="timestamp",
    how="left",
    validate="one_to_one"
)

# 5. 월별 시간 기록과 결측치 검사
merged["year"] = merged["trade_date"].dt.year
merged["month"] = merged["trade_date"].dt.month

print("\n=== 연도별 데이터 수 ===")
print(merged.groupby("year").size())

print("\n=== 결측치 개수 ===")
print(merged.isna().sum().to_string())

print("\n=== 발전량 통계 ===")
print(merged.groupby("year")["wind_mwh"].agg(
    ["count", "mean", "min", "max"]
))

# 6. 결과 저장
output_path = OUT / "wind_weather_2020_2023.csv"
merged.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n병합 파일 저장:", output_path)
print("전체 행 수:", len(merged))
