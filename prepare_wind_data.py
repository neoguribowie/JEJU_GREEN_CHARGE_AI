
from pathlib import Path
import pandas as pd

BASE = Path("data/raw")

def read_csv(path):
    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 확인 필요: {path}")

# 1. 연도별 풍력 데이터 불러오기
wind_files = sorted(
    (BASE / "wind").glob("제주발전기 연료원 시간대별_*.csv")
)

# 2. 연도별 기상자료 불러오기
weather_files = sorted(
    (BASE / "weather_wind").glob("OBS_ASOS_TIM_*.csv")
)

print("풍력 파일 수:", len(wind_files))
print("기상 파일 수:", len(weather_files))
print("\n=== 발견된 풍력 파일 ===")
for f in wind_files:
    print(f)

print("\n=== 발견된 기상 파일 ===")
for f in weather_files:
    print(f)

if len(wind_files) != 4 or len(weather_files) != 4:
    raise ValueError("2020~2023년 파일이 각각 4개인지 확인하세요.")

wind = pd.concat(
    [read_csv(f) for f in wind_files],
    ignore_index=True
)

weather = pd.concat(
    [read_csv(f) for f in weather_files],
    ignore_index=True
)

# 3. 풍력만 추출
wind["FUEL_CD"] = pd.to_numeric(
    wind["FUEL_CD"], errors="coerce"
)
wind = wind[wind["FUEL_CD"] == 16].copy()

# 4. 컬럼 확인
print("\n=== 풍력 데이터 컬럼 ===")
print(wind.columns.tolist())

print("\n=== 기상 데이터 컬럼 ===")
print(weather.columns.tolist())

# 5. 풍력 시간 확인
wind["TRAD_YMD"] = pd.to_datetime(
    wind["TRAD_YMD"].astype(str),
    errors="coerce"
)

wind["TRAD_HH"] = pd.to_numeric(
    wind["TRAD_HH"], errors="coerce"
)

wind["AMGO"] = pd.to_numeric(
    wind["AMGO"], errors="coerce"
)

print("\n=== 풍력 데이터 확인 ===")
print("전체 행:", len(wind))
print("시작:", wind["TRAD_YMD"].min())
print("종료:", wind["TRAD_YMD"].max())
print("거래시간:", sorted(wind["TRAD_HH"].dropna().unique()))
print("발전량 결측:", wind["AMGO"].isna().sum())

print(
    "날짜·시간 중복:",
    wind.duplicated(["TRAD_YMD", "TRAD_HH"]).sum()
)

print("\n=== 풍력 첫 5행 ===")
print(
    wind[["TRAD_YMD", "TRAD_HH", "AMGO"]]
    .head()
    .to_string(index=False)
)

# 6. 기상자료 시간과 지점 확인
print("\n=== 기상 첫 5행 ===")
print(weather.head().to_string(index=False))

print("\n=== 기상 관측지점 ===")
for col in ["지점", "지점명"]:
    if col in weather.columns:
        print(col, weather[col].unique())

print("\n=== 기상 시간 범위 ===")
if "일시" in weather.columns:
    weather["일시"] = pd.to_datetime(
        weather["일시"], errors="coerce"
    )
    print("시작:", weather["일시"].min())
    print("종료:", weather["일시"].max())
    print("시간 파싱 실패:", weather["일시"].isna().sum())
