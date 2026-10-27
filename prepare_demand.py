
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/demand")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    path = RAW / name
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"파일 인코딩 오류: {name}")

    df = df.rename(columns={"거래일자": "date", "날짜": "date"})
    df["date"] = pd.to_datetime(df["date"])

    hours = [f"{h}시" for h in range(1, 25)]
    assert set(hours).issubset(df.columns)
    assert not df["date"].duplicated().any()

    df[hours] = df[hours].apply(pd.to_numeric, errors="coerce")
    assert not df[hours].isna().any().any()

    return df[["date"] + hours].copy()

old = load("한국전력거래소_시간별 제주전력수요_20230228.csv")
middle = load("한국전력거래소_시간별 제주전력수요_20230930.csv")
new = load("한국전력거래소_시간별 제주전력수요_20240331.csv")

hours = [f"{h}시" for h in range(1, 25)]

# 1. 2023년 9월 중복자료 비교
overlap = middle.merge(new, on="date", suffixes=("_old", "_new"))
overlap = overlap[overlap["date"].dt.strftime("%Y-%m") == "2023-09"]

if len(overlap) != 30:
    raise ValueError("2023년 9월 중복 일수가 30일이 아닙니다.")

differences = []
for h in hours:
    a = overlap[f"{h}_old"].to_numpy() / 1000
    b = overlap[f"{h}_new"].to_numpy()
    differences.extend(np.abs(a - b))

max_diff = max(differences)
print("9월 자료 최대 단위변환 차이(MWh):", round(max_diff, 6))

# 수정정산 차이가 있을 수 있으므로 무조건 동일하다고 가정하지 않음
if max_diff > 1:
    raise ValueError(
        "9월 두 파일의 차이가 1 MWh를 초과합니다. "
        "집계 기준이나 수정정산 여부 확인 필요."
    )

# 2. 기간별 원본 선택
a = old[
    old["date"].between("2021-01-01", "2023-02-28")
].copy()
b = middle[
    middle["date"].between("2023-03-01", "2023-08-31")
].copy()
c = new[
    new["date"].between("2023-09-01", "2023-12-31")
].copy()

# 2023년 9월 이전 자료는 kWh -> MWh
a[hours] = a[hours] / 1000
b[hours] = b[hours] / 1000

combined = pd.concat([a, b, c], ignore_index=True)
combined = combined.sort_values("date")

# 3. 날짜 검증
expected = pd.date_range("2021-01-01", "2023-12-31")
assert len(combined) == len(expected)
assert combined["date"].is_unique
assert set(combined["date"]) == set(expected)

# 4. 일별 가로형 -> 시간별 세로형
hourly = combined.melt(
    id_vars="date",
    value_vars=hours,
    var_name="hour_label",
    value_name="demand_mwh"
)
hourly["trade_hour"] = (
    hourly["hour_label"].str.replace("시", "", regex=False)
    .astype(int)
)
hourly = hourly.rename(columns={"date": "trade_date"})
hourly = hourly[["trade_date", "trade_hour", "demand_mwh"]]
hourly = hourly.sort_values(
    ["trade_date", "trade_hour"]
).reset_index(drop=True)

assert len(hourly) == 26280
assert hourly["demand_mwh"].notna().all()
assert (hourly["demand_mwh"] > 0).all()
assert not hourly.duplicated(["trade_date", "trade_hour"]).any()

print("\n=== 연도별 전력수요 검증 ===")
print(hourly.groupby(hourly["trade_date"].dt.year)["demand_mwh"]
      .agg(["count", "mean", "min", "max"]))

print("\n전체 행 수:", len(hourly))
print("중복 시간:", hourly.duplicated(
    ["trade_date", "trade_hour"]
).sum())

path = OUT / "jeju_demand_hourly_2021_2023.csv"
hourly.to_csv(path, index=False, encoding="utf-8-sig")
print("\n저장 완료:", path)
