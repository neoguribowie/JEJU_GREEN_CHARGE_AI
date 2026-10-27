
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/curtailment")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"파일 인코딩 확인 필요: {path}")

def make_hourly(path, kind):
    df = read_csv(path)
    df["일자"] = pd.to_datetime(df["일자"], errors="raise")
    df = df[df["일자"].dt.year.between(2021, 2023)].copy()

    if df["일자"].duplicated().any():
        raise ValueError(f"{kind}: 중복 날짜 발견")

    hour_cols = [f"{h}시" for h in range(1, 25)]

    # 자료에 등장하는 실제 값 확인
    if df[hour_cols].isna().any().any():
        raise ValueError(f"{kind}: 시간별 빈칸 확인 필요")

    long = df.melt(
        id_vars=["일자"],
        value_vars=hour_cols,
        var_name="hour_col",
        value_name="raw_value"
    )

    long["trade_hour"] = (
        long["hour_col"].str.replace("시", "", regex=False)
        .astype(int)
    )

    if kind == "solar":
        raw = long["raw_value"].astype(str).str.strip()
        valid = raw.isin(["출력제어", "0", "0.0"])
        if not valid.all():
            print("태양광 미확인 값:")
            print(raw[~valid].value_counts())
            raise ValueError("태양광 시간별 값 확인 필요")
        long["occurred"] = raw.eq("출력제어").astype(int)

    else:
        amount = pd.to_numeric(long["raw_value"], errors="coerce")
        if amount.isna().any() or (amount < 0).any():
            raise ValueError("풍력 제어량에 비정상 값 발견")
        long["occurred"] = amount.gt(0).astype(int)

    return long[["일자", "trade_hour", "occurred"]].rename(
        columns={"occurred": f"{kind}_curtailment"}
    )

solar = make_hourly(
    RAW / "제주 태양광 출력제어횟수.csv", "solar"
)
wind = make_hourly(
    RAW / "제주 풍력 출력제어횟수 및 제어량.csv", "wind"
)

# 2021~2023년 모든 날짜 × 거래시간 1~24 생성
days = pd.date_range("2021-01-01", "2023-12-31", freq="D")
calendar = pd.MultiIndex.from_product(
    [days, range(1, 25)],
    names=["일자", "trade_hour"]
).to_frame(index=False)

# 출력제어가 기록되지 않은 시간은 0
merged = calendar.merge(
    solar, on=["일자", "trade_hour"],
    how="left", validate="one_to_one"
).merge(
    wind, on=["일자", "trade_hour"],
    how="left", validate="one_to_one"
)

merged[["solar_curtailment", "wind_curtailment"]] = (
    merged[["solar_curtailment", "wind_curtailment"]]
    .fillna(0).astype(int)
)

merged["curtailment"] = (
    (merged["solar_curtailment"] == 1) |
    (merged["wind_curtailment"] == 1)
).astype(int)

merged = merged.rename(columns={"일자": "trade_date"})
merged["year"] = merged["trade_date"].dt.year
merged["month"] = merged["trade_date"].dt.month

# 전력거래량과 동일한 거래시간 종료 기준 timestamp
# 최종 기상자료 병합 전 시간 정의 확인 필요
merged["timestamp"] = (
    merged["trade_date"] +
    pd.to_timedelta(merged["trade_hour"], unit="h")
)

print("\n=== 연도별 출력제어 발생 시간 ===")
print(merged.groupby("year")[
    ["solar_curtailment", "wind_curtailment", "curtailment"]
].sum())

print("\n=== 정답 데이터 개수 ===")
print(merged["curtailment"].value_counts())

output = OUT / "curtailment_hourly_2021_2023.csv"
merged.to_csv(output, index=False, encoding="utf-8-sig")

print("\n저장 완료:", output)
print("전체 행 수:", len(merged))
