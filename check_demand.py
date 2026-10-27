
from pathlib import Path
import pandas as pd

FOLDER = Path("data/raw/demand")
files = sorted(FOLDER.glob("*.csv"))

def read_csv(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(
                path, encoding=enc, low_memory=False
            )
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 확인 필요: {path}")

print("발견된 CSV 파일 수:", len(files))

if len(files) != 6:
    print("주의: 예상한 파일은 6개입니다.")

datasets = []

for path in files:
    df = read_csv(path)

    print("\n" + "=" * 55)
    print("파일:", path.name)
    print("크기:", df.shape)
    print("컬럼:", df.columns.tolist())
    print("\n첫 2행:")
    print(df.head(2).to_string(index=False))
    print("\n마지막 2행:")
    print(df.tail(2).to_string(index=False))

    date_candidates = [
        c for c in df.columns
        if ("일자" in c or "날짜" in c or "거래일" in c)
    ]

    if date_candidates:
        date_col = date_candidates[0]
        dates = pd.to_datetime(
            df[date_col], errors="coerce"
        )

        print("\n날짜 컬럼:", date_col)
        print("시작:", dates.min())
        print("종료:", dates.max())
        print("날짜 변환 실패:", dates.isna().sum())
        print("중복 날짜:", dates.duplicated().sum())

        df["_date"] = dates
        df["_source"] = path.name
        datasets.append(df)

    hour_cols = [
        c for c in df.columns
        if str(c).strip().endswith("시")
        and str(c).strip()[:-1].isdigit()
    ]

    print("\n시간별 컬럼 수:", len(hour_cols))

    if hour_cols:
        values = (
            df[hour_cols]
            .apply(pd.to_numeric, errors="coerce")
            .stack()
        )

        print("시간별 값 개수:", len(values))
        print("최솟값:", values.min())
        print("중앙값:", values.median())
        print("최댓값:", values.max())
        print("값 0 이하:", (values <= 0).sum())
        print(
            "시간별 결측:",
            df[hour_cols].size - len(values)
        )

print("\n" + "=" * 55)
print("=== 파일 간 중복 날짜 확인 ===")

if datasets:
    combined = pd.concat(datasets, ignore_index=True)

    if "_date" in combined.columns:
        counts = combined.groupby("_date")["_source"].nunique()
        duplicated = counts[counts > 1]

        print("서로 다른 파일에 중복된 날짜 수:", len(duplicated))

        if len(duplicated) > 0:
            print("\n중복 날짜 예시:")
            print(duplicated.head(15).to_string())

        print("\n연도별 수록 행 수:")
        print(
            combined.groupby(
                combined["_date"].dt.year
            ).size().to_string()
        )

print("\n검사 완료")
