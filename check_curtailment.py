
from pathlib import Path
import pandas as pd

BASE = Path("data/raw/curtailment")

files = {
    "태양광": BASE / "제주 태양광 출력제어횟수.csv",
    "풍력": BASE / "제주 풍력 출력제어횟수 및 제어량.csv",
}

def read_csv(path):
    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(f"인코딩 확인 필요: {path}")

for name, path in files.items():
    print(f"\n{'=' * 15} {name} {'=' * 15}")

    if not path.exists():
        print("파일을 찾을 수 없습니다:", path)
        continue

    df = read_csv(path)

    print("\n데이터 크기:", df.shape)

    print("\n컬럼명:")
    print(df.columns.tolist())

    print("\n첫 5행:")
    print(df.head(5).to_string(index=False))

    print("\n마지막 3행:")
    print(df.tail(3).to_string(index=False))

    print("\n컬럼별 결측치:")
    print(df.isna().sum().to_string())

print("\n원본 데이터 확인 완료")
