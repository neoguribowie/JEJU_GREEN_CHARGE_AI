
from pathlib import Path
import pandas as pd

# 1. 제주 태양광 CSV 파일 찾기
data_folder = Path("data/raw/sun")
files = list(data_folder.glob("제주#1*.csv"))

if not files:
    raise FileNotFoundError(
        "data/raw 폴더에서 제주#1 CSV 파일을 찾지 못했습니다."
    )

file_path = files[0]

# 2. CSV 데이터 불러오기
for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError("CSV 파일 인코딩 확인이 필요합니다.")

# 3. 데이터 기본 정보 확인
print("=== JEJU GREEN CHARGE AI ===")
print("파일명:", file_path.name)
print("데이터 크기:", df.shape)

print("\n=== 컬럼명 ===")
print(df.columns.tolist())

print("\n=== 처음 5개 행 ===")
print(df.head().to_string())

print("\n=== 데이터 타입 ===")
print(df.dtypes)

print("\n=== 결측치 개수 ===")
print(df.isnull().sum())
