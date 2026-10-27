
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. 실제 CSV 데이터 읽기
file_path = Path("data/raw/sun/제주#1.csv")

for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        break
    except UnicodeDecodeError:
        continue

# 2. 첫 실험은 금악태양광 발전소만 사용
df = df[df["발전기코드"] == "AJ85"].copy()

# 3. 날짜 및 시간변수 생성
df["거래일"] = pd.to_datetime(df["거래일"])
df["월"] = df["거래일"].dt.month
df = df.sort_values(["거래일", "시간"])

# 4. AI 입력변수와 정답 설정
features = [
    "월", "시간", "기온", "강수량",
    "풍속", "풍향", "습도",
    "일조", "일사", "지면온도"
]

target = "계량전력량"

# 5. 시간순으로 학습·검증 데이터 분리
train = df[df["거래일"] < "2025-01-01"]
test = df[
    (df["거래일"] >= "2025-01-01") &
    (df["거래일"] < "2026-01-01")
]

if train.empty or test.empty:
    raise ValueError("학습 또는 테스트 데이터가 비어 있습니다.")

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# 6. Random Forest 학습
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 7. 실제값과 예측값 비교
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("=== Random Forest 학습 완료 ===")
print("학습 데이터:", len(train), "행")
print("테스트 데이터:", len(test), "행")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# 8. 결과 저장
Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

joblib.dump(model, "models/solar_rf_aj85.pkl")

result = test[["거래일", "시간"]].copy()
result["actual"] = y_test.to_numpy()
result["predicted"] = predictions
result.to_csv(
    "results/solar_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

# 9. 처음 168시간(7일) 그래프
plt.figure(figsize=(12, 5))
plt.plot(result["actual"].iloc[:168].to_numpy(),
         label="Actual")
plt.plot(result["predicted"].iloc[:168].to_numpy(),
         label="Predicted")

plt.title("Solar Generation - Random Forest")
plt.xlabel("Hour")
plt.ylabel("Metered energy (check source unit)")
plt.legend()
plt.tight_layout()

plt.savefig("results/solar_prediction.png", dpi=150)
plt.close()

print("모델 및 그래프 저장 완료!")
