
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    average_precision_score,
    ConfusionMatrixDisplay
)

# 1. 경로
weather_path = Path("data/processed/wind_weather_2020_2023.csv")
target_path = Path("data/processed/curtailment_hourly_2021_2023.csv")
result_dir = Path("results/curtailment")
model_dir = Path("models")

result_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

weather = pd.read_csv(weather_path)
target = pd.read_csv(target_path)

for df in [weather, target]:
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["trade_hour"] = pd.to_numeric(df["trade_hour"]).astype(int)

# 2. 날짜·거래시간 기준 병합
keys = ["trade_date", "trade_hour"]

assert not weather.duplicated(keys).any()
assert not target.duplicated(keys).any()

# 정답이 있는 2021~2023년만 사용
weather = weather[
    weather["trade_date"].dt.year.between(2021, 2023)
].copy()

df = target.merge(
    weather,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True,
    suffixes=("", "_weather")
)

assert (df["_merge"] == "both").all(), "기상자료 병합 누락 발견"
assert len(df) == 26280, "전체 시간 수를 확인하세요"

# 3. 입력변수
stations = ["jeju", "gosan", "seongsan", "seogwipo"]
features = []

for station in stations:
    for variable in [
        "wind_speed", "temperature",
        "humidity", "pressure"
    ]:
        features.append(f"{station}_{variable}")

    # 풍향의 주기적 특성을 Sin/Cos로 변환
    direction = pd.to_numeric(
        df[f"{station}_wind_direction"],
        errors="coerce"
    )
    angle = np.deg2rad(direction)
    sin_col = f"{station}_direction_sin"
    cos_col = f"{station}_direction_cos"

    df[sin_col] = np.sin(angle)
    df[cos_col] = np.cos(angle)

    features.extend([sin_col, cos_col])

# 시간·월 주기 인코딩
df["hour_sin"] = np.sin(
    2 * np.pi * df["trade_hour"] / 24
)
df["hour_cos"] = np.cos(
    2 * np.pi * df["trade_hour"] / 24
)
df["month_sin"] = np.sin(
    2 * np.pi * df["trade_date"].dt.month / 12
)
df["month_cos"] = np.cos(
    2 * np.pi * df["trade_date"].dt.month / 12
)

features.extend([
    "hour_sin", "hour_cos",
    "month_sin", "month_cos"
])

# 발전량(wind_mwh) 등 실제 결과 변수는 입력에서 제외
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[features] = df[features].replace(
    [np.inf, -np.inf], np.nan
)

# 4. 시간순 학습/평가 분리
train = df[df["trade_date"].dt.year.isin([2021, 2022])]
test = df[df["trade_date"].dt.year == 2023]

X_train = train[features]
y_train = train["curtailment"].astype(int)
X_test = test[features]
y_test = test["curtailment"].astype(int)

print("학습 데이터:", len(train))
print("평가 데이터:", len(test))
print("학습 발생 건수:", y_train.sum())
print("평가 발생 건수:", y_test.sum())

# 5. Random Forest 모델
# 불균형 자료를 고려해 클래스 가중치를 사전에 고정
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

model.fit(X_train, y_train)

# 분류 기준 0.5 고정
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# 6. 성능 평가
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)
ap = average_precision_score(y_test, prob)
cm = confusion_matrix(y_test, pred, labels=[0, 1])

print("\n===== EXP-C001 결과 =====")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"PR-AUC(AP): {ap:.4f}")
print("\n혼동행렬 [실제 0, 실제 1]:")
print(cm)
print("\n", classification_report(
    y_test, pred, zero_division=0
))

# 7. 결과 파일 저장
summary = pd.DataFrame([{
    "experiment": "EXP-C001",
    "train_years": "2021-2022",
    "test_year": 2023,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "average_precision": ap,
    "TN": cm[0, 0],
    "FP": cm[0, 1],
    "FN": cm[1, 0],
    "TP": cm[1, 1],
}])

summary.to_csv(
    result_dir / "metrics.csv",
    index=False, encoding="utf-8-sig"
)

predictions = test[
    ["trade_date", "trade_hour", "curtailment"]
].copy()
predictions["predicted"] = pred
predictions["probability"] = prob

predictions.to_csv(
    result_dir / "predictions_2023.csv",
    index=False, encoding="utf-8-sig"
)

# 월별 평가
monthly = []
for month, group in predictions.groupby(
    predictions["trade_date"].dt.month
):
    monthly.append({
        "month": month,
        "actual_events": int(group["curtailment"].sum()),
        "predicted_events": int(group["predicted"].sum()),
        "precision": precision_score(
            group["curtailment"], group["predicted"],
            zero_division=0
        ),
        "recall": recall_score(
            group["curtailment"], group["predicted"],
            zero_division=0
        ),
        "f1": f1_score(
            group["curtailment"], group["predicted"],
            zero_division=0
        )
    })

monthly_df = pd.DataFrame(monthly)
monthly_df.to_csv(
    result_dir / "monthly_metrics.csv",
    index=False, encoding="utf-8-sig"
)

# 혼동행렬 그림
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No curtailment", "Curtailment"]
).plot(cmap="Blues", values_format="d")

plt.title("EXP-C001 Confusion Matrix (2023)")
plt.tight_layout()
plt.savefig(
    result_dir / "confusion_matrix.png",
    dpi=150
)
plt.close()

joblib.dump({
    "model": model,
    "features": features,
    "target": "curtailment",
    "threshold": 0.5,
    "train_years": [2021, 2022]
}, model_dir / "curtailment_rf_c001.pkl")

print("\n월별 평가:")
print(monthly_df.round(4).to_string(index=False))
print("\n결과 저장 완료:", result_dir)
