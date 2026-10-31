
import pandas as pd
from pathlib import Path

files = {
    2022: "results/curtailment_c005/predictions_2022_validation.csv",
    2023: "results/curtailment_c005/predictions_2023.csv",
}

thresholds = [0.10, 0.15, 0.20]

monthly_results = []
weekend_results = []

for year, filename in files.items():
    df = pd.read_csv(filename)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    daytime = df[df["trade_hour"].between(10, 16)].copy()

    for threshold in thresholds:
        candidates = daytime[
            daytime["probability"] >= threshold
        ].copy()

        top3 = (
            candidates.sort_values(
                ["trade_date", "probability", "trade_hour"],
                ascending=[True, False, True]
            )
            .groupby("trade_date")
            .head(3)
            .copy()
        )

        top3["month"] = top3["trade_date"].dt.month
        top3["is_weekend"] = top3["trade_date"].dt.dayofweek >= 5

        for month in range(1, 13):
            subset = top3[top3["month"] == month]
            hours = len(subset)
            hits = int(subset["curtailment"].sum())

            monthly_results.append({
                "year": year,
                "threshold": threshold,
                "month": month,
                "recommended_days": subset["trade_date"].nunique(),
                "recommended_hours": hours,
                "correct_hours": hits,
                "precision_pct": round(hits / hours * 100, 2) if hours else 0
            })

        for name, weekend_flag in [("평일", False), ("주말", True)]:
            subset = top3[top3["is_weekend"] == weekend_flag]
            hours = len(subset)
            hits = int(subset["curtailment"].sum())

            # 전체 해당 요일 일수
            all_dates = daytime[
                (daytime["trade_date"].dt.dayofweek >= 5) == weekend_flag
            ]["trade_date"].nunique()

            weekend_results.append({
                "year": year,
                "threshold": threshold,
                "day_type": name,
                "total_days": all_dates,
                "recommended_days": subset["trade_date"].nunique(),
                "recommended_hours": hours,
                "correct_hours": hits,
                "precision_pct": round(hits / hours * 100, 2) if hours else 0
            })

monthly = pd.DataFrame(monthly_results)
weekend = pd.DataFrame(weekend_results)

print("\n========== 월별 추천 가능 일수 ==========")
print(
    monthly.pivot_table(
        index=["year", "month"],
        columns="threshold",
        values="recommended_days"
    ).to_string()
)

print("\n========== 월별 추천 적중률(%) ==========")
print(
    monthly.pivot_table(
        index=["year", "month"],
        columns="threshold",
        values="precision_pct"
    ).to_string()
)

print("\n========== 평일 및 주말 분석 ==========")
print(weekend.to_string(index=False))

output = Path("results/curtailment_c005")
monthly.to_csv(
    output / "monthly_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)
weekend.to_csv(
    output / "weekend_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n분석 완료! CSV 파일 2개가 저장되었습니다.")
