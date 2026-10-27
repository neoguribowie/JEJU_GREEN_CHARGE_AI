
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

D001 = ROOT / "results/demand_d001"
D002 = ROOT / "results/demand_d002"
C005 = ROOT / "results/curtailment_c005"
C006 = ROOT / "results/curtailment_c006"


def read_csv(path):
    if not path.exists():
        st.warning(f"결과 파일이 없습니다: {path.name}")
        return None
    return pd.read_csv(path)


def render_power_prediction():
    """제주 전력수요 예측 화면."""

    st.header("📊 제주 전력수요 AI 분석")
    st.caption("EXP-D002 · Random Forest · 기상예보 기반 시험 예측")

    path = D002 / "future_demand_forecast.csv"
    df = read_csv(path)

    if df is None:
        return

    df["forecast_datetime"] = pd.to_datetime(
        df["forecast_datetime"], errors="coerce"
    )
    df["predicted_demand_mwh"] = pd.to_numeric(
        df["predicted_demand_mwh"], errors="coerce"
    )
    df = df.dropna(
        subset=["forecast_datetime", "predicted_demand_mwh"]
    ).sort_values("forecast_datetime")

    if df.empty:
        st.warning("표시할 전력수요 예측 결과가 없습니다.")
        return

    issued_at = str(df["forecast_issued_at"].iloc[0])
    st.caption(f"기상예보 발표 기준: {issued_at}")

    now_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    upcoming = df[df["forecast_datetime"] > now_kst]

    if upcoming.empty:
        st.warning(
            "저장된 예측 시간대가 모두 지났습니다. "
            "아래 그래프는 과거에 생성한 시험 예측 기록입니다."
        )
        display_df = df
    else:
        display_df = upcoming

        issued = pd.to_datetime(
            issued_at, format="%Y%m%d %H%M", errors="coerce"
        )
        if pd.isna(issued) or (
            now_kst - issued
        ).total_seconds() > 12 * 3600:
            st.warning(
                "기상예보 발표 기준이 오래되었습니다. "
                "최신 예보로 갱신이 필요합니다."
            )

    display_df = display_df.copy()
    display_df["date"] = display_df["forecast_datetime"].dt.date

    selected_date = st.selectbox(
        "예측 날짜 선택",
        sorted(display_df["date"].unique()),
        format_func=lambda d: d.strftime("%Y년 %m월 %d일"),
        key="power_forecast_date"
    )

    day = display_df[
        display_df["date"] == selected_date
    ].copy()

    average_demand = day["predicted_demand_mwh"].mean()

    st.metric(
        "선택 날짜 예측 시간대 평균 전력수요",
        f"{average_demand:,.1f} MWh"
    )

    chart = day.set_index("forecast_datetime")[
        ["predicted_demand_mwh"]
    ]

    st.line_chart(
        chart,
        y_label="시간당 전력수요량 (MWh)"
    )

    st.caption(
        "해당 그래프는 제주 발전단 전력수요량의 AI 예측값입니다. "
        "실제 전력수요나 잉여전력량을 나타내지 않습니다."
    )

    with st.expander("시간별 예측 수치 확인"):
        table = day[[
            "forecast_datetime", "predicted_demand_mwh"
        ]].copy()
        table.columns = ["예측 시각", "예측 전력수요 (MWh)"]
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.info(
        "전력수요가 낮다는 이유만으로 재생에너지 "
        "출력제어가 발생한다고 판단할 수는 없습니다."
    )


def render_model_comparison():
    """모델 개선 실험 결과 비교 화면."""

    st.header("🧠 AI 모델 성능 비교")
    st.caption("실제 2023년 자료를 이용한 과거 평가 결과")

    st.subheader("① 제주 전력수요 예측 모델")

    d001 = read_csv(D001 / "metrics_2023.csv")
    d002 = read_csv(D002 / "metrics_2023.csv")

    if d001 is not None and d002 is not None:
        a = d001.copy()
        b = d002.copy()

        a["model"] = "D001"
        b["model"] = "D002"

        comparison = pd.concat([a, b], ignore_index=True)

        comparison["period"] = comparison["period"].replace({
            "all_hours": "전체 시간",
            "10_to_16": "낮 10~16시"
        })

        st.dataframe(
            comparison[[
                "model", "period", "MAE_MWh",
                "RMSE_MWh", "R2"
            ]].rename(columns={
                "model": "모델",
                "period": "평가 시간",
                "MAE_MWh": "MAE (MWh)",
                "RMSE_MWh": "RMSE (MWh)",
                "R2": "R²"
            }),
            hide_index=True,
            use_container_width=True
        )

        daytime = comparison[
            comparison["period"] == "낮 10~16시"
        ].set_index("model")

        st.markdown("**낮 시간 MAE 비교**")
        st.bar_chart(
            daytime[["MAE_MWh"]],
            y_label="MAE (MWh)"
        )

        st.caption(
            "D002는 D001에 요일·주말 변수를 추가한 모델입니다. "
            "2023년 낮 시간대의 예측 오차가 감소했습니다."
        )

    st.divider()
    st.subheader("② 출력제어 위험 분류모델")

    c005 = read_csv(C005 / "metrics_2023.csv")
    c006 = read_csv(C006 / "metrics_2023.csv")

    if c005 is not None and c006 is not None:
        a = c005.copy()
        b = c006.copy()

        a["model"] = "C005"
        b["model"] = "C006"

        comparison = pd.concat([a, b], ignore_index=True)

        # 동일한 임계값 0.05만 비교
        comparison["threshold"] = pd.to_numeric(
            comparison["threshold"], errors="coerce"
        )

        comparison = comparison[
            (comparison["threshold"] - 0.05).abs() < 1e-9
        ].copy()

        if comparison.empty:
            st.warning("임계값 0.05에 해당하는 결과가 없습니다.")
            return

        st.dataframe(
            comparison[[
                "model", "precision", "recall",
                "f1", "average_precision"
            ]].rename(columns={
                "model": "모델",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1-score",
                "average_precision": "AP"
            }),
            hide_index=True,
            use_container_width=True
        )

        chart_data = comparison.set_index("model")[
            ["precision", "recall", "f1"]
        ]

        st.markdown("**Precision · Recall · F1 비교**")
        st.bar_chart(chart_data)

        st.caption(
            "C005는 기상정보 기반 모델, "
            "C006은 예측 전력수요를 추가한 모델입니다. "
            "두 모델의 학습 기간은 완전히 동일하지 않습니다."
        )

        st.info(
            "임계값 0.05에서 C005의 Recall과 F1이 "
            "근소하게 높아 현재 기본 모델로 유지합니다. "
            "모든 수치는 과거 자료 기반 사후 평가 결과입니다."
        )

def render_future_risk_comparison():
    """C005와 C006의 미래 위험 점수 비교."""

    st.subheader("🔬 AI 모델별 미래 위험 점수")

    c005 = read_csv(C005 / "future_risk_forecast.csv")
    c006 = read_csv(C006 / "future_risk_forecast.csv")

    if c005 is None or c006 is None:
        return

    for df in [c005, c006]:
        df["forecast_datetime"] = pd.to_datetime(
            df["forecast_datetime"], errors="coerce"
        )
        df["risk_score"] = pd.to_numeric(
            df["risk_score"], errors="coerce"
        )

    # 같은 예보 기준인지 확인
    issued_005 = c005["forecast_issued_at"].astype(str).unique()
    issued_006 = c006["forecast_issued_at"].astype(str).unique()

    if (
        len(issued_005) != 1
        or len(issued_006) != 1
        or issued_005[0] != issued_006[0]
    ):
        st.warning(
            "C005와 C006의 기상예보 발표 기준이 다릅니다. "
            "통합 예측 갱신을 다시 실행하세요."
        )
        return

    merged = c005[[
        "forecast_datetime", "risk_score"
    ]].merge(
        c006[["forecast_datetime", "risk_score"]],
        on="forecast_datetime",
        suffixes=("_c005", "_c006"),
        validate="one_to_one"
    )

    if merged.empty:
        st.warning("두 모델의 공통 예측 시간이 없습니다.")
        return

    merged = merged.dropna().sort_values("forecast_datetime")
    merged["date"] = merged["forecast_datetime"].dt.date

    st.caption(f"기상예보 발표 기준: {issued_005[0]}")

    date = st.selectbox(
        "비교 날짜 선택",
        sorted(merged["date"].unique()),
        format_func=lambda d: d.strftime("%Y년 %m월 %d일"),
        key="risk_comparison_date"
    )

    day = merged[merged["date"] == date].copy()

    chart_data = day.set_index("forecast_datetime")[[
        "risk_score_c005", "risk_score_c006"
    ]]

    st.line_chart(
        chart_data.rename(columns={
            "risk_score_c005": "C005 (기상 기반)",
            "risk_score_c006": "C006 (예측 수요 반영)"
        }),
        y_label="AI 위험 점수"
    )

    with st.expander("시간별 모델 비교 수치"):
        st.dataframe(
            day[[
                "forecast_datetime",
                "risk_score_c005",
                "risk_score_c006"
            ]].rename(columns={
                "forecast_datetime": "예측 시각",
                "risk_score_c005": "C005 점수",
                "risk_score_c006": "C006 점수"
            }),
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "두 모델의 점수는 실제 출력제어 발생 확률로 "
        "보정된 값이 아니며, 실제 잉여전력량을 의미하지 않습니다."
    )
