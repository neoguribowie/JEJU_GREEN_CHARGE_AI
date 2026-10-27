
import requests
import pandas as pd
import streamlit as st

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# 기상청 단기예보 API
API_URL = (
    "https://apis.data.go.kr/1360000/"
    "VilageFcstInfoService_2.0/getVilageFcst"
)

# 제주시 중심부 기상청 격자 좌표
NX = 52
NY = 38

# 단기예보 발표 기준시각
BASE_TIMES = [
    "0200", "0500", "0800", "1100",
    "1400", "1700", "2000", "2300"
]


def latest_base_time():
    """현재 시각 이전의 최근 발표 기준시각 선택."""

    now = datetime.now(ZoneInfo("Asia/Seoul"))

    # 발표 직후 데이터 반영 지연을 고려한 여유
    safe_now = now - timedelta(minutes=20)

    for day_shift in [0, 1]:
        day = safe_now.date() - timedelta(days=day_shift)

        for base in reversed(BASE_TIMES):
            candidate = datetime.strptime(
                f"{day:%Y%m%d}{base}",
                "%Y%m%d%H%M"
            ).replace(tzinfo=ZoneInfo("Asia/Seoul"))

            if candidate <= safe_now:
                return day.strftime("%Y%m%d"), base

    raise RuntimeError("예보 발표시각을 결정하지 못했습니다.")


def fetch_forecast():
    api_key = st.secrets["KMA_API_KEY"]

    base_date, base_time = latest_base_time()

    params = {
        "serviceKey": api_key,
        "pageNo": 1,
        "numOfRows": 2000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": NX,
        "ny": NY,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    try:
        data = response.json()
    except ValueError:
        print("응답 앞부분:", response.text[:500])
        raise RuntimeError(
            "JSON 응답이 아닙니다. 인증키나 API 응답을 확인하세요."
        )

    header = data["response"]["header"]

    if str(header["resultCode"]) != "00":
        raise RuntimeError(
            f"기상청 API 오류: {header}"
        )

    body = data["response"]["body"]
    items = body["items"]["item"]

    df = pd.DataFrame(items)

    if df.empty:
        raise RuntimeError("예보 데이터가 비어 있습니다.")

    # 시간별 데이터를 가로형 CSV로 변환
    forecast = df.pivot_table(
        index=["fcstDate", "fcstTime"],
        columns="category",
        values="fcstValue",
        aggfunc="first"
    ).reset_index()

    forecast.columns.name = None

    forecast["forecast_datetime"] = pd.to_datetime(
        forecast["fcstDate"] + forecast["fcstTime"],
        format="%Y%m%d%H%M"
    )

    forecast["issued_at"] = (
        f"{base_date} {base_time}"
    )

    forecast = forecast.sort_values(
        "forecast_datetime"
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / "jeju_weather_forecast.csv"
    )

    forecast.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("발표 기준:", base_date, base_time)
    print("예보 데이터 행 수:", len(forecast))
    print("제공 변수:", forecast.columns.tolist())

    print("\n기상예보 데이터 미리보기:")
    print(forecast.head(10).to_string(index=False))

    print("\n저장 완료:", output_path)


if __name__ == "__main__":
    fetch_forecast()
