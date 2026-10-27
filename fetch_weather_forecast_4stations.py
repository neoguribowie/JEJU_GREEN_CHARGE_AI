
import math
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from fetch_weather_forecast import (
    API_URL,
    latest_base_time
)

# 기존 ASOS 관측지점 부근 좌표
# 실제 예보는 가장 가까운 기상청 격자에서 조회됨
STATIONS = {
    "jeju": (33.5141, 126.5297),
    "gosan": (33.2938, 126.1628),
    "seongsan": (33.3868, 126.8802),
    "seogwipo": (33.2462, 126.5653),
}


def latlon_to_grid(lat, lon):
    """기상청 단기예보 위경도 -> 격자 변환."""
    RE, GRID = 6371.00877, 5.0
    SLAT1, SLAT2 = 30.0, 60.0
    OLON, OLAT = 126.0, 38.0
    XO, YO = 43.0, 136.0

    rad = math.pi / 180.0
    re = RE / GRID

    slat1 = SLAT1 * rad
    slat2 = SLAT2 * rad
    olon = OLON * rad
    olat = OLAT * rad

    sn = math.tan(math.pi / 4 + slat2 / 2)
    sn /= math.tan(math.pi / 4 + slat1 / 2)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)

    sf = math.tan(math.pi / 4 + slat1 / 2)
    sf = (sf ** sn) * math.cos(slat1) / sn

    ro = math.tan(math.pi / 4 + olat / 2)
    ro = re * sf / (ro ** sn)

    ra = math.tan(math.pi / 4 + lat * rad / 2)
    ra = re * sf / (ra ** sn)

    theta = lon * rad - olon
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi
    theta *= sn

    nx = int(ra * math.sin(theta) + XO + 0.5)
    ny = int(ro - ra * math.cos(theta) + YO + 0.5)

    return nx, ny


def fetch_station(station, lat, lon, api_key, base_date, base_time):
    nx, ny = latlon_to_grid(lat, lon)

    print(f"\n[{station}] 격자 좌표: nx={nx}, ny={ny}")

    params = {
        "serviceKey": api_key,
        "pageNo": 1,
        "numOfRows": 2000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }

    response = requests.get(
        API_URL, params=params, timeout=30
    )
    response.raise_for_status()

    data = response.json()
    header = data["response"]["header"]

    if str(header["resultCode"]) != "00":
        raise RuntimeError(
            f"{station} API 오류: {header}"
        )

    body = data["response"]["body"]
    items = body["items"]["item"]
    total = int(body["totalCount"])

    if total > len(items):
        raise RuntimeError(
            f"{station}: 응답 일부만 수집됨 "
            f"({len(items)}/{total})"
        )

    df = pd.DataFrame(items)

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

    forecast["station"] = station
    forecast["nx"] = nx
    forecast["ny"] = ny
    forecast["issued_at"] = f"{base_date} {base_time}"

    return forecast


def main():
    api_key = st.secrets["KMA_API_KEY"]
    base_date, base_time = latest_base_time()

    print("공통 예보 발표 기준:", base_date, base_time)

    results = []

    for station, (lat, lon) in STATIONS.items():
        forecast = fetch_station(
            station, lat, lon,
            api_key, base_date, base_time
        )

        results.append(forecast)

        print(
            f"{station}: {len(forecast)}개 예보 시점 수집"
        )
        time.sleep(0.3)

    combined = pd.concat(results, ignore_index=True)

    combined = combined.sort_values(
        ["forecast_datetime", "station"]
    )

    # 날짜·지역별 중복 검증
    assert not combined.duplicated(
        ["station", "forecast_datetime"]
    ).any()

    output = Path(
        "data/processed/jeju_weather_forecast_4stations.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== 최종 수집 결과 =====")

    print(
        combined.groupby("station")
        .agg(
            rows=("forecast_datetime", "size"),
            first=("forecast_datetime", "min"),
            last=("forecast_datetime", "max")
        )
        .to_string()
    )

    print("\n수집 변수:", combined.columns.tolist())
    print("\n저장 완료:", output)


if __name__ == "__main__":
    main()
