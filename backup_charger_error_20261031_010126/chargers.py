
import math
import requests
import pandas as pd
import streamlit as st

API_URL = (
    "https://apis.data.go.kr/"
    "B552584/EvCharger/getChargerInfo"
)

# 제주 관광지별 검색 시작 좌표
REGIONS = {
    "제주시": (33.4996, 126.5312),
    "애월": (33.4620, 126.3090),
    "한림": (33.4129, 126.2672),
    "서귀포": (33.2541, 126.5601),
    "중문": (33.2496, 126.4122),
    "성산": (33.4581, 126.9426),
    "표선": (33.3250, 126.8325),
}

def distance_km(lat1, lon1, lat2, lon2):
    """두 좌표 사이 직선거리(km)."""
    radius = 6371.0

    a1, a2 = map(math.radians, [lat1, lat2])
    da = math.radians(lat2 - lat1)
    db = math.radians(lon2 - lon1)

    a = (
        math.sin(da / 2) ** 2
        + math.cos(a1) * math.cos(a2)
        * math.sin(db / 2) ** 2
    )

    return radius * 2 * math.asin(min(1.0, math.sqrt(a)))


@st.cache_data(ttl=120, show_spinner=False)
def load_jeju_chargers(api_key):
    """한국환경공단 제주 충전기 기본정보 조회."""
    rows = []
    page = 1
    per_page = 9999

    while True:
        params = {
            "serviceKey": api_key,
            "pageNo": page,
            "numOfRows": per_page,
            "zcode": "50",
            "dataType": "JSON"
        }

        response = requests.get(
            API_URL, params=params, timeout=25
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(
                "API가 JSON을 반환하지 않았습니다. "
                "인증키 또는 응답 형식을 확인하세요."
            )

        if str(data.get("resultCode", "00")) != "00":
            raise RuntimeError(
                f"API 오류: {data.get('resultMsg', data)}"
            )

        items = data.get("items") or {}
        batch = items.get("item", []) if isinstance(items, dict) else []

        if isinstance(batch, dict):
            batch = [batch]

        rows.extend(batch)

        total = int(data.get("totalCount") or 0)
        if len(rows) >= total or not batch:
            break

        page += 1

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    required = [
        "statId", "statNm", "lat", "lng",
        "output", "stat"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"API 필수 컬럼 누락: {missing}"
        )

    for col in ["lat", "lng", "output"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # 급속충전기 기준: 정격출력 50kW 이상
    df = df[
        (df["output"] >= 50)
        & df["lat"].between(33.0, 34.0)
        & df["lng"].between(126.0, 127.2)
    ].copy()

    # 삭제되거나 이용이 제한된 충전소 제외
    if "delYn" in df.columns:
        df = df[df["delYn"].fillna("N") != "Y"]

    if "limitYn" in df.columns:
        df = df[df["limitYn"].fillna("N") != "Y"]

    return df


def find_nearby(df, latitude, longitude, radius=5):
    """반경 내 이용 가능한 급속충전기 검색."""
    if df.empty:
        return pd.DataFrame()

    nearby = df.copy()

    nearby["distance_km"] = nearby.apply(
        lambda row: distance_km(
            latitude, longitude,
            float(row["lat"]), float(row["lng"])
        ),
        axis=1
    )

    nearby = nearby[
        (nearby["distance_km"] <= radius)
        & (nearby["stat"].astype(str) == "2")
    ].copy()

    return nearby.sort_values(
        ["distance_km", "output"],
        ascending=[True, False]
    )

def rank_charging_stations(stations, radius=5):
    """
    충전소별 추천 점수 계산
    - 거리: 50점
    - 충전 출력: 30점
    - 사용 가능 대수: 20점
    총점: 100점
    """

    if stations.empty:
        return stations.copy()

    ranked = stations.copy()

    # 1. 거리 점수 (가까울수록 유리)
    ranked["distance_score"] = (
        1 - ranked["distance_km"] / radius
    ).clip(lower=0, upper=1) * 50

    # 2. 출력 점수 (200kW 이상은 만점)
    ranked["power_score"] = (
        ranked["max_output"] / 200
    ).clip(lower=0, upper=1) * 30

    # 3. 사용 가능 대수 (4기 이상은 만점)
    ranked["availability_score"] = (
        ranked["available_count"] / 4
    ).clip(lower=0, upper=1) * 20

    # 최종 추천 점수
    ranked["recommendation_score"] = (
        ranked["distance_score"]
        + ranked["power_score"]
        + ranked["availability_score"]
    ).round(1)

    return ranked.sort_values(
        ["recommendation_score", "distance_km"],
        ascending=[False, True]
    ).reset_index(drop=True)
