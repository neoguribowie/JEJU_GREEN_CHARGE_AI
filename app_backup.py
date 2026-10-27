
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from chargers import (
    REGIONS,
    load_jeju_chargers,
    find_nearby,
    rank_charging_stations
)
import streamlit as st
import pandas as pd
from pathlib import Path

# ====================================
# 기본 설정
# ====================================
st.set_page_config(
    page_title="Green Match",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 모바일 친화적 디자인
st.markdown("""
<style>
    .block-container {
        max-width: 560px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.8px;
    }

    div.stButton > button {
        width: 100%;
        min-height: 52px;
        border-radius: 14px;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e1e9e4;
        border-radius: 14px;
        padding: 14px;
    }

    .gm-hero {
        background: linear-gradient(
            135deg, #e7f6ee, #f2faf5
        );
        color: #174c35;
        border-radius: 20px;
        padding: 25px 20px;
        margin: 16px 0;
    }

    .gm-hero h2 {
        color: #174c35;
        margin: 10px 0;
        font-size: 1.65rem;
    }

    .gm-hero p {
        color: #476554;
        line-height: 1.65;
    }

    .gm-card {
        padding: 18px;
        border: 1px solid #dce9e0;
        border-radius: 16px;
        margin: 12px 0;
    }

    .gm-caption {
        font-size: 0.8rem;
        color: #668174;
    }
</style>
""", unsafe_allow_html=True)


# ====================================
# 공통 데이터 불러오기
# ====================================
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


WIND_PATH = Path(
    "data/processed/wind_weather_2020_2023.csv"
)

CURTAILMENT_PATH = Path(
    "data/processed/curtailment_hourly_2021_2023.csv"
)


# ====================================
# 화면 이동
# ====================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 홈"

pages = [
    "🏠 홈",
    "⚡ 충전소",
    "💰 할인",
    "📊 전력 분석",
    "🧠 AI 성능"
]

st.markdown("## 🌿 Green Match")
st.caption("제주 여행의 친환경 전기차 충전 파트너")

page = st.selectbox(
    "메뉴",
    pages,
    index=pages.index(st.session_state.page)
)
st.session_state.page = page

st.divider()


# ====================================
# 홈
# ====================================
if page == "🏠 홈":

    st.markdown("""
    <div class="gm-hero">
        <small>GREEN CHARGING IN JEJU</small>
        <h2>제주에서 더 똑똑하게 충전하세요.</h2>
        <p>
            관광 일정에 맞는 충전소를 찾고,
            친환경 충전 시간과 예상 충전비를
            확인해 보세요.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.subheader("📍 어디에서 충전할까요?")

    st.write("**① 내 현재 위치로 검색**")
    st.caption("아래 위치 버튼을 누르고 브라우저 위치 권한을 허용하세요.")

    location = streamlit_geolocation()

    if isinstance(location, dict) and (
        location.get("latitude") is not None
        and location.get("longitude") is not None
    ):
        lat = float(location["latitude"])
        lng = float(location["longitude"])

        st.success("현재 위치를 확인했습니다.")

        if st.button(
            "⚡ 내 위치 주변 충전소 찾기",
            type="primary"
        ):
            st.session_state.search_location = (lat, lng)
            st.session_state.search_name = "현재 위치"
            st.session_state.page = "⚡ 충전소"
            st.rerun()

    
    st.write("**② 제주 지역을 선택해서 검색**")

    region = st.selectbox(
        "제주 지역 선택",
        list(REGIONS.keys()),
        key="selected_region_input"
    )

    if st.button(
        "선택한 지역에서 충전소 찾기",
        key="search_region_button",
        use_container_width=True
    ):
        st.session_state.search_location = REGIONS[region]
        st.session_state.search_name = region
        st.session_state.page = "⚡ 충전소"
        st.rerun()



    st.markdown("""
    <div class="gm-card">
        <strong>🌱 친환경 충전 권장 시간</strong>
        <p>실제 미래 기상예보와 전력수요
        예측값을 연결한 뒤 제공됩니다.</p>
        <span class="gm-caption">
            현재는 과거 데이터 기반 AI 분석 단계
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="gm-card">
        <strong>💰 충전 할인 미리 계산하기</strong>
        <p>가상 플러스DR 연계 할인율에 따른
        충전비 절감 효과를 확인해 보세요.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("할인 시뮬레이터 열기"):
        st.session_state.page = "💰 할인"
        st.rerun()


# ====================================
# 충전소 탐색
# ====================================

elif page == "⚡ 충전소":

    st.header("⚡ 주변 급속충전소")

    search_name = st.session_state.get(
        "search_name", "제주시"
    )
    lat, lng = st.session_state.get(
        "search_location", REGIONS["제주시"]
    )

    st.caption(f"검색 기준: {search_name}")

    radius = st.slider(
        "검색 반경 (직선거리 km)",
        min_value=1,
        max_value=10,
        value=5
    )

    api_key = st.secrets.get("EV_API_KEY", "")

    if not api_key:
        st.error(
            ".streamlit/secrets.toml에 "
            "EV_API_KEY를 설정해주세요."
        )
        st.stop()

    try:
        with st.spinner("제주 충전소 정보를 조회하고 있어요..."):
            chargers = load_jeju_chargers(api_key)

        nearby = find_nearby(
            chargers, lat, lng, radius
        )

    except Exception as e:
        st.error(f"충전소 API 조회 실패: {e}")
        st.info(
            "공공데이터포털에서 해당 API의 "
            "활용신청 승인과 인증키를 확인해주세요."
        )
        st.stop()

    if nearby.empty:
        st.warning(
            "검색 범위에 사용 가능 상태의 "
            "급속충전기가 확인되지 않았습니다."
        )
        st.info(
            "검색 반경을 넓히거나 다른 지역을 선택해보세요."
        )
    else:
        # 같은 충전소에 속한 여러 충전기 집계
        stations = (
            nearby.groupby("statId", as_index=False)
            .agg(
                statNm=("statNm", "first"),
                addr=("addr", "first"),
                lat=("lat", "first"),
                lng=("lng", "first"),
                distance_km=("distance_km", "min"),
                available_count=("stat", "size"),
                max_output=("output", "max"),
                statUpdDt=("statUpdDt", "max")
            )
            .sort_values("distance_km")
        )

        # 추천 점수 계산 및 순위 정렬
        ranked = rank_charging_stations(
            stations, radius=radius
        )

        st.subheader("🌿 Green Match 추천 TOP 3")

        st.caption(
            "거리 50% · 충전 출력 30% · "
            "사용 가능 대수 20% 기준"
        )

        for i, (_, station) in enumerate(
            ranked.head(3).iterrows(), start=1
        ):
            with st.container(border=True):
                st.markdown(
                    f"### {i}위 · {station['statNm']}"
                )

                st.metric(
                    "Green Match 추천 점수",
                    f"{station['recommendation_score']:.1f}점"
                )

                st.write(
                    f"📍 직선거리 "
                    f"**{station['distance_km']:.1f}km**"
                )

                st.write(
                    f"⚡ 최대 출력 "
                    f"**{station['max_output']:.0f}kW**"
                )

                st.write(
                    f"🟢 사용 가능 급속충전기 "
                    f"**{station['available_count']}기**"
                )

                map_url = (
                    "https://www.google.com/maps/search/?api=1"
                    f"&query={station['lat']},{station['lng']}"
                )

                st.link_button(
                    "📍 지도 앱에서 위치 확인",
                    map_url,
                    use_container_width=True
                )

        st.success(
            f"사용 가능한 급속충전기를 보유한 "
            f"충전소 {len(stations)}곳을 찾았습니다."
        )

        st.caption(
            "표시된 충전기 상태는 API 제공 시점의 "
            "정보이며, 도착 시 이용 가능을 보장하지 않습니다."
        )

        # 지도 생성
        m = folium.Map(
            location=[lat, lng],
            zoom_start=12,
            control_scale=True
        )

        folium.Marker(
            [lat, lng],
            tooltip="검색 기준 위치",
            icon=folium.Icon(
                color="blue",
                icon="info-sign"
            )
        ).add_to(m)

        folium.Circle(
            [lat, lng],
            radius=radius * 1000,
            color="#239B68",
            fill=False
        ).add_to(m)

        for _, row in stations.iterrows():
            popup_text = (
                f"{row['statNm']} | "
                f"사용 가능 {row['available_count']}기 | "
                f"최대 {row['max_output']:.0f}kW"
            )

            folium.Marker(
                [row["lat"], row["lng"]],
                tooltip=popup_text,
                icon=folium.Icon(
                    color="green",
                    icon="flash"
                )
            ).add_to(m)

        st_folium(
            m,
            width="100%",
            height=370,
            returned_objects=[]
        )

        st.subheader("📋 가까운 충전소")

        for _, row in stations.head(15).iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['statNm']}**")
                st.caption(str(row["addr"]))

                st.write(
                    f"📍 직선거리 {row['distance_km']:.1f}km"
                    f" · 사용 가능 {row['available_count']}기"
                    f" · 최대 {row['max_output']:.0f}kW"
                )

                updated = str(row["statUpdDt"])
                st.caption(
                    f"최근 제공 상태갱신시각: {updated}"
                )

                # 스마트폰 지도 앱에서 좌표 검색
                map_url = (
                    "https://www.google.com/maps/search/?api=1"
                    f"&query={row['lat']},{row['lng']}"
                )
                st.link_button(
                    "지도 앱에서 위치 보기",
                    map_url,
                    use_container_width=True
                )



# ====================================
# 가상 할인 시뮬레이터
# ====================================
elif page == "💰 할인":

    st.header("💰 충전요금 계산기")
    st.caption("플러스DR 연계 가상 할인 시나리오")

    energy = st.number_input(
        "예상 충전량 (kWh)",
        min_value=1.0,
        max_value=150.0,
        value=20.0,
        step=5.0
    )

    unit_price = st.number_input(
        "기본 충전요금 (원/kWh)",
        min_value=0.0,
        value=348.4,
        step=10.0
    )

    discount = st.slider(
        "가상 할인율 (%)",
        min_value=0,
        max_value=50,
        value=20,
        step=5
    )

    original = energy * unit_price
    saving = original * discount / 100
    final_price = original - saving

    st.divider()

    st.metric(
        "기본 충전비",
        f"{original:,.0f}원"
    )

    st.metric(
        "예상 할인액",
        f"{saving:,.0f}원"
    )

    st.metric(
        "할인 후 충전비",
        f"{final_price:,.0f}원"
    )

    st.info(
        "이 할인은 실제 시행 중인 혜택이 아니라 "
        "경제성 검토를 위한 가상 시나리오입니다."
    )


# ====================================
# 실제 전력 분석
# ====================================
elif page == "📊 전력 분석":

    st.header("📊 제주 전력 분석")

    st.subheader("풍력 전력거래량 이력")

    if WIND_PATH.exists():
        wind = load_csv(str(WIND_PATH))

        wind["trade_date"] = pd.to_datetime(
            wind["trade_date"]
        )

        wind = wind[
            wind["trade_date"].dt.year == 2023
        ].copy()

        wind["wind_mwh"] = pd.to_numeric(
            wind["wind_mwh"],
            errors="coerce"
        )

        daily_wind = (
            wind.groupby("trade_date")["wind_mwh"]
            .sum()
        )

        st.line_chart(daily_wind)

        st.caption(
            "2023년 실제 풍력 전력거래량 이력입니다. "
            "AI 미래 예측 결과가 아닙니다. "
            "또한 전력시장 거래량은 제주 전체 "
            "풍력발전량과 다를 수 있습니다."
        )
    else:
        st.warning("풍력 데이터 파일이 없습니다.")

    st.divider()

    st.subheader("출력제어 발생 이력")

    if CURTAILMENT_PATH.exists():
        curtailment = load_csv(
            str(CURTAILMENT_PATH)
        )

        curtailment["trade_date"] = pd.to_datetime(
            curtailment["trade_date"]
        )

        curtailment = curtailment[
            curtailment["trade_date"].dt.year == 2023
        ].copy()

        daily_events = (
            curtailment.groupby("trade_date")[
                "curtailment"
            ].sum()
        )

        st.bar_chart(daily_events)

        st.caption(
            "2023년 날짜별 출력제어 발생 시간 수. "
            "출력제어량(MWh)이 아닌 발생 횟수입니다."
        )
    else:
        st.warning("출력제어 데이터 파일이 없습니다.")


# ====================================
# AI 모델 성능
# ====================================
elif page == "🧠 AI 성능":

    st.header("🧠 AI 모델 성능")

    st.write("출력제어 발생 분류 실험")

    metrics = pd.DataFrame({
        "모델": [
            "C001",
            "C003",
            "C004-C001",
            "C004-C003"
        ],
        "Precision": [
            0.9180, 1.0000, 0.4414, 0.5726
        ],
        "Recall": [
            0.1948, 0.1148, 0.9096, 0.8504
        ],
        "F1-score": [
            0.3214, 0.2059, 0.5943, 0.6844
        ]
    })

    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "C004는 2022년 검증으로 임계값 0.05를 "
        "선택한 후 수행한 탐색적 후속 실험입니다. "
        "C001·C003 원 실험과 C004는 "
        "재학습 결과에 일부 차이가 있습니다."
    )

    st.warning(
        "이 수치는 실제 관측 기상·전력수요를 "
        "사용한 과거 데이터 평가 결과이며, "
        "미래 출력제어 예측 성능이 아닙니다."
    )
