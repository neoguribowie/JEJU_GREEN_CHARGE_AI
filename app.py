
import textwrap
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
from ai_dashboard import (
    render_power_prediction,
    render_model_comparison,
    render_future_risk_comparison,
)
from charger_demo import (
    save_charger_snapshot,
    load_charger_snapshot
)
from tourist_spots import TOURIST_SPOTS

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

    /* 모바일 버튼 가독성 */
    div.stButton > button {
        min-height: 46px;
        border-radius: 12px;
        font-weight: 600;
    }

    /* 안내 문구 줄 간격 */
    .stAlert p {
        line-height: 1.6;
    }

    /* 모바일 화면 */
    @media (max-width: 600px) {
        .gm-hero h2 {
            font-size: 1.65rem;
            line-height: 1.35;
        }

        .gm-hero p {
            line-height: 1.65;
        }
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

AI_PREDICTION_PATH = Path(
    "results/curtailment_c004/C003_predictions_2023.csv"
)

FUTURE_RISK_PATH = Path(
    "results/curtailment_c005/future_risk_forecast.csv"
)

# ====================================
# 화면 이동
# ====================================
if "page" not in st.session_state:
    st.session_state.page = "🏠 홈"

pages = [
    "🏠 홈",
    "⚡ 충전소",
    "🌱 AI 충전 가이드",
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


    st.markdown(
        textwrap.dedent("""
        <style>
        .gm-home-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 20px 18px;
            margin-bottom: 12px;
        }
        .gm-home-logo {
            font-size: 14px;
            color: #222;
            margin-bottom: 18px;
        }
        .gm-home-title {
            font-size: 27px;
            font-weight: 750;
            line-height: 1.32;
            letter-spacing: -1px;
            color: #202124;
            margin-bottom: 14px;
            word-break: keep-all;
        }
        .gm-home-description {
            font-size: 13px;
            line-height: 1.65;
            color: #737373;
            margin-bottom: 16px;
            word-break: keep-all;
        }
        .gm-steps {
            background: #f6f6f6;
            border-radius: 14px;
            padding: 15px 14px;
        }
        .gm-steps-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 12px;
        }
        .gm-step {
            font-size: 13px;
            font-weight: 600;
            color: #303030;
            margin-bottom: 11px;
        }
        .gm-step:last-child {
            margin-bottom: 0;
        }
        .gm-home-note {
            margin-top: 12px;
            font-size: 11px;
            color: #888;
            line-height: 1.5;
        }
        @media (max-width: 600px) {
            .gm-home-card { padding: 16px 14px; }
            .gm-home-title { font-size: 25px; }
        }
        </style>
        <div class="gm-home-card">
            <div class="gm-home-logo">🌿 Green Match</div>
            <div class="gm-home-title">
                낮에 충전하고, 더 알뜰하게 제주를 즐겨요!
            </div>
            <div class="gm-home-description">
                AI 분석으로 충전을 검토할 시간대를 확인하고,
                여행지 주변 급속충전소를 찾아보세요.
            </div>
            <div class="gm-steps">
                <div class="gm-steps-label">서비스 이용 순서</div>
                <div class="gm-step">① 충전 날짜 확인</div>
                <div class="gm-step">② AI 분석 시간대 확인</div>
                <div class="gm-step">③ 주변 급속충전소 검색</div>
                <div class="gm-step">④ 가상 할인 혜택 비교</div>
            </div>
            <div class="gm-home-note">
                ※ 충전 할인은 현재 제공되는 혜택이 아닌
                가상 시뮬레이션입니다.
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


    if st.button(
        "🌱 AI 충전 시간 확인하기",
        key="home_to_ai",
        use_container_width=True
    ):
        st.session_state.page = "🌱 AI 충전 가이드"
        st.rerun()

    st.subheader("📍 어디에서 충전할까요?")

    home_mode = st.radio(
        "위치 선택 방법",
        ["관광지 선택", "지역 선택", "현재 위치"],
        horizontal=True,
        key="home_search_mode"
    )

    if home_mode == "관광지 선택":
        home_spot = st.selectbox(
            "방문할 관광지",
            list(TOURIST_SPOTS.keys()),
            index=None,
            placeholder="관광지를 선택해주세요",
            key="home_tourist_spot"
        )
        if st.button(
            "⚡ 이 관광지 주변 충전소 찾기",
            key="home_spot_search",
            use_container_width=True,
            type="primary"
        ):
            if home_spot is None:
                st.warning("먼저 관광지를 선택해 주세요.")
            else:
                st.session_state.search_location = TOURIST_SPOTS[home_spot]
                st.session_state.search_name = home_spot
                st.session_state.charger_search_requested = True
                st.session_state.page = "⚡ 충전소"
                st.rerun()

    elif home_mode == "지역 선택":
        home_region = st.selectbox(
            "제주 지역",
            list(REGIONS.keys()),
            index=None,
            placeholder="지역을 선택해주세요",
            key="home_region_select"
        )
        if st.button(
            "⚡ 이 지역 주변 충전소 찾기",
            key="home_region_search",
            use_container_width=True,
            type="primary"
        ):
            if home_region is None:
                st.warning("먼저 지역을 선택해 주세요.")
            else:
                st.session_state.search_location = REGIONS[home_region]
                st.session_state.search_name = home_region
                st.session_state.charger_search_requested = True
                st.session_state.page = "⚡ 충전소"
                st.rerun()

    else:
        st.caption("아래 위치 버튼을 누르고 브라우저 위치 권한을 허용하세요.")
        location = streamlit_geolocation()
        if isinstance(location, dict) and (
            location.get("latitude") is not None
            and location.get("longitude") is not None
        ):
            gps_lat = float(location["latitude"])
            gps_lng = float(location["longitude"])
            st.success("현재 위치를 확인했습니다.")
            if st.button(
                "⚡ 내 위치 주변 충전소 찾기",
                key="home_gps_search",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.search_location = (gps_lat, gps_lng)
                st.session_state.search_name = "현재 위치"
                st.session_state.charger_search_requested = True
                st.session_state.page = "⚡ 충전소"
                st.rerun()
        else:
            st.info("위치 정보가 확인되면 주변 충전소를 검색할 수 있어요.")


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

    demo_mode = st.toggle(
        "📂 저장 데이터 시연 모드",
        value=False,
        help="실시간 API 대신 사전에 저장한 충전소 정보를 사용합니다."
    )    

    st.header("⚡ 주변 급속충전소")

    # 홈에서 검색 버튼을 누른 경우, 홈의 좌표를 그대로 사용합니다.
    home_requested = st.session_state.pop("charger_search_requested", False)
    if home_requested:
        st.session_state.charger_search_mode = "홈에서 설정한 위치"

    st.markdown("#### 📍 어디에서 충전하고 싶으세요?")
    search_mode = st.radio(
        "검색 기준",
        ["관광지 선택", "지역 선택", "홈에서 설정한 위치"],
        horizontal=True,
        key="charger_search_mode"
    )

    selected_location = None
    selected_name = None

    if search_mode == "관광지 선택":
        selected_spot = st.selectbox(
            "방문할 관광지",
            list(TOURIST_SPOTS.keys()),
            index=None,
            placeholder="관광지를 선택해주세요",
            key="charger_tourist_spot"
        )
        if selected_spot is not None:
            selected_location = TOURIST_SPOTS[selected_spot]
            selected_name = selected_spot

    elif search_mode == "지역 선택":
        selected_region = st.selectbox(
            "제주 지역",
            list(REGIONS.keys()),
            index=None,
            placeholder="지역을 선택해주세요",
            key="charger_region_select"
        )
        if selected_region is not None:
            selected_location = REGIONS[selected_region]
            selected_name = selected_region

    else:
        selected_location = st.session_state.get("search_location")
        selected_name = st.session_state.get("search_name", "현재 위치")
        if selected_location is None:
            st.info("홈에서 관광지·지역·GPS 중 하나를 먼저 선택해 주세요.")

    if selected_location is not None:
        st.caption(f"📍 검색 기준: {selected_name}")

    radius = st.slider(
        "검색 반경 (직선거리 km)",
        min_value=1,
        max_value=10,
        value=5
    )

    search_clicked = st.button(
        "🔍 주변 급속충전소 검색",
        key="search_chargers",
        use_container_width=True,
        type="primary"
    )

    # 메뉴에서 직접 진입하면 버튼 클릭 전에는 API를 호출하지 않습니다.
    # 홈에서 검색해 넘어온 첫 실행만 버튼 없이 자동으로 조회합니다.
    if not search_clicked and not home_requested:
        st.info("관광지나 지역을 선택한 뒤 검색 버튼을 눌러주세요.")
        st.stop()

    if selected_location is None:
        st.warning("검색할 관광지나 지역을 먼저 선택해 주세요.")
        st.stop()

    lat, lng = selected_location
    # 홈에서 설정한 위치는 유지하고, 직접 검색한 위치만 갱신합니다.
    if search_mode != "홈에서 설정한 위치":
        st.session_state.search_location = selected_location
        st.session_state.search_name = selected_name

    # 충전소 API 인증키 불러오기
    api_key = st.secrets.get("EV_API_KEY", "")

    # 실시간 모드에서만 인증키 확인
    if not demo_mode and not api_key:
        st.error(
            "충전소 API 인증키가 없습니다. "
            "secrets.toml 설정을 확인해 주세요."
        )
        st.stop()
        
    try:
        if demo_mode:
            
            # 저장된 충전소 데이터 불러오기
            chargers_data, saved_at = load_charger_snapshot()

            chargers = (
                pd.DataFrame(chargers_data)
                if chargers_data is not None
                else None
            )

            if chargers is None or chargers.empty:
                st.warning(
                    "저장된 충전소 데이터가 없습니다. "
                    "먼저 실시간 조회를 진행해 주세요."
                )
                st.stop()

            st.warning(
                f"📂 저장 데이터 시연 모드\n\n"
                f"수집 시각: {saved_at}\n\n"
                "이용 가능 상태는 수집 당시의 정보입니다."
            )

        else:
            # 실시간 충전소 API 조회
            with st.spinner("제주 충전소 정보를 조회하고 있어요..."):
                chargers = load_jeju_chargers(api_key)

            if not chargers.empty:
                saved = save_charger_snapshot(
                    chargers.to_dict("records")
                )

                if saved:
                    st.success("✅ 충전소 시연 데이터 저장 완료!")
            else:
                st.warning("API에서 충전소 데이터가 반환되지 않았습니다.")

        # 실시간 및 시연 모드에서 공통으로 실행
        nearby = find_nearby(
            chargers, lat, lng, radius
        )

    except Exception as e:
        st.error(f"충전소 조회 실패: {e}")
        st.info(
            "실시간 API 연결 또는 저장 데이터 처리 중 "
            "문제가 발생했습니다."
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

        st.info(
            "🔌 지금은 이용 가능한 충전기라도, "
            "도착했을 때는 다른 사람이 사용 중일 수 있어요. "
            "출발 전에 한 번 더 확인해 주세요!"
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
# AI 친환경 충전 가이드
# ====================================
elif page == "🌱 AI 충전 가이드":

    st.header("🌱 AI 친환경 충전 가이드")


    # ==========================================
    # EXP-C005 미래 기상예보 기반 시험 예측
    # ==========================================
    st.subheader("☀️ 향후 충전 검토 시간대")

    st.caption(
        "실제 기상청 예보를 입력한 AI 출력제어 "
        "위험 참고 지표입니다. 미래 예측 정확도는 "
        "아직 검증되지 않았습니다."
    )

    if st.button(
        "🔄 저장된 예측 결과 다시 불러오기",
        key="reload_c005"
    ):
        st.cache_data.clear()
        st.rerun()

    if not FUTURE_RISK_PATH.exists():
        st.warning(
            "미래 예측 결과 파일이 없습니다. "
            "train_curtailment_c005.py를 먼저 실행하세요."
        )

    else:
        future = load_csv(str(FUTURE_RISK_PATH)).copy()

        future["forecast_datetime"] = pd.to_datetime(
            future["forecast_datetime"], errors="coerce"
        )
        future["risk_score"] = pd.to_numeric(
            future["risk_score"], errors="coerce"
        )

        future = future.dropna(
            subset=["forecast_datetime", "risk_score"]
        ).sort_values("forecast_datetime")

        # 현재 시각과 예보 시각 비교 (한국시간)
        now_kst = pd.Timestamp.now(tz="Asia/Seoul")
        now_local = now_kst.tz_localize(None)

        # 과거가 된 예보 시점은 제외
        upcoming = future[
            future["forecast_datetime"] > now_local
        ].copy()

        if future.empty:
            st.error("미래 예측 CSV에 유효한 데이터가 없습니다.")

        else:
            issued_raw = str(
                future["forecast_issued_at"].iloc[0]
            )

            issued = pd.to_datetime(
                issued_raw,
                format="%Y%m%d %H%M",
                errors="coerce"
            )

            st.caption(
                f"기상예보 발표 기준: {issued_raw}"
            )

            # 발표 후 12시간을 넘겼다면 갱신 권장
            if pd.isna(issued):
                st.warning(
                    "예보 발표시각을 확인할 수 없습니다."
                )
            elif (now_local - issued).total_seconds() > 12 * 3600:
                st.warning(
                    "⚠️ 저장된 예보가 발표된 지 "
                    "12시간 이상 지났습니다. "
                    "최신 예보로 다시 생성하는 것을 권장합니다."
                )

            if upcoming.empty:
                st.warning(
                    "현재 시각 이후의 예측이 없습니다. "
                    "기상예보를 새로 수집해야 합니다."
                )
            else:
                upcoming["date"] = (
                    upcoming["forecast_datetime"].dt.date
                )

                available_dates = sorted(
                    upcoming["date"].unique()
                )

                selected_date = st.selectbox(
                    "예측 날짜 선택",
                    options=available_dates,
                    format_func=lambda x: x.strftime(
                        "%Y년 %m월 %d일"
                    ),
                    key="future_date"
                )

                day = upcoming[
                    upcoming["date"] == selected_date
                ].copy()

                day["hour"] = (
                    day["forecast_datetime"].dt.hour
                )

                st.subheader("시간별 AI 위험 점수")

                st.line_chart(
                    day.set_index("hour")[["risk_score"]]
                )

                st.caption(
                    "점수가 높을수록 모델이 과거 출력제어 "
                    "발생 사례와 유사하게 판단한 시간입니다. "
                    "실제 잉여전력량이나 탄소저감량이 아닙니다."
                )

                # 선택한 날짜에서 점수가 높은 시간 3개
                top_hours = day.sort_values(
                    ["risk_score", "forecast_datetime"],
                    ascending=[False, True]
                ).head(3)

                                # 관광객용 AI 충전 안내 카드
                max_score = day["risk_score"].max()

                if max_score >= 0.05:
                    recommended_hours = top_hours[
                        top_hours["risk_score"] >= 0.05
                    ].copy()

                    hour_labels = [
                        t.strftime("%H:%M")
                        for t in recommended_hours[
                            "forecast_datetime"
                        ]
                    ]

                    time_text = ", ".join(hour_labels)

                    st.success(
                        "🌿 **낮 시간 충전을 검토해 보세요!**\n\n"
                        f"선택한 날짜의 AI 분석에서 "
                        f"**{time_text}** 시간대가 "
                        "상대적으로 높은 출력제어 위험 점수를 보였어요.\n\n"
                        "여행 일정에 맞는다면 주변 급속충전소를 "
                        "확인해 보세요."
                    )

                else:
                    st.info(
                        "☀️ **오늘은 일반 충전 안내를 확인해 보세요.**\n\n"
                        "선택한 날짜에는 AI 위험 점수가 "
                        "설정한 기준보다 높은 시간이 없어요.\n\n"
                        "필요한 시간에 주변 급속충전소를 "
                        "이용할 수 있어요."
                    )

                st.caption(
                    "※ AI 분석은 참고 정보이며, "
                    "실제 잉여전력 발생이나 할인 혜택을 "
                    "보장하지 않습니다."
                )

                st.caption(
                    "여행 일정에 맞는 시간대를 골랐다면, "
                    "이제 가까운 급속충전소를 찾아보세요!"
                )

                if st.button(
                    "⚡ 주변 급속충전소 찾기",
                    key="ai_guide_to_chargers",
                    use_container_width=True
                ):
                    st.session_state.page = "⚡ 충전소"
                    st.rerun()


                st.subheader("🌿 우선 검토할 시간대")

                for _, item in top_hours.iterrows():
                    hour_text = item[
                        "forecast_datetime"
                    ].strftime("%H:%M")

                    st.markdown(
                        f"**{hour_text} · "
                        f"AI 위험 점수 {item['risk_score']:.2f}**"
                    )

                if (day["risk_score"] < 0.05).all():
                    st.info(
                        "선택한 날짜에는 기존 임계값 "
                        "0.05 이상인 시간이 없습니다."
                    )

                st.info(
                    "위 시간대는 충전 검토용 참고 정보입니다. "
                    "실제 친환경 효과와 플러스DR 발령 여부는 "
                    "별도로 확인해야 합니다."
                )

    st.divider()

    with st.expander("🔬 전력수요를 고려한 AI 모델 비교"):
        render_future_risk_comparison()

    st.divider()
    st.subheader("📚 과거 AI 분석 결과")
    st.caption(
        "아래는 2023년 실제 관측자료와 "
        "출력제어 이력을 이용한 사후 평가입니다."
    )

    if not AI_PREDICTION_PATH.exists():
        st.warning(
            "C004 예측 결과 파일을 찾지 못했습니다."
        )
    else:
        ai = load_csv(str(AI_PREDICTION_PATH))

        ai["trade_date"] = pd.to_datetime(
            ai["trade_date"]
        )

        ai["trade_hour"] = pd.to_numeric(
            ai["trade_hour"], errors="coerce"
        )

        ai["probability"] = pd.to_numeric(
            ai["probability"], errors="coerce"
        )

        ai = ai.dropna(
            subset=["trade_date", "trade_hour", "probability"]
        ).copy()

        st.subheader("📅 과거 분석 날짜 선택")

        date = st.date_input(
            "2023년 날짜",
            value=pd.Timestamp("2023-04-15").date(),
            min_value=pd.Timestamp("2023-01-01").date(),
            max_value=pd.Timestamp("2023-12-31").date()
        )

        selected = ai[
            ai["trade_date"].dt.date == date
        ].sort_values("trade_hour")

        if selected.empty:
            st.warning("선택한 날짜의 기록이 없습니다.")
        else:
            actual_count = int(
                selected["curtailment"].sum()
            )

            predicted_count = int(
                selected["prediction_selected"].sum()
            )

            st.metric(
                "실제 출력제어 발생 시간",
                f"{actual_count}시간"
            )

            st.metric(
                "AI가 발생으로 분류한 시간",
                f"{predicted_count}시간"
            )

            st.subheader("시간별 출력제어 위험 점수")

            chart = selected[
                ["trade_hour", "probability"]
            ].copy()

            chart = chart.set_index("trade_hour")

            st.line_chart(chart)

            st.caption(
                "세로축은 Random Forest가 산출한 "
                "출력제어 발생 클래스 점수(0~1)입니다. "
                "미래 실제 발생 확률로 검증된 값은 아닙니다."
            )

            detected = selected[
                selected["prediction_selected"] == 1
            ]["trade_hour"].astype(int).tolist()

            if detected:
                hours_text = ", ".join(
                    f"{hour}시" for hour in detected
                )
                st.success(
                    "AI가 출력제어 발생으로 분류한 시간: "
                    + hours_text
                )
            else:
                st.write(
                    "선택 날짜에는 AI가 출력제어 발생으로 "
                    "분류한 시간이 없습니다."
                )

            with st.expander("📋 실제 발생 여부와 비교"):
                comparison = selected[
                    [
                        "trade_hour",
                        "probability",
                        "prediction_selected",
                        "curtailment"
                    ]
                ].copy()

                comparison.columns = [
                    "거래시간",
                    "AI 점수",
                    "AI 발생 분류",
                    "실제 발생"
                ]

                st.dataframe(
                    comparison,
                    hide_index=True,
                    use_container_width=True
                )

    
    # ====================================
    # 충전소 탐색 화면으로 이동
    # ====================================
    
    st.divider()

    st.subheader("🚗 실제 여행 중 충전하려면")

    st.write(
        "현재는 AI 미래 예보 대신 "
        "실시간 급속충전소 정보를 기준으로 안내합니다."
    )

    if st.button(
        "⚡ 주변 충전소 찾기",
        key="ai_go_chargers",
        use_container_width=True,
        type="primary"
    ):
        st.session_state.page = "⚡ 충전소"
        st.rerun()


# ====================================
# 플러스DR 연계 가상 할인 시뮬레이터
# ====================================
elif page == "💰 할인":

    st.header("💰 Green Match 충전 할인")
    st.caption("플러스DR 연계 경제성 시뮬레이션")

    st.info(
        "본 계산은 가상 할인 시나리오입니다. "
        "실제 플러스DR 참여·발령·정산 또는 "
        "관광객 할인 혜택을 보장하지 않습니다."
    )

    # 1. 관광객 충전 조건
    st.subheader("🚗 예상 충전 정보")

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
        max_value=1000.0,
        value=348.4,
        step=1.0
    )

    discount = st.slider(
        "가상 할인율 (%)",
        min_value=0,
        max_value=50,
        value=20,
        step=5
    )

    original_cost = energy * unit_price
    discount_cost = original_cost * discount / 100
    final_cost = original_cost - discount_cost

    st.divider()

    st.subheader("💳 관광객 예상 충전비")

    st.metric(
        "기본 충전비",
        f"{original_cost:,.0f}원"
    )

    st.metric(
        "할인 후 충전비",
        f"{final_cost:,.0f}원"
    )

    st.metric(
        "관광객 예상 절감액",
        f"{discount_cost:,.0f}원"
    )

    # 2. 플러스DR 정산 시나리오
    st.divider()
    st.subheader("⚡ 플러스DR 재원 검토")

    st.caption(
        "아래 정산단가와 인정비율은 시나리오 "
        "입력값이며, 실제 정산 확정값이 아닙니다."
    )

    dr_rate = st.number_input(
        "가정한 플러스DR 정산단가 (원/kWh)",
        min_value=0.0,
        max_value=500.0,
        value=96.0,
        step=1.0
    )

    recognized_ratio = st.slider(
        "충전량 중 DR 정산 인정 비율 (%)",
        min_value=0,
        max_value=100,
        value=100,
        step=10
    )

    # 실제 정산은 별도의 기준부하 및
    # 참여·이행 검증을 통해 결정됨.
    recognized_energy = (
        energy * recognized_ratio / 100
    )

    estimated_dr_fund = recognized_energy * dr_rate

    # 단순 경제성 비교:
    # 정산액 전액을 할인에 투입한다고 가정
    funding_gap = discount_cost - estimated_dr_fund

    st.metric(
        "가상 정산 인정량",
        f"{recognized_energy:,.1f} kWh"
    )

    st.metric(
        "가상 플러스DR 정산 재원",
        f"{estimated_dr_fund:,.0f}원"
    )

    if funding_gap > 0:
        st.error(
            f"할인 비용을 충당하려면 "
            f"추가 재원 {funding_gap:,.0f}원이 필요합니다."
        )
    else:
        st.success(
            "이 가정에서는 플러스DR 정산 재원이 "
            "할인액 이상입니다."
        )
        st.caption(
            f"단순 계산상 잔여 금액: "
            f"{abs(funding_gap):,.0f}원"
        )

    # 3. 결과 정리
    st.divider()
    st.subheader("📋 시뮬레이션 결과")

    summary = pd.DataFrame({
        "항목": [
            "충전량 (kWh)",
            "기본 충전요금 (원/kWh)",
            "가상 할인율 (%)",
            "관광객 절감액 (원)",
            "가상 DR 인정량 (kWh)",
            "가상 DR 정산 재원 (원)",
            "추가 필요 재원 (원)"
        ],
        "값": [
            energy,
            unit_price,
            discount,
            round(discount_cost),
            recognized_energy,
            round(estimated_dr_fund),
            round(max(0, funding_gap))
        ]
    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "단순 비교를 위해 DR 정산금을 전액 "
        "관광객 할인에 사용할 수 있다고 가정했습니다. "
        "실제 운영에는 기준부하 산정, 정산 인정량, "
        "사업자 배분과 운영비 등이 영향을 미칩니다."
    )


# ====================================
# 실제 전력 분석
# ====================================
elif page == "📊 전력 분석":

    render_power_prediction()
    st.divider()

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

    render_model_comparison()
    st.divider()

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
