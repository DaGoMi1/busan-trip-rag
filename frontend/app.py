from datetime import date, timedelta

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
MAX_TRIP_DAYS = 7
PURPOSES = ["힐링", "맛집", "문화관광", "쇼핑", "자연/액티비티", "종합"]
COMPANIONS = ["혼자", "커플", "친구", "가족", "기타"]
INTENSITY_LABELS = {
    1: "낮음 · 카페·실내, 하루 2~3곳",
    2: "가벼운 산책 · 하루 3~4곳",
    3: "보통 · 하루 4~5곳",
    4: "활발 · 언덕 OK, 하루 5~6곳",
    5: "높음 · 산·숲 가능, 하루 5~7곳",
}

st.set_page_config(page_title="부산 여행 코스 추천", page_icon="🏖️", layout="wide")

st.title("🏖️ 부산 여행 코스")
st.subheader("숙소 근처에서, 하루 흐름에 맞는 코스를 붙여 드려요")


@st.cache_data(ttl=60)
def fetch_lodgings(query: str = "") -> list[dict]:
    response = requests.get(
        f"{API_BASE}/api/lodgings",
        params={"q": query},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("lodgings") or []


def lodging_label(item: dict) -> str:
    district = item.get("district") or "구 미상"
    return f"{item.get('name')} ({district})"


def selected_lodging_id(label: str, lodgings: list[dict], key: str) -> str | None:
    if not lodgings:
        return None
    labels = [lodging_label(item) for item in lodgings]
    choice = st.selectbox(label, labels, key=key)
    return str(lodgings[labels.index(choice)]["id"])


with st.sidebar:
    st.header("여행 정보 입력")
    start_date = st.date_input("시작일", value=date.today())
    end_date = st.date_input("종료일", value=date.today() + timedelta(days=1))
    purpose = st.selectbox("여행 목적", PURPOSES)
    companion = st.selectbox("동행자", COMPANIONS)
    intensity = st.slider("활동 강도", min_value=1, max_value=5, value=3)
    st.caption(INTENSITY_LABELS[intensity])
    st.caption(
        "탐색은 전날 숙소 또는 당일 숙소 각 2km 이내(OR). "
        "같은 숙소에 여러 날 묵으면 2.5→4→5.5km로 조금씩 넓어집니다."
    )

    days = (end_date - start_date).days + 1
    if end_date < start_date:
        st.error("종료일은 시작일 이후여야 합니다.")
    elif days > MAX_TRIP_DAYS:
        st.error(f"여행 기간은 최대 {MAX_TRIP_DAYS}일입니다.")
    same_lodging = st.checkbox("전 기간 같은 숙소", value=True)
    lodging_query = st.text_input("숙소 검색", placeholder="해운대, 호텔 이름, 구…")

    lodgings: list[dict] = []
    lodging_error = ""
    try:
        lodgings = fetch_lodgings(lodging_query)
    except requests.exceptions.ConnectionError:
        lodging_error = "백엔드 서버에 연결할 수 없습니다. uvicorn이 실행 중인지 확인해 주세요."
    except requests.RequestException:
        lodging_error = "숙소 목록을 불러오지 못했습니다."

    if lodging_error:
        st.error(lodging_error)
        lodging_ids: list[str] = []
    elif not lodgings:
        st.warning("검색된 숙소가 없습니다. 검색어를 바꿔 보세요.")
        lodging_ids = []
    elif same_lodging:
        selected = selected_lodging_id("숙소", lodgings, "lodging_all")
        lodging_ids = [selected] if selected else []
    else:
        lodging_ids = []
        for offset in range(min(max(days, 0), MAX_TRIP_DAYS)):
            trip_day = start_date + timedelta(days=offset)
            selected = selected_lodging_id(
                f"{offset + 1}일차 ({trip_day.isoformat()}) 숙소",
                lodgings,
                f"lodging_day_{offset}",
            )
            if selected:
                lodging_ids.append(selected)

if st.button("코스 추천 받기 🚀"):
    if end_date < start_date:
        st.error("종료일은 시작일 이후여야 합니다.")
    elif days > MAX_TRIP_DAYS:
        st.error(f"여행 기간은 최대 {MAX_TRIP_DAYS}일입니다.")
    elif not lodging_ids:
        st.error("숙소를 선택해 주세요. 백엔드가 켜져 있는지도 확인해 주세요.")
    else:
        with st.spinner("AI가 맞춤 코스를 생성 중입니다..."):
            try:
                response = requests.post(
                    f"{API_BASE}/api/recommend",
                    json={
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "companion": companion,
                        "purpose": purpose,
                        "lodging_ids": lodging_ids,
                        "intensity": intensity,
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown(result.get("recommendation", ""))
                else:
                    detail = ""
                    try:
                        payload = response.json()
                        detail = payload.get("detail") or payload
                    except ValueError:
                        detail = response.text
                    st.error(f"추천 생성에 실패했습니다. {detail}")
            except requests.exceptions.ConnectionError:
                st.error("백엔드 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해 주세요.")
