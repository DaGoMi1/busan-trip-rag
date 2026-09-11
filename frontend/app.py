from datetime import date, timedelta

import requests
import streamlit as st

API_BASE = "http://localhost:8000"
MAX_TRIP_DAYS = 7
PURPOSES = ["힐링", "맛집", "문화관광", "쇼핑", "자연/액티비티", "종합"]
COMPANIONS = ["혼자", "커플", "친구", "가족", "기타"]
INTENSITY_LABELS = {
    1: "낮음 · 카페·실내·완만한 산책",
    2: "가벼운 산책 · 골목·전망·여유",
    3: "보통 · 일반적인 도보 관광",
    4: "활발 · 언덕·긴 도보 OK",
    5: "높음 · 산·숲·트레킹 가능",
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


def pick_lodging_id(
    *,
    select_label: str,
    select_key: str,
    query_key: str,
    query_label: str = "숙소 검색",
) -> tuple[str | None, str]:
    """검색어와 selectbox를 한 세트로 두고, 고른 숙소 id를 반환합니다."""
    query = st.text_input(
        query_label,
        key=query_key,
        placeholder="해운대, 호텔 이름, 구…",
    )
    try:
        lodgings = fetch_lodgings(query)
    except requests.exceptions.ConnectionError:
        return None, "백엔드 서버에 연결할 수 없습니다. uvicorn이 실행 중인지 확인해 주세요."
    except requests.RequestException:
        return None, "숙소 목록을 불러오지 못했습니다."

    if not lodgings:
        st.warning("검색된 숙소가 없습니다. 검색어를 바꿔 보세요.")
        return None, ""

    labels = [lodging_label(item) for item in lodgings]
    choice = st.selectbox(select_label, labels, key=select_key)
    return str(lodgings[labels.index(choice)]["id"]), ""


with st.sidebar:
    st.header("여행 정보 입력")
    start_date = st.date_input("시작일", value=date.today())
    end_date = st.date_input("종료일", value=date.today() + timedelta(days=1))
    purpose = st.selectbox("여행 목적", PURPOSES)
    companion = st.selectbox("동행자", COMPANIONS)
    intensity = st.slider("활동 강도", min_value=1, max_value=5, value=3)
    st.caption(INTENSITY_LABELS[intensity])
    st.caption(
        "탐색은 전날 숙소 또는 당일 숙소 각 2.5km 이내(OR). "
        "같은 숙소에 여러 날 묵으면 2.5→4→5.5km로 조금씩 넓어집니다."
    )

    days = (end_date - start_date).days + 1
    if end_date < start_date:
        st.error("종료일은 시작일 이후여야 합니다.")
    elif days > MAX_TRIP_DAYS:
        st.error(f"여행 기간은 최대 {MAX_TRIP_DAYS}일입니다.")

    same_lodging = st.checkbox("전 기간 같은 숙소", value=True)
    lodging_ids: list[str] = []
    lodging_error = ""

    if end_date >= start_date and days <= MAX_TRIP_DAYS:
        if same_lodging:
            selected, lodging_error = pick_lodging_id(
                select_label="숙소",
                select_key="lodging_all",
                query_key="lodging_q_all",
            )
            if selected:
                lodging_ids = [selected]
        else:
            st.caption("일차마다 숙소를 따로 검색·선택할 수 있어요.")
            for offset in range(min(max(days, 0), MAX_TRIP_DAYS)):
                trip_day = start_date + timedelta(days=offset)
                day_label = f"{offset + 1}일차 ({trip_day.isoformat()})"
                selected, err = pick_lodging_id(
                    select_label=f"{day_label} 숙소",
                    select_key=f"lodging_day_{offset}",
                    query_key=f"lodging_q_{offset}",
                    query_label=f"{day_label} 숙소 검색",
                )
                if err:
                    lodging_error = err
                    lodging_ids = []
                    break
                if selected:
                    lodging_ids.append(selected)

    if lodging_error:
        st.error(lodging_error)

if st.button("코스 추천 받기 🚀"):
    if end_date < start_date:
        st.error("종료일은 시작일 이후여야 합니다.")
    elif days > MAX_TRIP_DAYS:
        st.error(f"여행 기간은 최대 {MAX_TRIP_DAYS}일입니다.")
    elif not lodging_ids:
        st.error("숙소를 선택해 주세요. 백엔드가 켜져 있는지도 확인해 주세요.")
    elif (not same_lodging) and len(lodging_ids) != days:
        st.error("모든 일차의 숙소를 각각 선택해 주세요.")
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
                    st.divider()
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
