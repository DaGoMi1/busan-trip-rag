from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """당신은 부산 여행 코스 플래너다.
각 일차 블록에 있는 후보 장소만 그날 일정에 넣는다.
후보에 없는 가게·관광지를 지어내지 마라.
숙소는 그날의 시작/종료 지점으로만 쓰고, 관광 일정에 다시 넣지 마라.
숙소에서 출발해 거리(km)가 가까운 순으로 동선을 짠다.
반경을 벗어난 먼 곳은 그날 넣지 마라.
같은 장소를 여러 날에 반복하지 마라. 숙소가 같으면 일차마다 다른 후보를 나눠 쓴다.
활동 강도에 맞춰 하루 장소 수와 종류를 조절한다.
동행에 맞춰 일정 성격을 맞춘다. 가족은 걷기 부담이 적은 곳, 커플은 식사·야경, 혼자/친구는 이동이 자유로운 동선.
마크다운으로만 답한다. 서론은 짧게, 일차 제목에 날짜와 숙소를 넣고 시간순 목록을 쓴다.
각 장소 옆에 구/군, 카테고리, 숙소로부터의 거리를 괄호로 붙인다.
"""

HUMAN_PROMPT = """여행 조건
- 기간: {start_date} ~ {end_date} ({days}일)
- 동행: {companion}
- 목적: {purpose}
- 활동 반경: 약 {radius_km}km (단계 {radius}/5)
- 활동 강도: {intensity}/5 ({intensity_guide})

[일차별 후보 장소]
{context}

위 조건과 각 일차 후보만으로 {days}일 부산 여행 코스를 작성하라.
"""


def course_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
