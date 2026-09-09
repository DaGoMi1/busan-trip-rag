# 부산 여행 코스 추천

부산을 방문하는 관광객이 **날짜, 동행, 목적, 숙소, 활동 강도**를 넣으면  
TourAPI 장소 데이터를 RAG로 검색해 **전날·당일 숙소 근처**에서 목적에 맞는 일차별 코스를 만듭니다.

`.env`에 `OPENAI_API_KEY`가 있으면 `gpt-4o-mini`가 일정을 조합하고,  
없으면 FAISS 검색 초안만 반환합니다.

---

## 주요 기능

- 여행 시작일/종료일(최대 7일), 동행자, 목적에 따른 맞춤 검색
- TourAPI 숙박을 고르면 **전날 숙소 + 당일 숙소**를 탐색 앵커로 사용 (OR)
- **활동 강도 1~5만** 입력 → 하루 장소 수·활동 유형·hop 상한 (반경 슬라이더 없음)
- 탐색 반경 **기본 2.5km**, 같은 숙소 연속 투숙 시 **2.5 → 3.5 → 4.5…km** 확대 (cap 8km)
- 목적별 카테고리 쿼터·점수 가산 multi-recall
- 숙소가 바뀌는 날: 동선을 **전날 숙소 → 당일 숙소** 방향으로 정렬
- Streamlit → FastAPI `POST /api/recommend`
- 품질 스모크: `python -m scripts.eval_smoke` ([docs/EVAL.md](docs/EVAL.md))

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| Frontend | Streamlit |
| Backend | FastAPI |
| ML / AI | LangChain, sentence-transformers, FAISS, RAG |
| 데이터 | 한국관광공사 TourAPI 4.0 (`KorService2`) |
| Language | Python |

---

## 시스템 아키텍처

```mermaid
flowchart LR
    User["사용자"] --> Streamlit["Streamlit"]
    Streamlit --> FastAPI["FastAPI"]
    FastAPI --> RAG["LangChain RAG"]
    RAG --> FAISS["FAISS"]
    RAG --> LLM["gpt-4o-mini 선택"]
    FAISS --> RAG
    LLM --> RAG
    RAG --> FastAPI
    FastAPI --> Streamlit
    Streamlit --> User
```

추천 파이프라인:

1. 일차마다 **전날·당일 숙소** 반경 안에서 목적 쿼터별 multi-recall
2. 주 카테고리 점수 가산 → 후보 풀 구성 (이전 일차 place id 제외)
3. LLM이 `SELECTED_IDS_DAY{n}`으로 조합 (키 없으면 `ensure_daily_mix`)
4. 동선 정렬: 동일 숙소면 NN, **이동일이면 전날→당일 방향** + hop 상한

핵심 모듈: `ml/rag/preferences.py` (규칙) · `retriever.py` (검색) · `chain.py` (오케스트레이션) · `routing.py` (동선)

---

## 사용자 입력

| 필드 | 설명 |
|------|------|
| `start_date`, `end_date` | 여행 기간. 포함 일수 1~7일 |
| `companion` | 혼자 / 커플 / 친구 / 가족 / 기타 |
| `purpose` | 힐링 / 맛집 / 문화관광 / 쇼핑 / 자연/액티비티 / 종합 |
| `lodging_ids` | TourAPI 숙박 `id`. 1개면 전 기간 동일, 아니면 일차 수만큼 |
| `intensity` | 1~5. 하루 장소 수 · 산·숲 등 활동 유형 · hop 상한 |

### 숙소 앵커 · 반경

| 동일 숙소 연속일 | 반경 |
|------------------|------|
| 1일차 (stay 0) | 2.5km |
| 2일차 (stay 1) | 3.5km |
| 3일차 (stay 2) | 4.5km |
| … | cap 8km |

- 후보는 **전날 숙소 반경 안**이거나 **당일 숙소 반경 안**이면 통과 (OR).
- 숙소 변경일 동선: 체크아웃(전날) → 중간 장소 → 체크인(당일) 방향으로 정렬.
- 다음 날 숙소가 바뀌면 검색 점수에 다음 숙소 근접 가점도 적용.

관련 API:

- `GET /api/lodgings?q=` — 숙박 검색 (이름·구·주소)
- `POST /api/recommend` — 코스 생성

---

## 프로젝트 구조

```
.
├── README.md
├── requirements.txt
├── .env
├── frontend/app.py
├── backend/
│   ├── main.py
│   ├── routers/recommend.py
│   ├── schemas/request.py
│   └── services/recommend_service.py
├── ml/
│   ├── rag/                 # chain, retriever, routing, preferences, prompt, metrics
│   ├── embeddings/
│   └── vectorstore/
├── docs/
│   ├── EVAL.md
│   └── PROJECT_STATUS.md
├── scripts/
│   ├── collect_tourapi.py
│   ├── build_faiss.py
│   └── eval_smoke.py
└── data/
    ├── pipeline/
    ├── raw/
    └── processed/           # busan_places.jsonl, faiss_index/
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 환경 변수

```bash
cp .env.example .env
```

| 변수 | 용도 |
|------|------|
| `TOUR_API_KEY` | TourAPI 수집. 이미 `data/processed/`가 있으면 실행만 할 때는 없어도 됨 |
| `OPENAI_API_KEY` | 코스 문장 생성. 없으면 검색 초안만 반환 |

`.env`는 Git에 올리지 않습니다.

### 3. 데이터

실행에 필요한 것은 `data/processed/`(RAG 문서 + FAISS)입니다. 레포에 포함합니다.  
TourAPI 원본 `data/raw/`는 용량이 커서 Git에 올리지 않습니다. 다시 수집할 때만 아래를 쓰면 됩니다.

공공데이터포털에서 `한국관광공사_국문 관광정보 서비스_GW`를 신청한 뒤 `TOUR_API_KEY`를 넣습니다.  
부산은 구 `areaCode=6` 대신 법정동 `lDongRegnCd=26`을 사용합니다.

```bash
python -m scripts.collect_tourapi
python -m scripts.build_faiss
python -m scripts.build_faiss --skip-build --query "해운대 일몰 데이트"
```

- 원본(로컬만): `data/raw/`
- RAG 문서: `data/processed/busan_places.jsonl` (약 2,225건)
- 인덱스: `data/processed/faiss_index/`

### 4. 실행

서버가 켜질 때 임베딩 모델과 FAISS를 한 번 로드합니다.  
`.env`를 바꾼 뒤에는 **uvicorn을 다시 시작**하세요.

```bash
python -m uvicorn backend.main:app --reload
```

다른 터미널:

```bash
python -m streamlit run frontend/app.py
```

- API: http://localhost:8000
- UI: http://localhost:8501

포트 8000이 이미 사용 중이면 이전 uvicorn을 종료하거나 `--port 8001`을 쓰고, 프론트의 `API_BASE`를 맞춥니다.

### 5. 평가 스모크

```bash
python -m scripts.eval_smoke
```

지표·시나리오·기록은 [docs/EVAL.md](docs/EVAL.md)를 참고하세요.

---

## 라이선스

MIT License
