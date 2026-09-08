# 부산 여행 코스 추천 — 진행 현황

> ML 엔지니어 포트폴리오용 LLM + RAG 여행 코스 추천 서비스  
> 최종 업데이트: 2026-09-07

---

## 1. 프로젝트 한 줄 요약

부산을 방문하는 관광객이 **기간 / 동행 / 취향**을 넣으면, TourAPI 장소 데이터를 RAG로 검색해 **실현 가능한 여행 코스**를 LLM이 생성한다.

현재는 **데이터 수집 + FAISS 검색 + LangChain RAG 체인**까지 완료했고, 권역/동선 고도화와 LLM 키 연결이 남아 있다.

---

## 2. 목표 아키텍처

```text
Streamlit  →  FastAPI  →  LangChain RAG
                              ├─ Retriever (FAISS + 메타데이터 필터)
                              ├─ Prompt (일정/동선 제약)
                              └─ LLM
                 ↑
     data/processed/busan_places.jsonl
     data/processed/faiss_index/
```

| 구분 | 기술 |
|------|------|
| Frontend | Streamlit (골격만 있음) |
| Backend | FastAPI (골격만 있음) |
| Retrieval | sentence-transformers + FAISS |
| Generation | LangChain + LLM (**미연결**) |
| 데이터 | 한국관광공사 TourAPI 4.0 (`KorService2`) |

---

## 3. 지금까지 한 일

### 3.1 프로젝트 골격

- Streamlit / FastAPI / ML 디렉토리 분리
- 추천 API 스키마: 기간, 동행자, 스타일, 예산
- 프론트는 백엔드 `POST /api/recommend`를 호출하지만, 지금은 더미 문구만 반환

### 3.2 TourAPI 수집 파이프라인

TourAPI 4.0 기준으로 **구 `areaCode=6` 대신 법정동 `lDongRegnCd=26`(부산)** 을 사용한다.

| 모듈 | 역할 |
|------|------|
| `data/pipeline/client.py` | API 호출, Encoding/Decoding 키 처리, 일일 한도(22) 즉시 중단 |
| `data/pipeline/fetch.py` | 목록/상세 수집, 목록·상세 캐시, 이어받기 |
| `data/pipeline/parse.py` | 목록·상세 정규화, 구/군 매핑 |
| `data/pipeline/clean.py` | RAG 문서(`page_content` + metadata) 생성 |
| `scripts/collect_tourapi.py` | 수집 CLI |

수집 대상 contentType:

- 12 관광지, 14 문화시설, 15 축제, 28 레포츠, 32 숙박, 38 쇼핑, 39 음식점

상세는 `detailCommon2`의 **overview / homepage만** 받았다.  
이용시간·주차·메뉴(`detailIntro2`)는 의도적으로 보류했다.

### 3.3 수집하면서 겪은 이슈

| 이슈 | 대응 |
|------|------|
| HTTP 403 / 키 미등록 | GW 서비스 활용신청 + Encoding/Decoding 키 구분 |
| `areaCode=6` → `totalCount=0` | `lDongRegnCd=26`으로 전환 |
| HTTP 429 | 재시도 + 상세 캐시 |
| 일일 한도 초과 (코드 22) | 재시도 중단, 다음날 이어서 수집 |
| Windows 한글 경로 + FAISS 파일 저장 실패 | 인덱스를 바이트로 직렬화해 저장 |

### 3.4 확보한 데이터

| 산출물 | 내용 |
|--------|------|
| `data/raw/area_based_list_*.json` | 타입별 원본 목록 |
| `data/raw/detail_common_cache.json` | 장소별 `detailCommon2` 원본 (이어받기) |
| `data/processed/busan_places.jsonl` | RAG용 최종 문서 **2,225건** |

문서 필드: 이름, 카테고리, 구/군, 권역, 주소, 좌표, overview, `page_content`, metadata.

쇼핑(38)이 980건으로 가장 많다. 같은 아울렛의 브랜드 매장이 다수 포함되어 있다. 수집 중복이 아니라 TourAPI 원본 특성이다.

### 3.5 FAISS 검색 (완료)

| 모듈 | 역할 |
|------|------|
| `ml/embeddings/embedding_model.py` | `jhgan/ko-sroberta-multitask` |
| `ml/vectorstore/faiss_store.py` | IndexFlatIP (768차원, cosine용 정규화) |
| `ml/rag/retriever.py` | 쿼리 검색 + 카테고리/권역 필터 |
| `scripts/build_faiss.py` | 인덱스 구축 / 스모크 쿼리 |

인덱스 위치: `data/processed/faiss_index/` (2,225 벡터)

검증 예시:

- `돼지국밥 맛집` → 돼지국밥 식당 Top-5 (score ~0.73–0.76)
- 자연 경관 문장 쿼리 → 해안산책로·해파랑길 등 (score ~0.53–0.56)

검색은 된다. 다만 아직 **코스를 만들지 않고**, 장소가 구 단위로 흩어질 수 있다.

---

## 4. 앞으로 할 일

우선순위는 ML 엔지니어 포트폴리오 기준이다. 위가 핵심, 아래는 고도화다.

### P0 — 코스가 나오게 만들기 (완료)

- 프롬프트: `ml/rag/prompt_templates.py` (후보 장소만 사용, 권역 유지)
- RAG 체인: `ml/rag/chain.py` (스타일→카테고리 매핑 후 FAISS Top-K → LLM)
- FastAPI: 앱 시작 시 FAISS+임베딩 **1회 로드**, `/api/recommend`가 체인 호출
- `OPENAI_API_KEY`가 있으면 gpt-4o-mini로 코스 문장 생성
- 키가 없으면 검색 결과를 권역별로 나눈 초안 마크다운 반환

### P1 — 검색/코스 품질 (다음 고도화)

- **메타데이터 필터**: 스타일→카테고리 매핑 (맛집→음식점, 자연→관광지/레포츠)
- **권역 클러스터링**: 2박 3일이면 동부산/원도심처럼 하루 단위로 권역을 고정
- **동선**: 위경도로 가까운 순서 정렬, 해운대↔사하 왕복 방지
- 쇼핑 과다 시 인덱스/검색에서 쇼핑 제외 또는 하향
- 하이브리드 검색 (BM25 + FAISS) — 고유명사(밀면, 감천문화마을) 보강
- Reranker (선택)

### P2 — 데이터 보강 (필요할 때만)

- `detailIntro2`: 관광지 이용시간/휴무, 음식점 영업시간·대표메뉴  
  → 임베딩 텍스트가 아니라 **메타데이터**로 두고 코스 제약에 사용
- 주차는 우선 생략

### P3 — 평가 / 서빙 (이력서에 숫자 넣을 때)

- 검색 평가: 샘플 쿼리 세트, Hit@K / 권역 일치율
- RAG 평가: Faithfulness (가짜 장소 여부)
- FastAPI 스트리밍, Docker 배포

---

## 5. 지금 안 하는 이유

| 안 한 것 | 이유 |
|----------|------|
| 전체 `detailIntro2` | 호출 한도 대비 이득이 작고, 임베딩 노이즈 |
| Naive RAG만으로 코스 생성 | 동선 환각이 남음 → P1에서 좌표/권역으로 보정할 예정 |
| 프론트 디자인 다듬기 | 추천 품질이 나온 뒤 |

---

## 6. 실행 명령

```bash
# 가상환경
.venv\Scripts\activate

# 데이터 수집 (목록 캐시 + 상세 이어받기)
python -m scripts.collect_tourapi

# FAISS 구축
python -m scripts.build_faiss

# 검색만 테스트
python -m scripts.build_faiss --skip-build --query "돼지국밥 맛집"

# 서버 (아직 더미 추천)
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

---

## 7. 다음 작업 한 줄

**권역/좌표 기준으로 하루 동선을 묶고, OpenAI 키를 넣어 LLM 코스 품질을 확인한다.**
