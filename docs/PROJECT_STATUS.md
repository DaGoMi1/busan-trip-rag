# 부산 여행 코스 추천 — 진행 현황

> ML 엔지니어 포트폴리오용 LLM + RAG 여행 코스 추천 서비스  
> 최종 업데이트: 2026-09-09

---

## 1. 프로젝트 한 줄 요약

부산을 방문하는 관광객이 **기간 / 동행 / 취향 / 숙소·반경·강도**를 넣으면, TourAPI 장소 데이터를 **목적 가중 multi-recall**로 모은 뒤 LLM이 일정을 조합하고, 동선은 nearest-neighbor로 확정한다.

평가지표(Core)는 [`docs/EVAL.md`](EVAL.md)에 고정되어 있다.

---

## 2. 목표 아키텍처

```text
Streamlit  →  FastAPI  →  LangChain RAG
                              ├─ Multi-recall (카테고리별 쿼터 + 반경)
                              ├─ Routing (일차 bucket + NN)
                              ├─ LLM 조합 (후보 id 선택)
                              └─ NN 재정렬 + Faithfulness 가드
                 ↑
     data/processed/busan_places.jsonl
     data/processed/faiss_index/
```

| 구분 | 기술 |
|------|------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Retrieval | sentence-transformers + FAISS IndexFlatIP |
| Generation | LangChain + gpt-4o-mini (키 없으면 extractive) |
| 데이터 | 한국관광공사 TourAPI 4.0 (`KorService2`) |

---

## 3. 지금까지 한 일

### 3.1~3.5 요약

- TourAPI 수집, `busan_places.jsonl` 2,225건, FAISS 인덱스
- Streamlit → FastAPI `/api/recommend` → `CourseRag`
- 숙소 좌표 반경 필터, 활동 강도 필터, `OPENAI_API_KEY` 연결

### 3.6 추천 고도화

| 단계 | 내용 |
|------|------|
| 1차 | 일차 bucket, NN 동선, 음식 믹스, EVAL 스모크 |
| 2차 | **목적 가중 multi-recall** + LLM이 후보 중 조합 + id 파싱 후 NN 확정 |

주요 모듈: `preferences.PURPOSE_CATEGORY_QUOTAS`, `chain._search_weighted_hits`, `format_day_plans_by_category`, `apply_llm_selections`

---

## 4. 앞으로 할 일

### P0 — 코스 생성 (완료)

### P1 — 검색/코스 품질

- ~~목적 하드 필터 → 가중 multi-recall + LLM 조합~~ (완료)
- ~~동선 NN / 일차 중복 방지~~ (완료)
- 권역 클러스터링 고도화 — 후속
- 하이브리드 검색 (BM25 + FAISS) — 후속

### P2 — 데이터 보강 (필요할 때만)

- `detailIntro2` 메타데이터화

### P3 — 평가 / 서빙

- Core 지표: [`docs/EVAL.md`](EVAL.md) (ShoppingShare/Format/CategoryMatch 제거, PrimaryShare 추가)
- `python -m scripts.eval_smoke`
- 후속: Hit@K 자동화, Docker

---

## 5. 지금 안 하는 이유

| 안 한 것 | 이유 |
|----------|------|
| 전체 `detailIntro2` | 호출 한도 대비 이득이 작고, 임베딩 노이즈 |
| BM25 하이브리드 | multi-recall·LLM 조합이 우선 |
| 프론트 디자인 다듬기 | 추천 품질·지표 확인 뒤 |

---

## 6. 실행 명령

```bash
.venv\Scripts\activate

python -m scripts.eval_smoke
python -m uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

---

## 7. 다음 작업 한 줄

**EVAL 스모크 수치를 기록하고, 데모·README를 면접용으로 마감한 뒤 BM25 여부를 결정한다.**
