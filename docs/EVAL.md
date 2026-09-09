# 추천 품질 평가지표

> 고도화 전후로 같은 시나리오·같은 지표로 비교한다.  
> Core 지표만 유지한다.

실행:

```bash
python -m scripts.eval_smoke
# LLM Faithfulness까지 보려면
python -m scripts.eval_smoke --with-llm
```

---

## Core 지표

| 지표 | 정의 | 목표 |
|------|------|------|
| BandCompliance | 확정 장소가 **전날·당일 숙소 앵커 중 하나**라도 그날 `radius_km` 이내(OR) | ≈ 1.0 |
| CrossDayOverlap | 일차 간 동일 place id 비율 | 0 |
| FoodMixRate | 하루 일정에 음식점 ≥1 포함 비율 | 상승 (맛집·종합에서 특히) |
| DayCompactness | 하루 장소 centroid 대비 평균 거리(km) | 하락 |
| PrimaryShare | 하루 일정 중 목적 주 카테고리 비율 | 목적에 맞게 높게 |
| Faithfulness | 출력 장소명이 후보(context)에만 있는지 (LLM, `--with-llm`) | 위반 0 |

구현: `ml/rag/metrics.py`, 스모크: `scripts/eval_smoke.py`

---

## 제품 규칙 (평가와 맞춘 기준)

| 규칙 | 내용 |
|------|------|
| 입력 | 활동 **반경 UI 없음**. `intensity` 1~5만 (하루 장소 수·활동 유형·hop) |
| 반경 | 기본 **2km**. 동일 숙소 연속 투숙 시 **2→3→4…km** (cap 8) |
| 앵커 | **전날 숙소 OR 당일 숙소** 반경 안이면 후보 |
| 목적 | 카테고리 쿼터 + 주 카테고리 점수 가산 → PrimaryShare 상향 |
| 동선 | multi-recall → (LLM id 선택) → NN + hop 상한 |

---

## 시나리오 (S1–S3)

| ID | 숙소 힌트 | 일수 | 목적 | 강도 | 기대 |
|----|-----------|------|------|------|------|
| S1 | 영도 | 2 | 자연/액티비티 | 5 | Band≈1, CrossOverlap=0, PrimaryShare 높음, stay 2→3km |
| S2 | 해운대 | 3 | 문화관광 | 3 | stay 2→3→4, PrimaryShare≈문화/관광 |
| S3 | 서면 | 2 | 맛집 | 3 | PrimaryShare≈1.0 (음식점), FoodMix≈1.0 |

---

## 기록

`python -m scripts.eval_smoke` (2026-09-09, 이중 앵커 · 2→3→4km · 목적 가중 상향):

| 시나리오 | BandComp | DayCompact | CrossOverlap | FoodMix | PrimaryShare | 메모 |
|----------|----------|------------|--------------|---------|--------------|------|
| S1 | 1.0 | 0.99 | 0.0 | 1.0 | 0.79 | 영도, r=2→3km |
| S2 | 1.0 | 0.87 | 0.0 | 0.0 | 1.0 | 해운대 2→3→4; 좁은 반경+문화 목적 → 식사 후보 부족 가능 |
| S3 | 1.0 | 0.27 | 0.0 | 1.0 | 1.0 | 서면·맛집 음식점만 |

### 수동 데모 (참고)

연제구 → 영도구 숙소 변경, 맛집·강도 4, 2일: 일차별 음식점 중심·hop 대부분 1km 이내·숙소 권역과 일치. Faithfulness는 `--with-llm`로 별도 확인.
