# 추천 품질 평가지표

> 고도화 전후로 같은 시나리오·같은 지표로 비교한다.  
> Core 지표만 유지한다.

실행:

```bash
python -m scripts.eval_smoke
```

---

## Core 지표

| 지표 | 정의 | 목표 |
|------|------|------|
| RadiusCompliance | 일차 확정 장소 중 숙소 반경 안 비율 | ≈ 1.0 |
| CrossDayOverlap | 일차 간 동일 place id 비율 | 0 |
| FoodMixRate | 하루 일정에 음식점 ≥1 포함 비율 | 상승 |
| DayCompactness | 하루 장소 centroid 대비 평균 거리(km) | 하락 |
| PrimaryShare | 하루 일정 중 목적 주 카테고리 비율 | 목적에 맞게 높게 |
| Faithfulness | 출력 장소명이 후보(context)에만 있는지 (LLM) | 위반 0 |

---

## 시나리오 (S1–S3)

| ID | 숙소 힌트 | 일수 | 목적 | 반경 | 강도 | 기대 |
|----|-----------|------|------|------|------|------|
| S1 | 영도 | 2 | 자연/액티비티 | 2 (~6km) | 5 | 반경 준수, 음식 ≥1, 중복 0, PrimaryShare 높음 |
| S2 | 해운대 | 3 | 문화관광 | 3 (~10km) | 3 | 문화/관광 비중 + 식사 |
| S3 | 서면 | 2 | 맛집 | 2 (~6km) | 3 | PrimaryShare≈음식점 |

수동 체크(LLM): Faithfulness — 후보에 없는 상호명 없음

---

## 기록

`python -m scripts.eval_smoke` (2026-09-09, 목적 가중 multi-recall 후):

| 시나리오 | RadiusComp | DayCompact | CrossOverlap | FoodMix | PrimaryShare | 메모 |
|----------|------------|------------|--------------|---------|--------------|------|
| S1 | 1.0 | 1.32 | 0.0 | 1.0 | 0.86 | 영도·자연/액티비티 |
| S2 | 1.0 | 0.57 | 0.0 | 1.0 | 0.80 | 해운대·문화관광 |
| S3 | 1.0 | 1.59 | 0.0 | 1.0 | 1.0 | 서면·맛집 |