"""소형 평가 스모크: docs/EVAL.md Core 시나리오(S1–S3).

실행:
  python -m scripts.eval_smoke
  python -m scripts.eval_smoke --with-llm
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from backend.schemas.request import RecommendRequest
from ml.rag.chain import CourseRag
from ml.rag.metrics import summarize_metrics

SCENARIOS = [
    {
        "id": "S1",
        "lodging_query": "영도",
        "days": 2,
        "purpose": "자연/액티비티",
        "intensity": 5,
        "companion": "혼자",
    },
    {
        "id": "S2",
        "lodging_query": "해운대",
        "days": 3,
        "purpose": "문화관광",
        "intensity": 3,
        "companion": "커플",
    },
    {
        "id": "S3",
        "lodging_query": "서면",
        "days": 2,
        "purpose": "맛집",
        "intensity": 3,
        "companion": "친구",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="부산 RAG 코스 평가 스모크")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="OPENAI_API_KEY가 있으면 Faithfulness도 측정",
    )
    return parser.parse_args()


def pick_lodging(rag: CourseRag, query: str) -> dict:
    lodgings = rag.retriever.list_lodgings(query)
    if not lodgings:
        raise RuntimeError(f"숙소를 찾지 못했습니다: {query}")
    return lodgings[0]


def run_scenario(rag: CourseRag, scenario: dict, with_llm: bool) -> None:
    lodging = pick_lodging(rag, scenario["lodging_query"])
    start = date.today()
    end = start + timedelta(days=scenario["days"] - 1)
    request = RecommendRequest(
        start_date=start,
        end_date=end,
        companion=scenario["companion"],
        purpose=scenario["purpose"],
        lodging_ids=[str(lodging["id"])],
        intensity=scenario["intensity"],
    )
    lodgings = rag.resolve_lodgings(request.lodging_ids)
    candidates = rag._retrieve_by_day(request, lodgings, for_llm=True)
    day_plans = rag._narrow_plans(request, candidates)
    llm_text = None
    if with_llm and os.getenv("OPENAI_API_KEY"):
        llm_text = rag.recommend(request)
    metrics = summarize_metrics(
        day_plans,
        purpose=request.purpose,
        intensity=request.intensity,
        llm_text=llm_text,
    )
    print(f"\n=== {scenario['id']} | {scenario['lodging_query']} | {scenario['purpose']} ===")
    print(f"lodging: {lodging.get('name')} ({lodging.get('district')})")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    for index, plan in enumerate(day_plans, start=1):
        band = plan.get("radius_km")
        band_txt = f" r≤{band}km" if band is not None else ""
        names = [f"{h.get('name')}({h.get('category')})" for h in plan["hits"]]
        print(f"  day{index}{band_txt}: {', '.join(names) if names else '(empty)'}")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    print("Loading FAISS + embedding…")
    rag = CourseRag.from_disk()
    for scenario in SCENARIOS:
        try:
            run_scenario(rag, scenario, with_llm=args.with_llm)
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {scenario['id']} FAILED: {exc}")


if __name__ == "__main__":
    main()
