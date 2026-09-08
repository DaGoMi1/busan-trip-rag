import time
from typing import Any
from urllib.parse import quote, unquote, urlencode

import requests
from requests.exceptions import HTTPError, RequestException

from data.pipeline.config import BASE_URL


class TourAPIError(RuntimeError):
    pass


class RateLimitError(TourAPIError):
    pass


class DailyQuotaError(RateLimitError):
    pass


def _portal_error(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    header = payload.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader")
    return header if isinstance(header, dict) else None


def _raise_portal_error(endpoint: str, header: dict[str, Any]) -> None:
    code = str(header.get("returnReasonCode", ""))
    msg = header.get("returnAuthMsg") or header.get("errMsg") or ""
    if code == "22":
        raise DailyQuotaError(
            f"{endpoint} 일일 호출 한도 초과 (22). 내일 자정 이후 다시 실행하세요."
        )
    raise TourAPIError(f"{endpoint} 인증/호출 실패: {code} {msg}")


def encode_service_key(raw_key: str) -> str:
    """Encoding/Decoding 키를 URL 쿼리용으로 맞춥니다.

    - Encoding 키(%2B 등 포함): 추가 인코딩 없이 그대로 사용
    - Decoding 키: 한 번만 percent-encode
    """
    key = raw_key.strip().strip('"').strip("'")
    if "%" in key:
        return key
    return quote(unquote(key), safe="")


class TourAPIClient:
    """TourAPI 4.0 KorService2 클라이언트."""

    def __init__(
        self,
        service_key: str,
        mobile_app: str = "BusanCourseRecommender",
        timeout: int = 20,
        sleep_sec: float = 1.5,
        max_retries: int = 4,
    ) -> None:
        self.service_key = encode_service_key(service_key)
        self.timeout = timeout
        self.sleep_sec = sleep_sec
        self.max_retries = max_retries
        self.base_params = {
            "MobileOS": "ETC",
            "MobileApp": mobile_app,
            "_type": "json",
        }
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; BusanCourseRecommender/0.1)"}
        )

    def get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        clean_params = {k: v for k, v in params.items() if v not in (None, "")}
        query = urlencode({**self.base_params, **clean_params})
        url = f"{BASE_URL}/{endpoint.lstrip('/')}?serviceKey={self.service_key}&{query}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout)
            except RequestException as exc:
                last_error = exc
                wait = min(60, 2 ** attempt)
                print(f"[retry] {endpoint} 네트워크 오류, {wait}s 대기 ({attempt}/{self.max_retries})")
                time.sleep(wait)
                continue

            if response.status_code == 429:
                try:
                    header = _portal_error(response.json())
                except ValueError:
                    header = None
                if header:
                    _raise_portal_error(endpoint, header)
                wait = _retry_after_seconds(response, default=min(180, 90 * attempt))
                print(f"[retry] 429 Too Many Requests, {wait}s 대기 ({attempt}/{self.max_retries})")
                time.sleep(wait)
                last_error = HTTPError("429 Too Many Requests")
                continue

            if response.status_code >= 500:
                wait = min(60, 2 ** attempt)
                print(f"[retry] {endpoint} HTTP {response.status_code}, {wait}s 대기 ({attempt}/{self.max_retries})")
                time.sleep(wait)
                last_error = HTTPError(f"HTTP {response.status_code}")
                continue

            if response.status_code >= 400:
                raise TourAPIError(f"{endpoint} HTTP {response.status_code}")

            payload = response.json()
            auth_error = _portal_error(payload)
            if auth_error:
                _raise_portal_error(endpoint, auth_error)

            header = payload.get("response", {}).get("header", {})
            result_code = str(header.get("resultCode", ""))
            if result_code != "0000":
                raise TourAPIError(
                    f"{endpoint} 실패: {result_code} {header.get('resultMsg')}"
                )

            time.sleep(self.sleep_sec)
            return payload.get("response", {}).get("body", {})

        raise RateLimitError(f"{endpoint} 재시도 초과: {last_error}")

    def fetch_all_pages(
        self,
        endpoint: str,
        num_of_rows: int = 100,
        **params: Any,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_no = 1

        while True:
            body = self.get(
                endpoint,
                pageNo=page_no,
                numOfRows=num_of_rows,
                **params,
            )
            page_items = normalize_items(body.get("items", {}))
            items.extend(page_items)

            total_count = int(body.get("totalCount") or 0)
            if page_no * num_of_rows >= total_count or not page_items:
                break
            page_no += 1

        return items


def _retry_after_seconds(response: requests.Response, default: float) -> float:
    header = response.headers.get("Retry-After")
    if not header:
        return default
    try:
        return max(default, float(header))
    except ValueError:
        return default


def normalize_items(items: Any) -> list[dict[str, Any]]:
    """TourAPI items.item 이 단건 dict / 다건 list / 빈값일 수 있어 통일합니다."""
    if not items:
        return []

    item = items.get("item") if isinstance(items, dict) else items
    if item is None:
        return []
    if isinstance(item, list):
        return [x for x in item if isinstance(x, dict)]
    if isinstance(item, dict):
        return [item]
    return []
