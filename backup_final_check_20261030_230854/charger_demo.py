
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import json

SAVE_PATH = Path("data/demo/jeju_chargers_snapshot.json")


def save_charger_snapshot(stations):
    """API에서 조회한 실제 충전소 정보를 저장한다."""
    if not stations:
        return False

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "saved_at": datetime.now(
            ZoneInfo("Asia/Seoul")
        ).isoformat(),
        "source": "한국환경공단 전기차 충전소 API",
        "stations": stations
    }

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            payload, f,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    return True


def load_charger_snapshot():
    """저장된 충전소 데이터를 읽는다."""
    if not SAVE_PATH.exists():
        return None, None

    with open(SAVE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return (
        payload.get("stations", []),
        payload.get("saved_at")
    )
