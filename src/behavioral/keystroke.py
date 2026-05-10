"""
Keystroke and paste detection analysis.

Defines the interface and scoring logic for detecting paste-based submissions
vs. incremental typing. Requires the platform to send keystroke event metadata.

This module processes platform-provided keystroke data — it does not capture
keystrokes itself.
"""

from loguru import logger


def analyse_edit_events(events: list[dict]) -> dict:
    """
    Analyse a sequence of editor events to detect paste-based submission.

    Expected event format:
        {
            "type": "insert" | "delete" | "paste" | "undo" | "redo",
            "timestamp": float (epoch ms),
            "length": int (characters affected),
            "content": str (optional, the text involved),
        }

    Returns:
        {
            "total_events": int,
            "paste_count": int,
            "paste_char_ratio": float,    # chars pasted / total chars inserted
            "is_single_paste": bool,      # entire code was a single paste
            "edit_duration_seconds": float,
            "events_per_minute": float,
            "incremental_edit_ratio": float,  # small edits / total edits
            "keystroke_score": float,      # 0-1, higher = more suspicious
        }
    """
    if not events:
        return _empty_features()

    timestamps = [e.get("timestamp", 0) for e in events]
    duration_ms = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
    duration_sec = duration_ms / 1000.0

    total_inserted = 0
    paste_chars = 0
    paste_count = 0
    small_edits = 0

    for event in events:
        etype = event.get("type", "")
        length = event.get("length", 0)

        if etype == "paste":
            paste_count += 1
            paste_chars += length
            total_inserted += length
        elif etype == "insert":
            total_inserted += length
            if length <= 3:
                small_edits += 1

    total_events = len(events)
    paste_char_ratio = paste_chars / total_inserted if total_inserted > 0 else 0.0
    is_single_paste = paste_count == 1 and paste_char_ratio > 0.9

    insert_events = sum(1 for e in events if e["type"] in ("insert", "paste"))
    incremental_ratio = small_edits / insert_events if insert_events > 0 else 0.0

    events_per_minute = (total_events / duration_sec * 60) if duration_sec > 0 else 0.0

    # Scoring: combine paste ratio and lack of incremental editing
    score = 0.0
    if is_single_paste:
        score = 0.95
    elif paste_char_ratio > 0.7:
        score = 0.5 + 0.3 * paste_char_ratio
    elif paste_char_ratio > 0.3:
        score = 0.2 + 0.3 * paste_char_ratio
    else:
        score = max(0.0, 0.1 * (1.0 - incremental_ratio))

    if duration_sec < 10 and total_inserted > 100:
        score = max(score, 0.8)

    score = max(0.0, min(1.0, score))

    return {
        "total_events": total_events,
        "paste_count": paste_count,
        "paste_char_ratio": round(paste_char_ratio, 4),
        "is_single_paste": is_single_paste,
        "edit_duration_seconds": round(duration_sec, 2),
        "events_per_minute": round(events_per_minute, 2),
        "incremental_edit_ratio": round(incremental_ratio, 4),
        "keystroke_score": round(score, 4),
    }


def extract_features(metadata: dict) -> dict:
    """
    Public interface: extract keystroke features from submission metadata.

    Expected metadata key: "edit_events" -> list of event dicts.
    """
    events = metadata.get("edit_events", [])
    if not events:
        return _empty_features()
    return analyse_edit_events(events)


def _empty_features() -> dict:
    return {
        "total_events": 0,
        "paste_count": 0,
        "paste_char_ratio": 0.0,
        "is_single_paste": False,
        "edit_duration_seconds": 0.0,
        "events_per_minute": 0.0,
        "incremental_edit_ratio": 0.0,
        "keystroke_score": 0.0,
    }
