"""Build day-windows from raw messages.

The unit you label must be the unit production scores, or the POC measures a task the
app never performs. OrgPulse scores at two granularities, and both are reproduced here
exactly as the production code builds them:

  channel  mirrors CliqService.analyze_channel_messages
           "[{user_name}]: {text}" joined by newline
  user     mirrors CliqService._upsert_individual_scores_from_docs
           raw text joined by newline, no name prefix

The name prefix is not cosmetic: it changes token count and apparent turn structure, and
therefore the answer. Day boundaries are UTC, matching CliqService._utc_day.

Output is the full eligible population. Stratum, sampling weight and split are added
later by the sampling step -- see sampling_guide.md.

Usage:
    python src\\build_windows.py
    python src\\build_windows.py --input ..\\OrgPulse\\OrgPulse-API\\fixtures\\sandbox_demo.json
    python src\\build_windows.py --min-messages 1 --min-chars 0     # tiny fixtures
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_day(ts: float) -> str:
    """Identical to CliqService._utc_day."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def analyse_language(text: str) -> Tuple[str, str]:
    """Return (script, language) for stratification. Label-free and cheap.

    Uses laya's own dependency-free detector so the POC strata match the routing
    decision the model will actually make. Falls back to unknown if laya is absent,
    because window building should not require the model to be installed.
    """
    try:
        from laya.lang import analyse
    except Exception:
        return "unknown", "unknown"
    try:
        info = analyse(text)
        return str(info.get("script", "unknown")), str(info.get("language", "unknown"))
    except Exception:
        return "unknown", "unknown"


def load_messages(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load raw_messages.json or OrgPulse's sandbox_demo.json -- same shape."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    messages = payload.get("messages") or []
    channel_names = {
        str(c.get("channel_id")): str(c.get("name") or c.get("channel_id"))
        for c in payload.get("channels") or []
    }
    return messages, channel_names


def build(
    messages: List[Dict[str, Any]],
    channel_names: Dict[str, str],
    scopes: Tuple[str, ...],
    min_messages: int,
    min_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    channel_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    user_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    skipped_no_text = 0

    for msg in messages:
        text = (msg.get("text") or "").strip()
        if not text:
            skipped_no_text += 1
            continue
        day = utc_day(msg.get("timestamp") or 0)
        channel_id = str(msg.get("channel_id") or "")
        user_id = str(msg.get("user_id") or "")
        if channel_id:
            channel_groups[(channel_id, day)].append(msg)
        if user_id:
            user_groups[(user_id, day)].append(msg)

    windows: List[Dict[str, Any]] = []
    dropped_small = 0

    def emit(scope: str, key: Tuple[str, str], group: List[Dict[str, Any]]) -> None:
        nonlocal dropped_small
        ordered = sorted(group, key=lambda m: m.get("timestamp") or 0)
        if scope == "channel":
            text = "\n".join(
                f"[{m.get('user_name', 'User')}]: {m['text'].strip()}" for m in ordered
            )
        else:
            text = "\n".join(m["text"].strip() for m in ordered)

        if len(ordered) < min_messages or len(text) < min_chars:
            dropped_small += 1
            return

        ident, day = key
        channel_id = ident if scope == "channel" else str(ordered[0].get("channel_id") or "")
        script, language = analyse_language(text)
        origins = {str(m.get("origin") or "cliq") for m in ordered}

        windows.append({
            "window_id": f"{ident}|{day}|{scope}",
            "scope": scope,
            "channel_id": channel_id,
            "channel_name": channel_names.get(channel_id, channel_id),
            "user_id": ident if scope == "user" else None,
            "user_name": ordered[0].get("user_name") if scope == "user" else None,
            "date": day,
            "message_ids": [str(m.get("message_id")) for m in ordered],
            "text": text,
            "n_messages": len(ordered),
            "n_chars": len(text),
            "origin": "mixed" if len(origins) > 1 else origins.pop(),
            "lang": language,
            "script": script,
        })

    if "channel" in scopes:
        for key, group in channel_groups.items():
            emit("channel", key, group)
    if "user" in scopes:
        for key, group in user_groups.items():
            emit("user", key, group)

    windows.sort(key=lambda w: (w["date"], w["scope"], w["window_id"]))
    stats = {
        "messages_in": len(messages),
        "messages_skipped_no_text": skipped_no_text,
        "windows_dropped_too_small": dropped_small,
        "windows_out": len(windows),
    }
    return windows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "training" / "data" / "raw_messages.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "training" / "data" / "windows.json")
    parser.add_argument("--scopes", default="channel,user",
                        help="comma list: channel,user")
    parser.add_argument("--min-messages", type=int, default=3,
                        help="drop windows with fewer messages (0 to keep all)")
    parser.add_argument("--min-chars", type=int, default=80,
                        help="drop windows with less text (0 to keep all)")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    scopes = tuple(s.strip() for s in args.scopes.split(",") if s.strip())
    messages, channel_names = load_messages(args.input)
    windows, stats = build(messages, channel_names, scopes, args.min_messages, args.min_chars)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "meta": {
                    "input": str(args.input),
                    "scopes": list(scopes),
                    "min_messages": args.min_messages,
                    "min_chars": args.min_chars,
                    **stats,
                },
                "windows": windows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    by_scope: Dict[str, int] = defaultdict(int)
    for w in windows:
        by_scope[w["scope"]] += 1
    print(f"wrote {len(windows)} windows -> {args.output}")
    print(f"  by scope: {dict(by_scope)}")
    print(f"  {stats}")
    if not windows:
        print("  no windows survived the filters; try --min-messages 1 --min-chars 0")


if __name__ == "__main__":
    main()
