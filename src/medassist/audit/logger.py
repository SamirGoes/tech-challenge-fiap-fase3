"""Logging de auditoria em JSONL.

Cada interação gera eventos com o mesmo `trace_id`, o que permite reconstruir
quem perguntou o quê, quais dados do paciente e quais fontes foram usadas,
o que o modelo respondeu e quais regras de segurança foram acionadas.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medassist.config import get_settings

_console = logging.getLogger("medassist")
if not _console.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
    _console.addHandler(_handler)
    _console.setLevel(logging.INFO)


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


class AuditLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().log_dir / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, trace_id: str, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        _console.info("[%s] %s %s", trace_id, event, _short(payload))
        return record

    def read(self, trace_id: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
        return [r for r in rows if trace_id is None or r["trace_id"] == trace_id]


def _short(payload: dict[str, Any], limit: int = 160) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "..."
