"""Audit log + RBAC lite (Fase 3).

CrewAI AMP ships SSO/RBAC/immutable audit trails. Mangaba gets a
file-based, dependency-free subset suitable for PoCs and small prod:

- :class:`AuditLogger` appends JSONL rows (actor, action, resource, result).
- :class:`Role` + :func:`require_role` guard sensitive operations.
- :class:`AuditCallback` forwards crew/task/agent lifecycle events to the log.

Example::

    from mangaba.observability.audit import AuditLogger, AuditCallback
    from mangaba.observability import configure_observability

    logger = AuditLogger(path=".mangaba/audit.jsonl", actor="api")
    configure_observability(AuditCallback(logger))
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from mangaba.core.events import BaseCallback, Event, EventType

log = logging.getLogger(__name__)


class Role(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"

    def can(self, action: str) -> bool:
        if self == Role.ADMIN:
            return True
        if self == Role.OPERATOR:
            return action in ("run", "chat", "test", "replay", "view")
        return action in ("view",)


def require_role(role: Role, action: str) -> None:
    """Raise PermissionError when *role* may not perform *action*."""
    if not role.can(action):
        raise PermissionError(f"role '{role.value}' cannot perform '{action}'")


class AuditLogger:
    """Append-only JSONL audit log."""

    def __init__(self, path: str = ".mangaba/audit.jsonl", actor: str = "system") -> None:
        self.path = path
        self.actor = actor
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def record(self, action: str, resource: str = "", result: str = "ok", details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row = {
            "ts": datetime.now().isoformat(),
            "actor": self.actor,
            "action": action,
            "resource": resource,
            "result": result,
            "details": details or {},
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    def read(self, limit: int = 100) -> list:
        if not os.path.isfile(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


_AUDIT_EVENTS = {
    EventType.CREW_START: ("crew.run", "start"),
    EventType.CREW_END: ("crew.run", "ok"),
    EventType.CREW_ERROR: ("crew.run", "error"),
    EventType.TASK_START: ("task.run", "start"),
    EventType.TASK_END: ("task.run", "ok"),
    EventType.TASK_ERROR: ("task.run", "error"),
}


class AuditCallback(BaseCallback):
    """Forward lifecycle events to an :class:`AuditLogger`."""

    def __init__(self, logger: AuditLogger) -> None:
        super().__init__()
        self.logger = logger
        self.enabled = True

    def on_event(self, event: Event) -> None:
        mapping = _AUDIT_EVENTS.get(event.event_type)
        if not mapping:
            return
        action, _ = mapping
        try:
            self.logger.record(action, resource=str(event.source_id or ""), result=str((event.data or {}).get("status", mapping[1])), details={"event": event.event_type.value})
        except Exception:
            log.debug("audit record failed", exc_info=True)


__all__ = ["Role", "require_role", "AuditLogger", "AuditCallback"]
