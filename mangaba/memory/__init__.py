"""Memory system for Mangaba AI v3.0"""

from mangaba.memory.base import BaseMemory
from mangaba.memory.short_term import ShortTermMemory
from mangaba.memory.long_term import LongTermMemory
from mangaba.memory.entity import EntityMemory
from mangaba.memory.unified import (
    InMemoryBackend,
    Memory,
    MemoryEntry,
    MemoryScope,
    MemoryWeights,
    SQLiteBackend,
    StorageBackend,
)

try:
    from mangaba.memory.external import BaseExternalMemory, Mem0Memory
except Exception:  # pragma: no cover - never break the base import
    BaseExternalMemory = object  # type: ignore
    Mem0Memory = None  # type: ignore

__all__ = [
    "BaseMemory",
    "ShortTermMemory",
    "LongTermMemory",
    "EntityMemory",
    "Memory",
    "MemoryEntry",
    "MemoryScope",
    "MemoryWeights",
    "StorageBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    "BaseExternalMemory",
    "Mem0Memory",
]
