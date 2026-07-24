"""
Base abstraction for Mangaba AI knowledge sources.

A *knowledge source* is anything that can be turned into a list of
:class:`~mangaba.rag.document.Document` objects for grounding an agent.
This is deliberately distinct from conversational memory: memory records
what happened during a run, knowledge is curated reference material that is
ingested once and retrieved semantically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from mangaba.rag.document import Document

log = logging.getLogger(__name__)


class BaseKnowledgeSource(BaseModel, ABC):
    """Abstract source of grounding documents.

    Subclasses implement :meth:`load` and normally delegate the actual
    parsing to a loader from :mod:`mangaba.rag.loaders`.

    Attributes:
        metadata: Extra key/value pairs merged into every produced Document.

    Example::

        class InlineSource(BaseKnowledgeSource):
            text: str

            def load(self) -> List[Document]:
                return self._finalize([Document(content=self.text)])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @abstractmethod
    def load(self) -> List[Document]:
        """Return the documents contributed by this source."""
        ...

    @property
    def source_name(self) -> str:
        """Human-readable identifier used for attribution in prompts."""
        return type(self).__name__

    def _finalize(self, documents: List[Document]) -> List[Document]:
        """Stamp shared metadata onto documents produced by a loader.

        Adds ``knowledge_source`` (the concrete class name) and merges
        ``self.metadata``, without clobbering keys the loader already set
        (notably ``source``).
        """
        for doc in documents:
            merged: Dict[str, Any] = dict(self.metadata)
            merged.update(doc.metadata)
            merged.setdefault("source", self.source_name)
            merged["knowledge_source"] = type(self).__name__
            doc.metadata = merged
        log.debug("%s produced %d document(s)", type(self).__name__, len(documents))
        return documents
