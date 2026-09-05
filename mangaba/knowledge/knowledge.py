"""
Knowledge: the RAG-style grounding layer of Mangaba AI.

``Knowledge`` ingests :class:`~mangaba.knowledge.base.BaseKnowledgeSource`
objects, chunks them with
:class:`~mangaba.rag.splitters.RecursiveTextSplitter`, embeds the chunks and
stores them in a vector store so agents can retrieve curated reference
material at prompt time.

This is *not* conversational memory: nothing is written here as a side effect
of a run — you decide what goes in.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence

from mangaba.embeddings.base import BaseEmbedding
from mangaba.knowledge.base import BaseKnowledgeSource
from mangaba.rag.document import Document
from mangaba.rag.retriever import Retriever
from mangaba.rag.splitters import RecursiveTextSplitter
from mangaba.vectorstores.base import BaseVectorStore
from mangaba.vectorstores.factory import create_vectorstore

log = logging.getLogger(__name__)

#: Backend used when no vector store is supplied — pure Python, no server.
DEFAULT_STORAGE_BACKEND = "inmemory"

DEFAULT_COLLECTION_NAME = "mangaba_knowledge"
DEFAULT_RESULTS_LIMIT = 3
DEFAULT_SCORE_THRESHOLD = 0.35


class KnowledgeConfig:
    """Declarative config for :class:`Knowledge` (CrewAI-parity).

    Example::

        cfg = KnowledgeConfig(collection_name="handbook", results_limit=5, score_threshold=0.4)
        knowledge = Knowledge(embedding=emb, sources=[...], config=cfg)
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        results_limit: int = DEFAULT_RESULTS_LIMIT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        storage_backend: str = DEFAULT_STORAGE_BACKEND,
    ) -> None:
        self.collection_name = collection_name
        self.results_limit = results_limit
        self.score_threshold = score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_backend = storage_backend


class Knowledge:
    """A queryable collection of grounding documents.

    Args:
        sources: Knowledge sources ingested immediately on construction.
        embedding: Any :class:`~mangaba.embeddings.base.BaseEmbedding`.
        vectorstore: Pre-built store. When omitted one is created through
            :func:`~mangaba.vectorstores.factory.create_vectorstore` using
            ``storage_backend`` (default ``"inmemory"``, which needs no
            external server).
        collection_name: Collection/namespace name, forwarded to backends
            that accept it (e.g. Chroma).
        results_limit: Default number of chunks returned by :meth:`query`.
        score_threshold: Minimum similarity score a chunk must reach to be
            returned. Scores come from the vector store (cosine similarity
            for the in-memory backend, so roughly ``-1.0..1.0``).
        chunk_size / chunk_overlap: Passed to the default splitter.
        splitter: Custom splitter overriding ``chunk_size``/``chunk_overlap``.
        storage_backend: Name understood by ``create_vectorstore``.

    Example::

        knowledge = Knowledge(
            sources=[
                StringKnowledgeSource(content="Mangaba AI ships multi-agent crews."),
                PDFKnowledgeSource(file_path="handbook.pdf"),
            ],
            embedding=OpenAIEmbedding(api_key="..."),
        )

        docs = knowledge.query("What does Mangaba ship?")
        prompt_block = knowledge.to_context_string("What does Mangaba ship?")
    """

    def __init__(
        self,
        embedding: BaseEmbedding,
        sources: Optional[Sequence[BaseKnowledgeSource]] = None,
        vectorstore: Optional[BaseVectorStore] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        results_limit: int = DEFAULT_RESULTS_LIMIT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter: Optional[RecursiveTextSplitter] = None,
        storage_backend: str = DEFAULT_STORAGE_BACKEND,
        config: Optional[KnowledgeConfig] = None,
    ) -> None:
        if config is not None:
            collection_name = config.collection_name
            results_limit = config.results_limit
            score_threshold = config.score_threshold
            chunk_size = config.chunk_size
            chunk_overlap = config.chunk_overlap
            storage_backend = config.storage_backend
        self.embedding = embedding
        self.collection_name = collection_name
        self.results_limit = results_limit
        self.score_threshold = score_threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.storage_backend = storage_backend

        self.splitter = splitter or RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.store: BaseVectorStore = vectorstore or _build_default_store(
            storage_backend, collection_name
        )
        self.retriever = Retriever(embedding=self.embedding, store=self.store)

        self.sources: List[BaseKnowledgeSource] = []
        if sources:
            self.add_sources(sources)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_sources(self, sources: Sequence[BaseKnowledgeSource]) -> List[str]:
        """Load, chunk, embed and store the given sources.

        Returns the vector-store IDs of the stored chunks.

        Example::

            knowledge.add_sources([TextFileKnowledgeSource(file_path="faq.txt")])
        """
        if isinstance(sources, BaseKnowledgeSource):  # tolerate a single source
            sources = [sources]

        documents: List[Document] = []
        for source in sources:
            if not isinstance(source, BaseKnowledgeSource):
                raise TypeError(
                    f"Expected BaseKnowledgeSource, got {type(source).__name__}"
                )
            loaded = source.load()
            log.debug("Knowledge: %s loaded %d document(s)", source.source_name, len(loaded))
            documents.extend(loaded)
            self.sources.append(source)

        return self.add_documents(documents)

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Chunk, embed and store already-built Documents.

        Example::

            knowledge.add_documents(TextLoader("notes.txt").load())
        """
        documents = [d for d in documents if d.content and d.content.strip()]
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        chunks = [c for c in chunks if c.content.strip()]
        if not chunks:
            return []

        ids = self.retriever.add_documents(chunks)
        log.info(
            "Knowledge[%s]: stored %d chunk(s) from %d document(s)",
            self.collection_name,
            len(chunks),
            len(documents),
        )
        return ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(
        self,
        text: str,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """Return the most relevant chunks for *text*.

        Results below ``score_threshold`` are dropped. Each returned
        Document carries a ``score`` key in its metadata.

        Example::

            for doc in knowledge.query("refund policy", limit=5):
                print(doc.metadata["score"], doc.metadata["source"])
        """
        if not text or not text.strip():
            return []

        top_k = limit if limit is not None else self.results_limit
        threshold = (
            score_threshold if score_threshold is not None else self.score_threshold
        )

        results = self.retriever.search(text, top_k=top_k)
        filtered = [d for d in results if float(d.metadata.get("score", 0.0)) >= threshold]
        log.debug(
            "Knowledge[%s]: query %r -> %d/%d chunk(s) above threshold %.3f",
            self.collection_name,
            text[:60],
            len(filtered),
            len(results),
            threshold,
        )
        return filtered

    def to_context_string(
        self,
        query: str,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        header: str = "Relevant knowledge:",
    ) -> str:
        """Format :meth:`query` results as a block ready for prompt injection.

        Returns an empty string when nothing clears the threshold, so callers
        can concatenate it unconditionally.

        Example::

            context = knowledge.to_context_string("refund policy")
            prompt = f"{context}\\n\\nQuestion: {question}"
        """
        docs = self.query(query, limit=limit, score_threshold=score_threshold)
        if not docs:
            return ""

        blocks: List[str] = [header]
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            score = float(doc.metadata.get("score", 0.0))
            attribution = f"[{i}] source: {source} (relevance {score:.2f})"
            page = doc.metadata.get("page")
            sheet = doc.metadata.get("sheet")
            if page is not None:
                attribution += f" | page {page}"
            if sheet is not None:
                attribution += f" | sheet {sheet}"
            blocks.append(f"{attribution}\n{doc.content.strip()}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop every stored chunk and forget the registered sources.

        Example::

            knowledge.reset()
            knowledge.add_sources([StringKnowledgeSource(content="fresh start")])
        """
        self.store.clear()
        self.sources = []
        log.info("Knowledge[%s]: reset", self.collection_name)

    @property
    def count(self) -> int:
        """Number of chunks currently stored."""
        return self.store.count

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return (
            f"Knowledge(collection={self.collection_name!r}, "
            f"sources={len(self.sources)}, chunks={self.count})"
        )


def _build_default_store(storage_backend: str, collection_name: str) -> BaseVectorStore:
    """Create a vector store, forwarding ``collection_name`` only if supported."""
    kwargs: Dict[str, Any] = {}
    try:
        from mangaba.vectorstores.factory import STORE_REGISTRY

        store_cls = STORE_REGISTRY.get(storage_backend.lower())
        if store_cls is not None:
            params = inspect.signature(store_cls.__init__).parameters
            if "collection_name" in params:
                kwargs["collection_name"] = collection_name
    except (TypeError, ValueError):  # pragma: no cover - defensive
        log.debug("Knowledge: could not inspect store %r constructor", storage_backend)

    return create_vectorstore(storage_backend, **kwargs)
