"""Knowledge subsystem for Mangaba AI v3.0.

A RAG-style grounding layer: curated reference material (files, URLs, raw
strings) ingested once and retrieved semantically at prompt time. Distinct
from :mod:`mangaba.memory`, which records what happened during a run.

Example::

    from mangaba.knowledge import Knowledge, StringKnowledgeSource, PDFKnowledgeSource

    knowledge = Knowledge(
        embedding=my_embedding,
        sources=[
            StringKnowledgeSource(content="Mangaba AI orchestrates agent crews."),
            PDFKnowledgeSource(file_path="handbook.pdf"),
        ],
    )
    context = knowledge.to_context_string("How are crews orchestrated?")
"""

from mangaba.knowledge.base import BaseKnowledgeSource
from mangaba.knowledge.knowledge import Knowledge
from mangaba.knowledge.sources import (
    CSVKnowledgeSource,
    DirectoryKnowledgeSource,
    DOCXKnowledgeSource,
    ExcelKnowledgeSource,
    JSONKnowledgeSource,
    PDFKnowledgeSource,
    StringKnowledgeSource,
    TextFileKnowledgeSource,
    URLKnowledgeSource,
)

__all__ = [
    "BaseKnowledgeSource",
    "Knowledge",
    "StringKnowledgeSource",
    "TextFileKnowledgeSource",
    "PDFKnowledgeSource",
    "DOCXKnowledgeSource",
    "CSVKnowledgeSource",
    "ExcelKnowledgeSource",
    "JSONKnowledgeSource",
    "URLKnowledgeSource",
    "DirectoryKnowledgeSource",
]
