"""
Concrete knowledge sources for Mangaba AI.

Each class here is a thin, declarative wrapper around a loader from
:mod:`mangaba.rag.loaders`. They exist so a :class:`~mangaba.knowledge.knowledge.Knowledge`
instance can be configured with plain data (paths, URLs, options) that is
validated up front by Pydantic and only touched at ingest time.

Optional third-party dependencies are never imported here — the underlying
loader raises a clear ``ImportError`` naming the extra to install.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from mangaba.knowledge.base import BaseKnowledgeSource
from mangaba.rag.document import Document
from mangaba.rag.loaders import (
    CSVLoader,
    DirectoryLoader,
    DOCXLoader,
    ExcelLoader,
    JSONLoader,
    PDFLoader,
    TextLoader,
    WebPageLoader,
)

log = logging.getLogger(__name__)


class StringKnowledgeSource(BaseKnowledgeSource):
    """Raw in-memory text.

    Example::

        source = StringKnowledgeSource(
            content="Mangaba AI is a multi-agent framework.",
            name="company-facts",
        )
    """

    content: str
    name: str = "string"

    @property
    def source_name(self) -> str:
        return self.name

    def load(self) -> List[Document]:
        return self._finalize(
            [Document(content=self.content, metadata={"source": self.name})]
        )


class TextFileKnowledgeSource(BaseKnowledgeSource):
    """A plain-text/Markdown file on disk.

    Example::

        source = TextFileKnowledgeSource(file_path="handbook.md")
    """

    file_path: str
    encoding: str = "utf-8"

    @property
    def source_name(self) -> str:
        return self.file_path

    def load(self) -> List[Document]:
        loader = TextLoader(self.file_path, encoding=self.encoding)
        return self._finalize(loader.load())


class PDFKnowledgeSource(BaseKnowledgeSource):
    """A PDF file (requires ``pypdf``).

    Example::

        source = PDFKnowledgeSource(file_path="policy.pdf", split_pages=True)
    """

    file_path: str
    split_pages: bool = True
    password: Optional[str] = None

    @property
    def source_name(self) -> str:
        return self.file_path

    def load(self) -> List[Document]:
        loader = PDFLoader(
            self.file_path,
            split_pages=self.split_pages,
            password=self.password,
        )
        return self._finalize(loader.load())


class DOCXKnowledgeSource(BaseKnowledgeSource):
    """A Word ``.docx`` file (requires ``python-docx``).

    Example::

        source = DOCXKnowledgeSource(file_path="contract.docx")
    """

    file_path: str
    include_tables: bool = True

    @property
    def source_name(self) -> str:
        return self.file_path

    def load(self) -> List[Document]:
        loader = DOCXLoader(self.file_path, include_tables=self.include_tables)
        return self._finalize(loader.load())


class CSVKnowledgeSource(BaseKnowledgeSource):
    """A CSV file — one document per row.

    Example::

        source = CSVKnowledgeSource(file_path="faq.csv", content_columns=["q", "a"])
    """

    file_path: str
    content_columns: Optional[List[str]] = None
    encoding: str = "utf-8"

    @property
    def source_name(self) -> str:
        return self.file_path

    def load(self) -> List[Document]:
        loader = CSVLoader(
            self.file_path,
            content_columns=self.content_columns,
            encoding=self.encoding,
        )
        return self._finalize(loader.load())


class ExcelKnowledgeSource(BaseKnowledgeSource):
    """An Excel workbook (requires ``openpyxl``); handles multiple sheets.

    Example::

        source = ExcelKnowledgeSource(file_path="sales.xlsx", mode="row")
    """

    file_path: str
    sheet_names: Optional[List[str]] = None
    mode: str = "sheet"

    @property
    def source_name(self) -> str:
        return self.file_path

    def load(self) -> List[Document]:
        loader = ExcelLoader(
            self.file_path,
            sheet_names=self.sheet_names,
            mode=self.mode,
        )
        return self._finalize(loader.load())


class JSONKnowledgeSource(BaseKnowledgeSource):
    """A JSON file, optionally narrowed with a dot-path selector.

    Example::

        source = JSONKnowledgeSource(
            file_path="catalog.json",
            jq_like="products[].description",
        )
    """

    file_path: str
    jq_like: Optional[str] = None
    encoding: str = "utf-8"

    @property
    def source_name(self) -> str:
        return self.file_path

    def load(self) -> List[Document]:
        loader = JSONLoader(
            self.file_path,
            jq_like=self.jq_like,
            encoding=self.encoding,
        )
        return self._finalize(loader.load())


class URLKnowledgeSource(BaseKnowledgeSource):
    """One or more web pages fetched over HTTP.

    Example::

        source = URLKnowledgeSource(urls=["https://example.com/docs"])
    """

    urls: List[str] = Field(default_factory=list)

    @property
    def source_name(self) -> str:
        return ", ".join(self.urls) if self.urls else "url"

    def load(self) -> List[Document]:
        docs: List[Document] = []
        for url in self.urls:
            docs.extend(WebPageLoader(url).load())
        return self._finalize(docs)


class DirectoryKnowledgeSource(BaseKnowledgeSource):
    """Every supported file inside a directory tree.

    Example::

        source = DirectoryKnowledgeSource(path="./docs", glob="*.md", recursive=True)
    """

    path: str
    glob: str = "*"
    recursive: bool = True
    silent_errors: bool = True
    loader_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @property
    def source_name(self) -> str:
        return self.path

    def load(self) -> List[Document]:
        loader = DirectoryLoader(
            self.path,
            glob=self.glob,
            recursive=self.recursive,
            silent_errors=self.silent_errors,
            loader_kwargs=self.loader_kwargs or None,
        )
        return self._finalize(loader.load())
