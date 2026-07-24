"""RAG (Retrieval-Augmented Generation) pipeline for Mangaba AI v3.0"""

from mangaba.rag.document import Document
from mangaba.rag.loaders import (
    EXTENSION_LOADERS,
    CSVLoader,
    DirectoryLoader,
    DOCXLoader,
    ExcelLoader,
    JSONLoader,
    PDFLoader,
    TextLoader,
    WebPageLoader,
    load_file,
)
from mangaba.rag.splitters import RecursiveTextSplitter
from mangaba.rag.retriever import Retriever
from mangaba.rag.chain import RAGChain

__all__ = [
    "Document",
    "TextLoader",
    "CSVLoader",
    "WebPageLoader",
    "PDFLoader",
    "DOCXLoader",
    "ExcelLoader",
    "JSONLoader",
    "DirectoryLoader",
    "EXTENSION_LOADERS",
    "load_file",
    "RecursiveTextSplitter",
    "Retriever",
    "RAGChain",
]
