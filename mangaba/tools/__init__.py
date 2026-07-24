"""Tools for Mangaba AI agents."""

from mangaba.tools.base import BaseTool
from mangaba.tools.decorator import tool
from mangaba.tools.toolkit import BaseToolkit, FileToolkit, WebToolkit
from mangaba.tools.math_tools import CalculatorTool
from mangaba.tools.text_tools import TextSplitterTool, WordCounterTool

# Web, documents, code and data
from mangaba.tools.web_tools import HTTPRequestTool, ScrapeWebsiteTool
from mangaba.tools.document_tools import DocumentSearchTool, FileSearchTool
from mangaba.tools.code_tools import CodeInterpreterTool
from mangaba.tools.data_tools import SQLQueryTool, UnsafeQueryError
from mangaba.tools.registry import REGISTRY, ToolRegistry

# Tools borrowed from external MCP servers
from mangaba.tools.mcp_client import MCPClient, MCPTool

__all__ = [
    "BaseTool",
    "tool",
    "BaseToolkit",
    "FileToolkit",
    "WebToolkit",
    "CalculatorTool",
    "TextSplitterTool",
    "WordCounterTool",
    # Web
    "ScrapeWebsiteTool",
    "HTTPRequestTool",
    # Documents
    "DocumentSearchTool",
    "FileSearchTool",
    # Code — off by default, see the class docstring before enabling
    "CodeInterpreterTool",
    # Data — read-only SQL
    "SQLQueryTool",
    "UnsafeQueryError",
    # Registry
    "ToolRegistry",
    "REGISTRY",
    # MCP
    "MCPClient",
    "MCPTool",
]
