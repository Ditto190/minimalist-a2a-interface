"""Mangaba AI

Um pacote Python para criação de agentes de IA inteligentes e versáteis.
"""

from mangaba import __version__  # single source of truth — see mangaba/__init__.py

__author__ = "Mangaba AI Team"
__email__ = "contato@mangaba.ai"
__description__ = "Agente de IA inteligente e versátil"

from mangaba_agent import MangabaAgent
from config import config, Config

__all__ = [
    "MangabaAgent",
    "config",
    "Config",
]
