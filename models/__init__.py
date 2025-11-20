__init__.py"""Models module for AIOS Brain.

Provides unified access to LLM and embedding models.
"""

from .llm import LLMModel, LLMInterface
from .embedding import TextEmbedding, EmbeddingModel

__version__ = '1.0.0'
__author__ = 'Str8biddness'
__all__ = ['LLMModel', 'LLMInterface', 'TextEmbedding', 'EmbeddingModel']

if __name__ == "__main__":
    print("Models module initialized")
