"""Memory module for AIOS Brain.

Provides integrated access to long-term, episodic, and semantic memory systems.
"""

from .long_term import LongTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory

__version__ = '1.0.0'
__author__ = 'Str8biddness'
__all__ = ['LongTermMemory', 'EpisodicMemory', 'SemanticMemory', 'MemoryManager']


class MemoryManager:
    """Unified memory management system.
    
    Integrates long-term, episodic, and semantic memory systems
    for comprehensive knowledge and experience management.
    """
    
    def __init__(self, base_path: str = "./memory"):
        """Initialize memory manager.
        
        Args:
            base_path: Base path for memory storage
        """
        self.base_path = base_path
        self.long_term = LongTermMemory(f"{base_path}/long_term_storage.json")
        self.episodic = EpisodicMemory(f"{base_path}/episodic_storage.json")
        self.semantic = SemanticMemory(f"{base_path}/semantic_storage.json")
    
    def get_all_stats(self) -> dict:
        """Get statistics from all memory systems.
        
        Returns:
            Dictionary containing stats from all memory types
        """
        return {
            'long_term': self.long_term.get_stats(),
            'episodic': self.episodic.get_stats(),
            'semantic': self.semantic.get_stats()
        }
    
    def save_all(self) -> None:
        """Save all memory systems to persistent storage."""
        self.long_term.save_memory()
        self.episodic.save_episodes()
        self.semantic.save_knowledge()


if __name__ == "__main__":
    print("* " * 30)
    print("MEMORY MODULE INITIALIZATION")
    print("* " * 30)
    
    # Initialize unified memory manager
    memory_manager = MemoryManager()
    
    print(f"\nMemory Manager initialized successfully")
    print(f"Version: {__version__}")
    print(f"\nAvailable memory systems:")
    print(f"  - Long-term memory")
    print(f"  - Episodic memory")
    print(f"  - Semantic memory")
    
    # Get aggregate statistics
    stats = memory_manager.get_all_stats()
    print(f"\nMemory Statistics:")
    for mem_type, mem_stats in stats.items():
        print(f"  {mem_type}: {mem_stats}")
    
    print(f"\n{'* ' * 30}\n")
