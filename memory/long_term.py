"""Long-term memory management for AIOS Brain.

Handles persistent memory storage, retrieval, and management of
long-term knowledge and experiences.
"""

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Manages persistent long-term memory storage.
    
    Stores facts, knowledge, experiences, and learned information
    that persists across sessions and interactions.
    """
    
    def __init__(self, storage_path: str = "./memory/long_term_storage.json"):
        """Initialize long-term memory.
        
        Args:
            storage_path: Path to persistent storage file
        """
        self.storage_path = storage_path
        self.memory: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.access_count: Dict[str, int] = defaultdict(int)
        self.load_memory()
        logger.info(f"Long-term memory initialized at {storage_path}")
    
    def store_fact(self, category: str, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """Store a fact in long-term memory.
        
        Args:
            category: Category for organizing facts
            key: Unique key for the fact
            value: The fact value to store
            metadata: Optional metadata about the fact
        """
        try:
            self.memory[category][key] = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'access_count': 0
            }
            logger.info(f"Stored fact: {category}/{key}")
        except Exception as e:
            logger.error(f"Error storing fact: {e}")
    
    def retrieve_fact(self, category: str, key: str) -> Optional[Any]:
        """Retrieve a fact from long-term memory.
        
        Args:
            category: Category of the fact
            key: Key of the fact
            
        Returns:
            The fact value or None if not found
        """
        try:
            if category in self.memory and key in self.memory[category]:
                entry = self.memory[category][key]
                entry['access_count'] += 1
                logger.info(f"Retrieved fact: {category}/{key}")
                return entry['value']
            logger.warning(f"Fact not found: {category}/{key}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving fact: {e}")
            return None
    
    def get_category(self, category: str) -> Dict[str, Any]:
        """Get all facts in a category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary of all facts in the category
        """
        try:
            return dict(self.memory.get(category, {}))
        except Exception as e:
            logger.error(f"Error getting category: {e}")
            return {}
    
    def delete_fact(self, category: str, key: str) -> bool:
        """Delete a fact from long-term memory.
        
        Args:
            category: Category of the fact
            key: Key of the fact
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if category in self.memory and key in self.memory[category]:
                del self.memory[category][key]
                logger.info(f"Deleted fact: {category}/{key}")
                return True
            logger.warning(f"Fact not found for deletion: {category}/{key}")
            return False
        except Exception as e:
            logger.error(f"Error deleting fact: {e}")
            return False
    
    def save_memory(self) -> None:
        """Save memory to persistent storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(dict(self.memory), f, indent=2)
            logger.info("Long-term memory saved")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def load_memory(self) -> None:
        """Load memory from persistent storage."""
        try:
            with open(self.storage_path, 'r') as f:
                self.memory = defaultdict(dict, json.load(f))
            logger.info("Long-term memory loaded")
        except FileNotFoundError:
            logger.info("No existing long-term memory found")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        total_facts = sum(len(cat) for cat in self.memory.values())
        categories = len(self.memory)
        return {
            'total_facts': total_facts,
            'categories': categories,
            'storage_path': self.storage_path
        }


if __name__ == "__main__":
    # Demonstration of long-term memory functionality
    print("* " * 30)
    print("LONG-TERM MEMORY DEMONSTRATION")
    print("* " * 30)
    
    memory = LongTermMemory()
    
    # Store some facts
    memory.store_fact('knowledge', 'pi_value', 3.14159, {'source': 'math'})
    memory.store_fact('knowledge', 'earth_circumference', 40075, {'unit': 'km'})
    memory.store_fact('experiences', 'first_interaction', 'session_001', {'date': '2025-11-20'})
    
    # Retrieve facts
    pi = memory.retrieve_fact('knowledge', 'pi_value')
    print(f"\nRetrieved fact - Pi value: {pi}")
    
    # Get category
    knowledge = memory.get_category('knowledge')
    print(f"\nAll knowledge facts: {len(knowledge)} items")
    
    # Get statistics
    stats = memory.get_stats()
    print(f"\nMemory Statistics:")
    print(f"  Total facts: {stats['total_facts']}")
    print(f"  Categories: {stats['categories']}")
    
    # Save memory
    memory.save_memory()
    print(f"\nMemory saved to {memory.storage_path}")
    
    print(f"\n{'* ' * 30}\n")
