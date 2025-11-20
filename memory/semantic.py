"""Semantic memory management for AIOS Brain.

Handles storage and retrieval of semantic knowledge, meanings,
and general facts independent of specific experiences.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class SemanticMemory:
    """Manages semantic memory storage and retrieval.
    
    Stores general knowledge, facts, concepts, and their relationships.
    Independent of personal experiences (unlike episodic memory).
    """
    
    def __init__(self, storage_path: str = "./memory/semantic_storage.json"):
        """Initialize semantic memory.
        
        Args:
            storage_path: Path to persistent storage file
        """
        self.storage_path = storage_path
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, Set[str]] = defaultdict(set)
        self.load_knowledge()
        logger.info(f"Semantic memory initialized at {storage_path}")
    
    def add_concept(self, concept_id: str, definition: str, 
                    category: str, properties: Optional[Dict] = None) -> None:
        """Add a concept to semantic memory.
        
        Args:
            concept_id: Unique identifier for the concept
            definition: Definition of the concept
            category: Category of the concept
            properties: Additional properties/attributes
        """
        try:
            self.concepts[concept_id] = {
                'definition': definition,
                'category': category,
                'properties': properties or {},
                'created_at': __import__('datetime').datetime.now().isoformat()
            }
            logger.info(f"Added concept: {concept_id}")
        except Exception as e:
            logger.error(f"Error adding concept: {e}")
    
    def get_concept(self, concept_id: str) -> Optional[Dict]:
        """Retrieve a concept from semantic memory.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Concept information or None if not found
        """
        try:
            return self.concepts.get(concept_id)
        except Exception as e:
            logger.error(f"Error getting concept: {e}")
            return None
    
    def add_relationship(self, concept1: str, concept2: str, 
                        relationship_type: str) -> None:
        """Add a relationship between concepts.
        
        Args:
            concept1: First concept ID
            concept2: Second concept ID
            relationship_type: Type of relationship
        """
        try:
            key = f"{concept1}--{relationship_type}--{concept2}"
            self.relationships[concept1].add(key)
            logger.info(f"Added relationship: {key}")
        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
    
    def get_related_concepts(self, concept_id: str) -> List[str]:
        """Get concepts related to a specific concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            List of related concept IDs
        """
        try:
            return list(self.relationships.get(concept_id, set()))
        except Exception as e:
            logger.error(f"Error getting related concepts: {e}")
            return []
    
    def get_concepts_by_category(self, category: str) -> List[str]:
        """Get all concepts in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of concept IDs in the category
        """
        try:
            return [cid for cid, info in self.concepts.items() 
                    if info.get('category') == category]
        except Exception as e:
            logger.error(f"Error getting concepts by category: {e}")
            return []
    
    def search_concepts(self, query: str) -> List[str]:
        """Search for concepts by keyword.
        
        Args:
            query: Search query
            
        Returns:
            List of matching concept IDs
        """
        try:
            query_lower = query.lower()
            results = []
            for cid, info in self.concepts.items():
                if query_lower in cid.lower() or \
                   query_lower in info.get('definition', '').lower():
                    results.append(cid)
            return results
        except Exception as e:
            logger.error(f"Error searching concepts: {e}")
            return []
    
    def save_knowledge(self) -> None:
        """Save semantic knowledge to persistent storage."""
        try:
            data = {
                'concepts': self.concepts,
                'relationships': {k: list(v) for k, v in self.relationships.items()}
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Semantic memory saved")
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
    
    def load_knowledge(self) -> None:
        """Load semantic knowledge from persistent storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.concepts = data.get('concepts', {})
                relationships = data.get('relationships', {})
                self.relationships = defaultdict(set)
                for k, v in relationships.items():
                    self.relationships[k] = set(v)
            logger.info(f"Loaded {len(self.concepts)} concepts")
        except FileNotFoundError:
            logger.info("No existing semantic memory found")
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about semantic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        categories = defaultdict(int)
        for info in self.concepts.values():
            categories[info.get('category', 'unknown')] += 1
        
        return {
            'total_concepts': len(self.concepts),
            'total_relationships': sum(len(v) for v in self.relationships.values()),
            'categories': dict(categories),
            'storage_path': self.storage_path
        }


if __name__ == "__main__":
    # Demonstration of semantic memory functionality
    print("* " * 30)
    print("SEMANTIC MEMORY DEMONSTRATION")
    print("* " * 30)
    
    memory = SemanticMemory()
    
    # Add concepts
    memory.add_concept('animal', 'Living organism that can move', 'biology')
    memory.add_concept('dog', 'Domesticated canine mammal', 'biology', 
                       {'legs': 4, 'sound': 'bark'})
    memory.add_concept('cat', 'Domesticated feline mammal', 'biology',
                       {'legs': 4, 'sound': 'meow'})
    
    # Add relationships
    memory.add_relationship('dog', 'animal', 'is_a')
    memory.add_relationship('cat', 'animal', 'is_a')
    
    # Retrieve concepts
    concept = memory.get_concept('dog')
    print(f"\nRetrieved concept: {concept}")
    
    # Get concepts by category
    bio_concepts = memory.get_concepts_by_category('biology')
    print(f"\nBiology concepts: {len(bio_concepts)}")
    
    # Search concepts
    search_results = memory.search_concepts('dog')
    print(f"\nSearch results for 'dog': {search_results}")
    
    # Get statistics
    stats = memory.get_stats()
    print(f"\nMemory Statistics:")
    print(f"  Total concepts: {stats['total_concepts']}")
    print(f"  Categories: {stats['categories']}")
    
    # Save knowledge
    memory.save_knowledge()
    print(f"\nKnowledge saved to {memory.storage_path}")
    
    print(f"\n{'* ' * 30}\n")
