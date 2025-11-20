"""Episodic memory management for AIOS Brain.

Handles event-based memory storage and retrieval of specific
interactions, conversations, and temporal experiences.
"""

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents a single episodic memory."""
    episode_id: str
    timestamp: str
    event_type: str
    content: Dict[str, Any]
    context: Dict[str, Any]
    emotional_tag: Optional[str] = None
    significance: float = 0.5


class EpisodicMemory:
    """Manages episodic memory storage and retrieval.
    
    Stores specific episodes, events, interactions, and their
    associated contexts, enabling recall and analysis of 
    specific temporal experiences.
    """
    
    def __init__(self, storage_path: str = "./memory/episodic_storage.json"):
        """Initialize episodic memory.
        
        Args:
            storage_path: Path to persistent storage file
        """
        self.storage_path = storage_path
        self.episodes: List[Episode] = []
        self.episode_index: Dict[str, int] = {}
        self.load_episodes()
        logger.info(f"Episodic memory initialized at {storage_path}")
    
    def record_episode(self, event_type: str, content: Dict[str, Any], 
                       context: Optional[Dict] = None, 
                       emotional_tag: Optional[str] = None,
                       significance: float = 0.5) -> str:
        """Record a new episodic memory.
        
        Args:
            event_type: Type of event (interaction, conversation, etc.)
            content: Event content
            context: Associated context information
            emotional_tag: Optional emotional tag
            significance: Significance score (0-1)
            
        Returns:
            Episode ID of the recorded episode
        """
        try:
            episode_id = f"ep_{len(self.episodes)}_{datetime.now().timestamp()}"
            episode = Episode(
                episode_id=episode_id,
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                content=content,
                context=context or {},
                emotional_tag=emotional_tag,
                significance=significance
            )
            self.episodes.append(episode)
            self.episode_index[episode_id] = len(self.episodes) - 1
            logger.info(f"Recorded episode: {episode_id}")
            return episode_id
        except Exception as e:
            logger.error(f"Error recording episode: {e}")
            return ""
    
    def retrieve_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by ID.
        
        Args:
            episode_id: ID of the episode to retrieve
            
        Returns:
            Episode object or None if not found
        """
        try:
            if episode_id in self.episode_index:
                idx = self.episode_index[episode_id]
                logger.info(f"Retrieved episode: {episode_id}")
                return self.episodes[idx]
            logger.warning(f"Episode not found: {episode_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving episode: {e}")
            return None
    
    def get_episodes_by_type(self, event_type: str) -> List[Episode]:
        """Get all episodes of a specific type.
        
        Args:
            event_type: Type of events to retrieve
            
        Returns:
            List of matching episodes
        """
        try:
            return [ep for ep in self.episodes if ep.event_type == event_type]
        except Exception as e:
            logger.error(f"Error getting episodes by type: {e}")
            return []
    
    def get_episodes_by_timerange(self, start_time: str, end_time: str) -> List[Episode]:
        """Get episodes within a time range.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            
        Returns:
            List of episodes in the time range
        """
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            return [ep for ep in self.episodes 
                    if start <= datetime.fromisoformat(ep.timestamp) <= end]
        except Exception as e:
            logger.error(f"Error getting episodes by time: {e}")
            return []
    
    def get_significant_episodes(self, min_significance: float = 0.7) -> List[Episode]:
        """Get episodes above significance threshold.
        
        Args:
            min_significance: Minimum significance score
            
        Returns:
            List of significant episodes
        """
        try:
            return [ep for ep in self.episodes if ep.significance >= min_significance]
        except Exception as e:
            logger.error(f"Error getting significant episodes: {e}")
            return []
    
    def save_episodes(self) -> None:
        """Save episodes to persistent storage."""
        try:
            episodes_dict = [asdict(ep) for ep in self.episodes]
            with open(self.storage_path, 'w') as f:
                json.dump(episodes_dict, f, indent=2)
            logger.info("Episodic memory saved")
        except Exception as e:
            logger.error(f"Error saving episodes: {e}")
    
    def load_episodes(self) -> None:
        """Load episodes from persistent storage."""
        try:
            with open(self.storage_path, 'r') as f:
                episodes_dict = json.load(f)
                self.episodes = [Episode(**ep) for ep in episodes_dict]
                self.episode_index = {ep.episode_id: i for i, ep in enumerate(self.episodes)}
            logger.info(f"Loaded {len(self.episodes)} episodes")
        except FileNotFoundError:
            logger.info("No existing episodic memory found")
        except Exception as e:
            logger.error(f"Error loading episodes: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        event_types = {}
        for ep in self.episodes:
            event_types[ep.event_type] = event_types.get(ep.event_type, 0) + 1
        
        return {
            'total_episodes': len(self.episodes),
            'event_types': event_types,
            'storage_path': self.storage_path
        }


if __name__ == "__main__":
    # Demonstration of episodic memory functionality
    print("* " * 30)
    print("EPISODIC MEMORY DEMONSTRATION")
    print("* " * 30)
    
    memory = EpisodicMemory()
    
    # Record episodes
    ep1 = memory.record_episode('conversation', 
                                 {'message': 'Hello', 'response': 'Hi there'},
                                 {'user': 'user_001'},
                                 'positive', 0.8)
    
    ep2 = memory.record_episode('interaction',
                                 {'action': 'query', 'topic': 'weather'},
                                 {'location': 'NYC'},
                                 'neutral', 0.6)
    
    # Retrieve episodes
    episode = memory.retrieve_episode(ep1)
    print(f"\nRetrieved episode: {episode.event_type}")
    
    # Get episodes by type
    convs = memory.get_episodes_by_type('conversation')
    print(f"\nConversation episodes: {len(convs)}")
    
    # Get significant episodes
    significant = memory.get_significant_episodes(0.7)
    print(f"Significant episodes (>0.7): {len(significant)}")
    
    # Get statistics
    stats = memory.get_stats()
    print(f"\nMemory Statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Event types: {stats['event_types']}")
    
    # Save episodes
    memory.save_episodes()
    print(f"\nEpisodes saved to {memory.storage_path}")
    
    print(f"\n{'* ' * 30}\n")
