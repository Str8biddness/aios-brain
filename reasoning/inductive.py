inductive.py"""Inductive reasoning - patterns from observations."""
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class InductiveEngine:
    def __init__(self):
        self.observations: List[Dict[str, Any]] = []
        self.patterns: List[Dict[str, Any]] = []
        logger.info("Inductive engine initialized")
    
    def add_observation(self, obs: Dict[str, Any]) -> None:
        self.observations.append(obs)
        logger.info(f"Added observation")
    
    def infer_pattern(self, attr: str, threshold: float = 0.7) -> Dict[str, Any]:
        if not self.observations:
            return {}
        values = defaultdict(int)
        for obs in self.observations:
            if attr in obs:
                values[obs[attr]] += 1
        if values:
            val, cnt = max(values.items(), key=lambda x: x[1])
            conf = cnt / len(self.observations)
            if conf >= threshold:
                p = {'attr': attr, 'val': val, 'conf': conf, 'cnt': cnt}
                self.patterns.append(p)
                logger.info(f"Pattern: {attr}={val} ({conf:.1%})")
                return p
        return {}
    
    def generalize(self) -> str:
        if not self.patterns:
            return "No patterns"
        high = [p for p in self.patterns if p['conf'] > 0.8]
        return f"Strong: {len(high)}, Weak: {len(self.patterns)-len(high)}"

if __name__ == "__main__":
    engine = InductiveEngine()
    for _ in range(10):
        engine.add_observation({'color': 'red'})
    engine.add_observation({'color': 'blue'})
    engine.infer_pattern('color')
    print(engine.generalize())
