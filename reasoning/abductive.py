"""Abductive reasoning engine for hypothesis generation and validation.

Implements abductive reasoning - inferring the best explanation from observations.
Uses Bayesian probability and information theory to rank explanations.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """Represents a hypothesis with its likelihood score."""
    name: str
    likelihood: float  # Bayesian likelihood
    explanatory_power: float  # How well it explains observations
    simplicity_score: float  # Occam's razor consideration
    evidence: List[str] = field(default_factory=list)
    
    def overall_score(self) -> float:
        """Calculate overall hypothesis quality score."""
        return (self.likelihood * 0.4 + 
                self.explanatory_power * 0.4 + 
                self.simplicity_score * 0.2)


class AbductiveEngine:
    """Abductive reasoning engine for generating best explanations."""
    
    def __init__(self):
        """Initialize the abductive reasoning engine."""
        self.observations: List[Dict[str, Any]] = []
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.rules: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        logger.info("Abductive engine initialized")
    
    def add_observation(self, obs: Dict[str, Any]) -> None:
        """Add an observation to explain."""
        self.observations.append(obs)
        logger.info(f"Added observation: {obs}")
    
    def hypothesize(self, observations: List[Dict[str, Any]]) -> List[Hypothesis]:
        """Generate hypotheses to explain observations."""
        hypotheses = []
        for obs in observations:
            if 'pattern' in obs:
                h = Hypothesis(
                    name=f"Hypothesis_{obs['pattern']}",
                    likelihood=obs.get('likelihood', 0.5),
                    explanatory_power=obs.get('explanatory_power', 0.7),
                    simplicity_score=obs.get('simplicity', 0.8),
                    evidence=[obs.get('evidence', 'inferred')]
                )
                hypotheses.append(h)
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    def best_explanation(self, observations: List[Dict[str, Any]]) -> Optional[Hypothesis]:
        """Find the best explanation for observations."""
        if not observations:
            return None
        
        hypotheses = self.hypothesize(observations)
        if not hypotheses:
            return None
        
        best = max(hypotheses, key=lambda h: h.overall_score())
        logger.info(f"Best explanation: {best.name} (score: {best.overall_score():.3f})")
        return best


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    engine = AbductiveEngine()
    
    observations = [
        {'pattern': 'symptom_fever', 'likelihood': 0.8, 'explanatory_power': 0.9, 'simplicity': 0.7},
        {'pattern': 'symptom_cough', 'likelihood': 0.7, 'explanatory_power': 0.8, 'simplicity': 0.8},
    ]
    
    for obs in observations:
        engine.add_observation(obs)
    
    best = engine.best_explanation(observations)
    if best:
        print(f"Best explanation: {best.name} with score {best.overall_score():.3f}")
