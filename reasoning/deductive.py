"""Deductive reasoning engine for AIOS Brain.

Applies logical rules to derive conclusions from premises.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DeductiveEngine:
    """Deductive reasoning engine.
    
    Uses logical rules and premises to derive valid conclusions.
    """
    
    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self.premises: List[str] = []
        logger.info("Deductive reasoning engine initialized")
    
    def add_rule(self, rule_name: str, premise_conditions: List[str], 
                 conclusion: str) -> None:
        """Add a logical rule.
        
        Args:
            rule_name: Name of the rule
            premise_conditions: List of premise conditions
            conclusion: The conclusion if all conditions are met
        """
        try:
            rule = {
                'name': rule_name,
                'premises': premise_conditions,
                'conclusion': conclusion
            }
            self.rules.append(rule)
            logger.info(f"Added rule: {rule_name}")
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
    
    def add_premise(self, premise: str) -> None:
        """Add a known premise.
        
        Args:
            premise: The premise statement
        """
        try:
            self.premises.append(premise)
            logger.info(f"Added premise: {premise}")
        except Exception as e:
            logger.error(f"Error adding premise: {e}")
    
    def deduce(self) -> List[str]:
        """Apply deductive reasoning to derive conclusions.
        
        Returns:
            List of derived conclusions
        """
        try:
            conclusions = []
            for rule in self.rules:
                # Check if all premises of the rule are satisfied
                if all(premise in self.premises for premise in rule['premises']):
                    conclusions.append(rule['conclusion'])
                    logger.info(f"Deduced: {rule['conclusion']} via rule {rule['name']}")
            return conclusions
        except Exception as e:
            logger.error(f"Error during deduction: {e}")
            return []
    
    def get_reasoning_chain(self, conclusion: str) -> Optional[Dict[str, Any]]:
        """Get the reasoning chain for a conclusion.
        
        Args:
            conclusion: The conclusion to trace
            
        Returns:
            Dict with reasoning chain or None
        """
        try:
            for rule in self.rules:
                if rule['conclusion'] == conclusion:
                    return {
                        'conclusion': conclusion,
                        'rule': rule['name'],
                        'premises': rule['premises']
                    }
            return None
        except Exception as e:
            logger.error(f"Error getting reasoning chain: {e}")
            return None


if __name__ == "__main__":
    print("* " * 25)
    print("DEDUCTIVE REASONING DEMO")
    print("* " * 25)
    
    engine = DeductiveEngine()
    engine.add_premise("All humans are mortal")
    engine.add_premise("Socrates is human")
    engine.add_rule("modus_ponens", 
                   ["All humans are mortal", "Socrates is human"],
                   "Socrates is mortal")
    
    conclusions = engine.deduce()
    print(f"\nDeduced conclusions: {conclusions}")
    print(f"* " * 25)
