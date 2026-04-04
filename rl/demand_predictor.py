from collections import Counter
from typing import List, Dict, Any
from env.models import ObservationModel

class DemandPredictor:
    """Predicts future emergency hotspots based on frequency of past occurrences."""
    
    def __init__(self):
        self.history = []  # List of (timestamp, node)
        self.node_counts = Counter()

    def update(self, observation: ObservationModel):
        """Records emergency locations from the current observation."""
        timestamp = observation.step
        for emg in observation.emergencies:
            # Only record if it's a new emergency (or based on some logic)
            # In this simple implementation, we record all that are present
            # but usually we'd want to avoid double-counting.
            # Assuming we only record 'new' ones or just the distribution.
            self.history.append((timestamp, emg.node))
            self.node_counts[emg.node] += 1

    def predict(self, n: int = 5) -> List[int]:
        """Returns the top N likely nodes based on past frequency."""
        if not self.node_counts:
            return []
        
        # Get top N most common nodes
        top_n = self.node_counts.most_common(n)
        return [node for node, count in top_n]

    def get_hotspot_distribution(self) -> Dict[int, float]:
        """Returns the normalized frequency of each node."""
        total = sum(self.node_counts.values())
        if total == 0:
            return {}
        return {node: count / total for node, count in self.node_counts.items()}
