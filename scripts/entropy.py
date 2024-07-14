import nest
import numpy
from collections import Counter
import math

class EntropyAnalysis:
    def __init__(self, spike_trains):
        self.spike_trains = spike_trains

    def compute_entropy(self):
        spike_patterns = [''.join(map(str, train)) for train in self.spike_trains]
        pattern_counts = Counter(spike_patterns)
        total_patterns = len(spike_patterns)
        pattern_probabilities = {pattern: count / total_patterns for pattern, count in pattern_counts.items()}
        entropy = -sum(p * math.log2(p) for p in pattern_probabilities.values())
        return entropy
