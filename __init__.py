"""SIAS - Streaming Importance-Aware Agent System.

Core components for agent tool selection and sample importance:
- CoresetSelector: Importance-aware sample selection
- OnlineContinualLearner: Experience replay with importance weighting
- SelectionSummary: Summary statistics for selection operations

Usage:
    from sage_agentic.sias import CoresetSelector, OnlineContinualLearner

    selector = CoresetSelector(strategy="hybrid")
    selected = selector.select(samples, target_size=1000)

    learner = OnlineContinualLearner(buffer_size=2048, replay_ratio=0.25)
    batch = learner.update_buffer(new_samples)
"""

from .coreset_selector import CoresetSelector, SelectionSummary
from .continual_learner import OnlineContinualLearner
from .types import ImportanceScore, SampleWithImportance

__all__ = [
    "CoresetSelector",
    "OnlineContinualLearner",
    "SelectionSummary",
    "ImportanceScore",
    "SampleWithImportance",
]

__version__ = "0.1.0"
