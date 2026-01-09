"""Compatibility shim for SIAS."""

SIAS: Streaming Importance-Aware Agent System (internal reasoning/tool-selection)

SIAS has moved to `sage.libs.agentic.sias`. Importing from `sage.libs.sias`

remains supported but is deprecated.Purpose

"""-------

- Agent-side reasoning for **tool selection** and **sample importance**.

from __future__ import annotations- Provides coreset-based selection and continual learning primitives used by agent

  training and runtime decision components (e.g., choosing which tools/trajectories

import warnings  to keep or replay).

- Pure L3 library: no control-plane/gateway deps; fail fast on missing resources.

from sage.libs.agentic.sias import *  # noqa: F401,F403

Core Components

warnings.warn(---------------

    "`sage.libs.sias` has moved to `sage.libs.agentic.sias`. Update your imports.",- **CoresetSelector**: Importance-aware sample selection (keeps high-value items)

    DeprecationWarning,- **OnlineContinualLearner**: Experience replay with importance weighting

    stacklevel=2,- **StreamingImportanceScorer**: (TODO) SSIS prioritization for streaming traces

)

Usage
-----
    from sage.libs.sias import CoresetSelector, OnlineContinualLearner

    selector = CoresetSelector(strategy="hybrid")
    selected = selector.select(samples, target_size=1000)

    learner = OnlineContinualLearner(buffer_size=2048, replay_ratio=0.25)
    batch = learner.update_buffer(new_samples)

Future (planned extraction to independent repo)
----------------------------------------------
- StreamingImportanceScorer: I(x) = α·L_grad + β·D_ctx + γ·T_exec
- ReflectiveMemoryStore: Experience storage with pattern extraction
- AdaptiveExecutor: Pre/post verification and localized replanning
- MultiAgentRouter: Task decomposition and agent collaboration
"""

from .core import (
    CoresetSelector,
    OnlineContinualLearner,
    SelectionSummary,
)

__all__ = [
    # Core components (migrated from finetune/agent)
    "CoresetSelector",
    "OnlineContinualLearner",
    "SelectionSummary",
]

__version__ = "0.1.0"
