# SAGE Tool Use SIAS (Sample-Importance-Aware Selection)

**Tool selection algorithm using sample-importance-aware selection for agent training and tool curation**

[![PyPI version](https://badge.fury.io/py/isage-agentic-tooluse-sias.svg)](https://badge.fury.io/py/isage-agentic-tooluse-sias)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

`sage-agentic-tooluse-sias` provides Sample-Importance-Aware Selection algorithms specifically designed for:

- **Tool Selection**: Select important tools for agent use
- **Agent Training**: Select important trajectories for fine-tuning
- **Continual Learning**: Efficient sample selection for continual/lifelong learning
- **Tool/Trajectory Curation**: Curate representative samples for agent development

## 📦 Installation

```bash
# Basic installation
pip install isage-agentic-tooluse-sias

# With PyTorch support
pip install isage-agentic-tooluse-sias[torch]

# Development installation
pip install isage-agentic-tooluse-sias[dev]
```

## 🚀 Quick Start

### Continual Learning

```python
from sage_sias import ContinualLearner

# Create continual learner
learner = ContinualLearner(
    buffer_size=1000,
    selection_strategy="importance"
)

# Add samples
for data, label in stream:
    learner.add_sample(data, label)

# Get selected samples
important_samples = learner.get_buffer()
```

### Coreset Selection

```python
from sage_sias import CoresetSelector

# Create coreset selector
selector = CoresetSelector(
    target_size=100,
    method="kmeans++"
)

# Select representative samples
coreset = selector.select(dataset, features)
```

## 📚 Key Components

### 1. **Continual Learner** (`continual_learner.py`)

Manages sample selection for continual learning:
- Buffer management with importance-based eviction
- Multiple selection strategies (random, importance, diversity)
- Support for experience replay

### 2. **Coreset Selector** (`coreset_selector.py`)

Selects representative subsets:
- K-means++ based selection
- Diversity-aware sampling
- Importance scoring
- Support for large-scale datasets

### 3. **Types** (`types.py`)

Common data types and protocols:
- Sample representation
- Importance scoring interfaces
- Selection strategies

## 🔧 Architecture

```
sage_sias/
├── continual_learner.py    # Continual learning with buffer management
├── coreset_selector.py      # Coreset selection algorithms
├── types.py                 # Common types and protocols
└── __init__.py             # Public API exports
```

## 🎓 Use Cases

1. **Agent Training**: Select important trajectories for fine-tuning
2. **Data Pruning**: Reduce dataset size while maintaining performance
3. **Active Learning**: Query most informative samples
4. **Memory Management**: Maintain representative samples in limited buffers
5. **Transfer Learning**: Select relevant samples for adaptation

## 🔗 Integration with SAGE

This package is part of the SAGE ecosystem but can be used independently:

```python
# Standalone usage
from sage_sias import ContinualLearner, CoresetSelector

# With SAGE agentic (optional)
from sage_agentic import AgentTrainer
from sage_sias import CoresetSelector

trainer = AgentTrainer()
selector = CoresetSelector(target_size=100)
important_trajectories = selector.select(all_trajectories)
trainer.train(important_trajectories)
```

## 📖 Documentation

- **Repository**: https://github.com/intellistream/sage-tooluse-sias
- **SAGE Documentation**: https://intellistream.github.io/SAGE-Pub/
- **Issues**: https://github.com/intellistream/sage-tooluse-sias/issues

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Originally part of the [SAGE](https://github.com/intellistream/SAGE) framework, now maintained as an independent package for broader community use.

## 📧 Contact

- **Team**: IntelliStream Team
- **Email**: shuhao_zhang@hust.edu.cn
- **GitHub**: https://github.com/intellistream
